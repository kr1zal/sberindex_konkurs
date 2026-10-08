"""Генератор главной страницы и данных стенда: `scripts/build_site.py`.

Сборка (единственная дорогая часть — чтение `per_series.csv` и матрицы панели, доли
секунды) идёт один раз в `setUpClass`, во временный каталог (`--out`), чтобы не трогать
закоммиченные `index.html`/`demo/data/`. Числа главной этот файл считает заново,
короткими выражениями pandas по тем же `results/*.csv` — независимо от генератора: тест,
вызывающий функции самого генератора, проверял бы только то, что код совпадает сам
с собой, а не то, что число на странице верное. То же — для данных `landing-data`
(медиана, ряды выборки, MAE горизонтов): сверка с матрицей и CSV, а не с самим собой.
"""
from __future__ import annotations

import ast
import contextlib
import datetime as dt
import importlib.util
import io
import json
import math
import re
import sys
import tempfile
import unittest
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_matrix, load_panel  # noqa: E402
from src.results_guard import read_results, refused  # noqa: E402

# scripts/ — не пакет: скрипт грузится по пути, как в tests/test_forecast_forward.py.
_spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
build_site = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_site)

INDEX_JSON_KEYS = {
    "built", "unit", "origin", "panel_months", "forecast_months", "n_series",
    "n_no_region", "n_homonym_names", "n_homonym_series", "n_hash_names", "n_hash_series",
    "default_mo", "mean_unit", "quick", "forecast_rule", "known_model", "breaks", "folds", "models",
    "panel_mae", "series",
}
AGGREGATE_JSON_KEYS = {
    "unit", "origin", "horizon", "model", "model_label",
    "history", "check", "rule_names", "mape",
}
LANDING_JSON_KEYS = {"story", "horizons", "fact", "teaser", "breaks"}
BREAKS_KEYS = {"months", "share", "top", "top_labels"}
STORY_KEYS = {"months", "base_months", "ids", "series", "median", "n_total", "n_sample", "decomposition",
              "spread_share", "spread_pct"}
HORIZONS_KEYS = {"list", "main", "labels", "unit", "models", "notes"}
TEASER_ITEM_KEYS = {"id", "short", "kind", "region", "fact", "forecast", "breaks", "baseline", "summary"}
TEASER_SUMMARY_KEYS = {"forecast_mean", "baseline_mean", "growth", "best", "reference_mae"}

_MONTH_OF = ["января", "февраля", "марта", "апреля", "мая", "июня",
             "июля", "августа", "сентября", "октября", "ноября", "декабря"]
_MONTH_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь",
              "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
_MONTH_IN = ["январе", "феврале", "марте", "апреле", "мае", "июне",
             "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"]
_ORDINAL = {0: "первом", 1: "втором", 2: "третьем"}


def _fmt(value: float, digits: int) -> str:
    """Число в формате отчёта — неразрывный пробел в разрядах, запятая, минус «−» —
    написанное заново, а не импортом из `build_site.py`: если у генератора здесь баг,
    независимая реализация его не повторит."""
    text = f"{abs(round(float(value), digits)):.{digits}f}"
    whole, _, frac = text.partition(".")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    grouped = " ".join(groups)
    sign = "−" if round(float(value), digits) < 0 else ""
    return sign + grouped + ("," + frac if frac else "")


def _fmt_signed(value: float, digits: int) -> str:
    """Как `_fmt`, но с «+» у положительных чисел (ноль — без знака): формат ошибок
    агрегата в demo/data/aggregate.json, независимый от build_site.num(..., sign=True)."""
    text = _fmt(value, digits)
    return f"+{text}" if round(float(value), digits) > 0 else text


def _fmt_rub_or_dash(value: float) -> str:
    """Как build_site.rub — целое число, неразрывный пробел в разрядах; пропуск (NaN) —
    «—», формат рублёвых строк агрегата (actual_rub/forecast_rub), независимый от
    build_site.rub/_rub_or_dash. Величины агрегата в этих данных неотрицательны, поэтому
    расхождение в знаке минуса (build_site.rub пишет ASCII-дефис, здесь — «−», как
    у остальных чисел этого файла) не проверяется — оно не встречается."""
    return "—" if not np.isfinite(value) else _fmt(value, digits=0)


def _written_forms(token: str) -> set[str]:
    """Как число могли бы вписать в код: как есть, с точкой вместо запятой и целое — ещё и
    с разрядами через обычный или неразрывный пробел."""
    forms = {token, token.replace(",", ".")}
    if token.isdigit() and len(token) >= 4:
        grouped = f"{int(token):,}"
        forms.update({grouped.replace(",", " "), grouped.replace(",", "\u00a0")})
    return forms


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


class _LinkCollector(HTMLParser):
    """Собирает значения href/src — независимо от того, как их резолвит браузер."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.links.append(value)


class _ResourceCollector(HTMLParser):
    """Ресурсы, которые браузер грузит сам (скрипты, стили, картинки, фреймы) — не обычные
    ссылки `<a href>`: те вправе вести куда угодно (репозиторий, дашборд СберИндекса)."""

    RESOURCE_ATTR = {"script": "src", "link": "href", "img": "src", "iframe": "src", "source": "src"}

    def __init__(self) -> None:
        super().__init__()
        self.resources: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_name = self.RESOURCE_ATTR.get(tag)
        if not attr_name:
            return
        value = dict(attrs).get(attr_name)
        if value:
            self.resources.append((tag, value))


class _StatsParser(HTMLParser):
    """Четыре пункта `<li class="stat">` полосы чисел на обложке, по порядку: текст `.stat-number` и подпись
    (`.stat-caption`) каждого — целиком, отдельно `.stat-label` (что за число) и `.stat-note` (строка оговорки).
    Связь «число ↔ его подпись» проверяется, только если число и текст сверяются в границах одного и того же
    пункта, а не «где-то в HTML»."""

    def __init__(self) -> None:
        super().__init__()
        self.stats: list[dict[str, str]] = []
        self._depth_in_li = 0
        self._current: dict[str, str] | None = None
        self._capture: str | None = None  # "number" | "caption" | None
        self._span: str | None = None  # "label" | "note" | None
        self._inner_spans = 0  # вложенные в подпись span (диапазон, который не рвётся) — тот же текст

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "li" and "stat" in classes:
            self._current = {"number": "", "caption": "", "label": "", "note": ""}
            self.stats.append(self._current)
            self._depth_in_li = 1
            return
        if self._current is None:
            return
        if tag == "li":
            self._depth_in_li += 1
        if tag == "p" and "stat-number" in classes:
            self._capture = "number"
        elif tag == "p" and "stat-caption" in classes:
            self._capture = "caption"
        elif tag == "span" and self._span:
            self._inner_spans += 1
        elif tag == "span" and "stat-label" in classes:
            self._span = "label"
        elif tag == "span" and "stat-note" in classes:
            self._span = "note"

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag == "span":
            if self._inner_spans:
                self._inner_spans -= 1
            else:
                self._span = None
        if tag == "p" and self._capture:
            self._capture = None
        if tag == "li":
            self._depth_in_li -= 1
            if self._depth_in_li <= 0:
                self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture:
            self._current[self._capture] += data
            if self._span:
                self._current[self._span] += data


class _AnchorCollector(HTMLParser):
    """Ссылки `<a>` страницы: адрес, target, rel и весь текст внутри — с вложенными знаками и скрытыми подписями."""

    def __init__(self) -> None:
        super().__init__()
        self.anchors: list[dict[str, str | None]] = []
        self._open: dict[str, str | None] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            values = dict(attrs)
            self._open = {"href": values.get("href"), "target": values.get("target"),
                          "rel": values.get("rel"), "text": ""}
            self.anchors.append(self._open)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._open = None

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._open["text"] += data

    def handle_entityref(self, name: str) -> None:
        if self._open is not None:
            self._open["text"] += {"nbsp": "\u00a0"}.get(name, f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._open is not None:
            self._open["text"] += chr(int(name[1:], 16) if name[:1] in "xX" else int(name))


class _TextCollector(HTMLParser):
    """Видимый читателю текст шаблона и тексты атрибутов, которые читают программы чтения
    с экрана и поисковики (`alt`, `aria-label`, `title`, `content` у description), — без
    содержимого `<script>` и `<style>`. Нужен проверке «вписанных руками чисел нет»."""

    TEXT_ATTRS = ("alt", "aria-label", "title")

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        values = dict(attrs)
        for name in self.TEXT_ATTRS:
            if values.get(name):
                self.chunks.append(values[name])
        if tag == "meta" and values.get("name") == "description" and values.get("content"):
            self.chunks.append(values["content"])

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.chunks.append(data)


class _IdCollector(HTMLParser):
    """Все `id` страницы и их теги — для проверки якорей и обязательных узлов разметки."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: dict[str, dict[str, str | None]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids[values["id"]] = {"tag": tag, **values}


class _NodeText(HTMLParser):
    """Видимый текст узлов по их `id` — вместе с вложенными тегами. Подстановки вне полосы чисел
    (подзаголовок обложки, шаг 01, блок «Прогноз по муниципалитету», заголовок раздела 03, ползунок) сверяются
    в границах своего узла: «где-то на странице» этим значениям не подходит — подмена одной
    подстановки другой (`n_series_rub` на `forecast_year`) оставалась бы незамеченной."""

    VOID = {"br", "hr", "img", "input", "link", "meta"}

    def __init__(self, ids) -> None:
        super().__init__()
        self.wanted = set(ids)
        self.text = {node_id: "" for node_id in ids}
        self._stack: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.VOID:
            return
        node_id = dict(attrs).get("id")
        self._stack.append((tag, node_id if node_id in self.wanted else None))

    def handle_endtag(self, tag: str) -> None:
        while self._stack:
            if self._stack.pop()[0] == tag:
                break

    def handle_data(self, data: str) -> None:
        for _, node_id in self._stack:
            if node_id:
                self.text[node_id] += data


class _IdsOf:
    """Тексты и атрибуты узлов страницы по `id`: `text(id)` — видимый текст с вложенными тегами,
    `attr(id, имя)` — значение атрибута."""

    def __init__(self, html: str) -> None:
        ids = _IdCollector()
        ids.feed(html)
        self._attrs = ids.ids
        nodes = _NodeText(ids.ids)
        nodes.feed(html)
        self._text = {node_id: _collapse(text) for node_id, text in nodes.text.items()}

    def text(self, node_id: str) -> str:
        return self._text[node_id]

    def attr(self, node_id: str, name: str) -> str:
        return self._attrs[node_id][name]


def _anchors(html: str) -> list[dict[str, str | None]]:
    """Все ссылки страницы с адресом, target, rel и текстом (см. `_AnchorCollector`)."""
    collector = _AnchorCollector()
    collector.feed(html)
    return collector.anchors


def _collapse(text: str) -> str:
    """Пробелы разметки — в один; неразрывный пробел остаётся (его пишут числа и подписи)."""
    return re.sub(r"[ \t\n\r\f\v]+", " ", text).strip()


# Числа словами в тексте страницы: основы и формы падежей, простым списком. Проверка ищет их
# в видимом тексте шаблона, где число обязано быть подстановкой генератора. «Обе» и «пол» сюда
# не входят — это не счёт.
_NUMBER_WORDS = re.compile(
    r"(?<!\w)(?:"
    r"од(?:ин|н\w*)|дв(?:а|е|ое)(?!\w)|дву\w*|двух\w*|двум\w*|двен\w*|двад\w*|двест\w*|"
    r"тр(?:и|[её]х\w*|ем\w*)(?!\w)|трин\w*|трид\w*|трист\w*|трет\w*|"
    r"четыр\w*|четвёрт\w*|четверт\w*|"
    r"пят\w*|шест\w*|сем(?:ь|и|ью|ьсот|ьдесят)(?!\w)|седьм\w*|семнад\w*|"
    r"восем\w*|восьм\w*|девят\w*|девян\w*|десят\w*|сорок\w*|"
    r"перв(?:ый|ого|ому|ым|ом|ая|ой|ую|ое|ые|ых|ыми)(?!\w)|"
    r"втор(?:ой|ого|ому|ым|ом|ая|ую|ое|ые|ых|ыми)(?!\w)|"
    r"сто(?!\w)|ста(?!\w)|сот(?:ня|ни|ен|ый|ого)\w*|тысяч\w*|миллион\w*|миллиард\w*|"
    r"нол[ьяюеи](?!\w)|нуль(?!\w)|нулев\w*|дважды|трижды|четырежды"
    r")\w*",
    re.IGNORECASE,
)

# То, что числом не является: имя сайта, фигура речи про сам вывод работы, один муниципалитет на
# линию графика, «Один из … крупно» (один пример, а не счёт), слово «миллионник» о городе, название
# этапа модели, прилагательное «двухэтапная» и отрицание «ни одной из моделей». Исключения — целыми
# оборотами, а не основами: «первый год» или «двух месяцев» они не скрывают.
_NOT_NUMBERS = (
    "Одно число",
    "одно число",
    "одного числа",
    "одно на всех",
    "один муниципалитет",
    "Один из",
    r"миллионник\w*",
    r"перв\w+ этап\w*",
    r"двухэтапн\w*",
    r"ни одн\w+",
)


def _number_words_in(text: str) -> list[str]:
    """Числа словами в тексте, кроме оборотов из `_NOT_NUMBERS`."""
    for phrase in _NOT_NUMBERS:
        text = re.sub(phrase, " ", text)
    return [match.group(0) for match in _NUMBER_WORDS.finditer(text)]


class BuildSiteTest(unittest.TestCase):
    """Сборка один раз на все проверки; данные для независимой сверки — тоже один раз."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls._tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            build_site.main(["--out", str(cls.out_dir)])

        cls.index_html_path = cls.out_dir / "index.html"
        cls.html = cls.index_html_path.read_text(encoding="utf-8")
        cls.data_dir = cls.out_dir / "demo" / "data"
        cls.index_json = json.loads((cls.data_dir / "index.json").read_text(encoding="utf-8"))
        cls.aggregate_json = json.loads((cls.data_dir / "aggregate.json").read_text(encoding="utf-8"))

        # Данные главной — JSON внутри <script id="landing-data">: берём ровно то, что получит
        # браузер, из готового index.html, а не из функций генератора.
        landing_scripts = re.findall(
            r'<script type="application/json" id="landing-data">(.*?)</script>', cls.html, re.S
        )
        cls.landing_raw = landing_scripts[0] if len(landing_scripts) == 1 else None
        cls.landing = json.loads(cls.landing_raw) if cls.landing_raw is not None else None

        # Четыре пункта <li class="stat"> по порядку — число, подпись и оговорка каждого отдельно,
        # без схлопывания разметки, но с нормализацией пробелов (отступы и переносы строк
        # шаблона иначе попали бы прямо в текст подписи).
        stats_parser = _StatsParser()
        stats_parser.feed(cls.html)
        # Только пробелы разметки схлопываются в один (отступы и переносы строк шаблона);
        # неразрывный пробел разрядов числа (rub()/num() генератора) — не обычный пробел
        # и не входит в этот класс символов, а str.split() без аргументов его тоже считает
        # пробелом и тем самым стёр бы разряды («1 581» → «1 581» уже с обычным пробелом,
        # не совпадающим с тем, что пишет _fmt ниже).
        collapse = lambda text: re.sub(r"[ \t\n\r\f\v]+", " ", text).strip()  # noqa: E731
        cls.stats = [
            {"number": s["number"].strip(), "caption": collapse(s["caption"]),
             "label": collapse(s["label"]), "note": collapse(s["note"])}
            for s in stats_parser.stats
        ]

        # demo/index.html — рукописный файл, генератор его не пишет (в отличие от
        # index.html и demo/data/**), поэтому читаем закоммиченный файл репозитория,
        # а не то, что build_site.main() положил во временный out_dir.
        cls.demo_html_path = ROOT / "demo" / "index.html"
        cls.demo_html = cls.demo_html_path.read_text(encoding="utf-8")

        # Источники чисел — те же results/*.csv и конфиги, что читает генератор,
        # но независимо: свои pandas-выражения, не вызов build_site.compute_placeholders.
        cls.full_cfg = yaml.safe_load((ROOT / "configs" / "full.yaml").read_text(encoding="utf-8"))
        cls.forward_cfg = yaml.safe_load(
            (ROOT / "configs" / "forecast_forward.yaml").read_text(encoding="utf-8")
        )
        cls.horizons_cfg = yaml.safe_load((ROOT / "configs" / "horizons.yaml").read_text(encoding="utf-8"))
        cls.summary = read_results(ROOT / "results" / "summary.csv", index_col=0)
        per_series = read_results(ROOT / "results" / "per_series.csv", dtype={"error": object})
        cls.ok = per_series[~refused(per_series)]
        cls.horizons_summary = read_results(ROOT / "results" / "horizons_summary.csv")
        cls.horizons_folds = read_results(ROOT / "results" / "horizons_folds.csv")
        cls.agg_check = read_results(ROOT / "results" / "forecast_2025_aggregate_check.csv")

        panel = load_panel(ROOT / cls.forward_cfg["data"]["path"])
        cls.wide, _ = build_matrix(
            panel, cls.forward_cfg["data"]["category"], max_gap=cls.forward_cfg["data"]["max_gap"]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # -- файлы и ключи -----------------------------------------------------

    def test_index_html_and_json_files_exist(self) -> None:
        self.assertTrue(self.index_html_path.exists())
        self.assertTrue((self.data_dir / "index.json").exists())
        self.assertTrue((self.data_dir / "aggregate.json").exists())
        self.assertTrue(any((self.data_dir / "mo").glob("*.json")))

    def test_index_json_top_level_keys(self) -> None:
        self.assertEqual(set(self.index_json), INDEX_JSON_KEYS)

    def test_hash_suffixed_series_counts_match_matrix_columns(self) -> None:
        # Подсказка поиска говорит «их различает номер после «#»» про ряды с суффиксом
        # " #N" в series_id — считаем их независимо, регулярным выражением над столбцами
        # матрицы, а не значением, которое положил генератор. У пяти названий вторая
        # копия выпала из панели по пропускам, и подсчёт по совпадению базового имени
        # (n_homonym_series/_names) их бы недосчитал.
        suffixed = [s for s in self.wide.columns if re.search(r" #\d+$", s)]
        names = {re.sub(r" #\d+$", "", s) for s in suffixed}
        self.assertEqual(self.index_json["n_hash_series"], len(suffixed))
        self.assertEqual(self.index_json["n_hash_names"], len(names))

    def test_series_without_region_carry_the_last_year_mean_as_a_distinguisher(self) -> None:
        # Одноимённые ряды без региона («Михайловский … #1 … #4») подсказка поиска иначе отличить не может:
        # пятый элемент записи — средние расходы за последние двенадцать месяцев панели, тыс. ₽ на человека
        # в месяц (единица и период — `mean_unit`), готовой строкой формата отчёта (пересчитано здесь по матрице
        # панели); у рядов с регионом записи остаются из четырёх элементов.
        last_year = self.wide.iloc[-12:].mean()
        without_region = 0
        for row in self.index_json["series"]:
            series_id, region = row[0], row[1]
            if region:
                self.assertEqual(len(row), 4, series_id)
                continue
            without_region += 1
            self.assertEqual(len(row), 5, series_id)
            self.assertEqual(row[4], _fmt(float(last_year[series_id]) / 1000, 1), series_id)
        self.assertEqual(without_region, self.index_json["n_no_region"])
        # Различитель различает: у одноимённых Михайловских районов значения разные.
        homonyms = [row[4] for row in self.index_json["series"]
                    if row[0].startswith("Михайловский муниципальный район #")]
        self.assertGreaterEqual(len(homonyms), 2)
        self.assertEqual(len(set(homonyms)), len(homonyms))

    def test_mean_unit_names_the_unit_and_the_period_of_the_distinguisher_from_the_panel(self) -> None:
        # «24,3» без единицы читалось бы как расходы всего района: единица — «на человека в месяц» — и период стоят
        # рядом с числом, и обе страницы показывают строку как есть. Период — из панели: последние двенадцать
        # месяцев — календарный год, иначе подпись называет месяцы (написано здесь заново, не вызовом генератора).
        last = self.wide.index[-12:]
        if last[0].month == 1 and last[-1].month == 12 and last[0].year == last[-1].year:
            period = f"{last[0].year} год"
        else:
            period = "последние 12 месяцев панели"
        self.assertEqual(self.index_json["mean_unit"], f"тыс.\u00a0₽ на человека в\u00a0месяц за\u00a0{period}")
        # Тот же период называют функции генератора на панелях, где «год» был бы неправдой.
        for start, expected in (("2023-01-01", "2024 год"), ("2023-03-01", "последние 12 месяцев панели")):
            with self.subTest(start=start):
                panel = pd.DataFrame(index=pd.date_range(start, periods=24, freq="MS"))
                self.assertEqual(build_site.mean_period_label(panel), expected)
        short = pd.DataFrame(index=pd.date_range("2024-01-01", periods=6, freq="MS"))
        self.assertEqual(build_site.mean_period_label(short), "последние 12 месяцев панели")

    def test_scripts_do_not_write_the_period_or_the_unit_of_the_distinguisher_by_hand(self) -> None:
        # Период различителя — «за последний год панели» — стоял в трёх строках скриптов и повторял константу
        # генератора: при другой панели строки остались бы прежними. Теперь он приходит из данных, и в коде страниц
        # (вне комментариев) ни его слов, ни единицы нет. Различитель показывает только страница прогноза (поле поиска
        # главной удалено), поэтому строку из данных берёт один её скрипт.
        for name in ("site/landing.js", "site/landing-lib.js", "demo/demo.js"):
            code = self._js_without_comments((ROOT / name).read_text(encoding="utf-8"))
            with self.subTest(file=name):
                for forbidden in ("последний год", "последнего года", "последние 12", "тыс. ₽", "тыс.\u00a0₽",
                                  "тыс.\\u00a0₽", "в мес.", "MEAN_HINT"):
                    self.assertNotIn(forbidden, code)
                if name == "demo/demo.js":
                    self.assertRegex(code, r"mean_unit|meanUnit")
                else:
                    self.assertNotRegex(code, r"mean_unit|meanUnit|regionLabel")

    def test_quick_buttons_point_at_panel_series_with_short_labels(self) -> None:
        # Быстрые кнопки стенда: каждая — ряд панели (иначе кнопка открыла бы МО по умолчанию
        # с сообщением «в данных стенда нет»), подписи короткие и не повторяются.
        quick = self.index_json["quick"]
        self.assertGreaterEqual(len(quick), 3)
        columns = set(self.wide.columns)
        for item in quick:
            with self.subTest(series=item["id"]):
                self.assertEqual(set(item), {"id", "short"})
                self.assertIn(item["id"], columns)
                self.assertTrue(item["short"].strip())
                self.assertLess(len(item["short"]), len(item["id"]))
        ids = [item["id"] for item in quick]
        self.assertEqual(len(set(ids)), len(ids), "ряд встречается среди кнопок дважды")
        shorts = {item["id"]: item["short"] for item in quick}
        self.assertEqual(len(set(shorts.values())), len(quick), "две кнопки с одной подписью")
        # Страница без ?mo= открывается на МО по умолчанию — его кнопка должна быть в списке.
        self.assertIn(self.index_json["default_mo"], ids)
        # В наборе есть и ряд с регионом, и ряд без региона, и ряд-омоним с номером «#N».
        regions = {row[0]: row[1] for row in self.index_json["series"]}
        self.assertTrue(any(regions[i] for i in ids), "нет ряда с регионом")
        self.assertTrue(any(not regions[i] for i in ids), "нет ряда без региона")
        self.assertTrue(any(re.search(r" #\d+$", i) for i in ids), "нет ряда с номером «#N»")
        # Общие с примерами главной ряды подписаны там так же.
        for series_id, short, _ in build_site.TEASER_MO:
            if series_id in shorts:
                self.assertEqual(shorts[series_id], short)

    def test_build_quick_rejects_series_missing_from_the_panel(self) -> None:
        with self.assertRaises(ValueError):
            build_site.build_quick([build_site.STAND_QUICK_MO[0][0]])
        self.assertEqual(build_site.build_quick(self.wide.columns), self.index_json["quick"])

    def test_aggregate_json_top_level_keys(self) -> None:
        self.assertEqual(set(self.aggregate_json), AGGREGATE_JSON_KEYS)

    def test_aggregate_history_starts_at_module_start_constant(self) -> None:
        # Как на графике отчёта (src/charts.py::aggregate_check, start="2022-01") —
        # не с первого месяца длинного ряда, иначе 2025 год прижат к правому краю.
        history = self.aggregate_json["history"]
        self.assertEqual(history["months"][0], build_site.AGGREGATE_CHART_START)
        origin = pd.Period(self.aggregate_json["origin"], "M")
        start = pd.Period(build_site.AGGREGATE_CHART_START, "M")
        self.assertEqual(history["months"][-1], str(origin))
        self.assertEqual(len(history["months"]), int((origin - start).n) + 1)

    def test_aggregate_model_label_matches_module_mapping(self) -> None:
        model_id = self.aggregate_json["model"]
        self.assertIn(model_id, build_site.AGGREGATE_MODEL_LABELS)
        self.assertEqual(self.aggregate_json["model_label"], build_site.AGGREGATE_MODEL_LABELS[model_id])

    def test_aggregate_error_pct_and_mape_are_report_formatted_strings(self) -> None:
        # check.error_pct, mape и рублёвые check.actual_rub/forecast_rub — уже готовые
        # строки: генератор форматирует их сам (num/rub), а JS показывает как есть, без
        # своего округления — иначе на одном и том же месяце стенд и отчёт могут
        # разойтись в последнем знаке («−3,9%» на стенде против «−4,0» в отчёте, если
        # JS ещё раз округляет уже округлённые питоном сотые). Сверяем со своими
        # форматтерами (_fmt_signed/_fmt/_fmt_rub_or_dash), посчитанными по исходному
        # CSV независимо от build_site.num/rub — те же принципы, что у остальных
        # тестов файла.
        horizon = max(int(h) for h in self.agg_check["horizon"].unique())
        year = self.agg_check.loc[self.agg_check["horizon"] == horizon]
        months = sorted(year["month"].unique())
        actual_by_month = year.groupby("month")["actual"].first().reindex(months)
        forecast_by_method = year.pivot(index="month", columns="method", values="forecast").reindex(months)
        error_by_method = year.pivot(index="month", columns="method", values="error_pct").reindex(months)
        abs_error = self.agg_check.assign(e=self.agg_check["error_pct"].abs())
        mape = abs_error.loc[abs_error["horizon"] == horizon].groupby("method")["e"].mean()

        for method in ("two_stage", "naive", "seasonal_naive"):
            with self.subTest(method=method):
                expected_series = [_fmt_signed(v, 1) for v in error_by_method[method]]
                self.assertEqual(self.aggregate_json["check"]["error_pct"][method], expected_series)
                self.assertEqual(self.aggregate_json["mape"][method], _fmt(mape[method], 1))

        expected_actual_rub = [_fmt_rub_or_dash(v) for v in actual_by_month]
        expected_forecast_rub = [_fmt_rub_or_dash(v) for v in forecast_by_method["two_stage"]]
        self.assertEqual(self.aggregate_json["check"]["actual_rub"], expected_actual_rub)
        self.assertEqual(self.aggregate_json["check"]["forecast_rub"]["two_stage"], expected_forecast_rub)

    def test_forecast_rule_label_matches_models_list(self) -> None:
        # Независимо от MODEL_LABELS генератора: label каждого отрезка forecast_rule
        # сверяется с label той же модели в models — если модель там есть, подписи
        # обязаны совпасть дословно (одно название и там, и там).
        models_by_id = {m["id"]: m["label"] for m in self.index_json["models"]}
        for segment in self.index_json["forecast_rule"]:
            with self.subTest(model=segment["model"], horizon=segment["horizon"]):
                expected = models_by_id.get(segment["model"], segment["model"])
                self.assertEqual(segment["label"], expected)

    def test_model_names_are_one_dictionary_for_both_pages(self) -> None:
        # «Лучшая панельная» на главной и «Лучшая в среднем по панели» на странице прогноза, «Эталон конкурса»
        # и «Эталон (Prophet)» — одни и те же модели под разными названиями. Теперь название задаёт роль, и словарь
        # один: у моделей, которые есть на обеих страницах, название и пояснение совпадают дословно.
        landing = {m["id"]: m for m in self.landing["horizons"]["models"]}
        stand = {m["id"]: m for m in self.index_json["models"]}
        shared = sorted(set(landing) & set(stand))
        self.assertGreaterEqual(len(shared), 3)
        for model_id in shared:
            with self.subTest(model=model_id):
                self.assertEqual(landing[model_id]["role"], stand[model_id]["role"])
                self.assertEqual(landing[model_id]["label"], stand[model_id]["name"])
                self.assertEqual(landing[model_id]["note"], stand[model_id]["note"])
        # Сами названия — написаны здесь заново, а не взяты из словаря генератора.
        best = self.summary["MAE"].idxmin()
        expected = {
            "prophet": ("Эталон конкурса", "Prophet по умолчанию"),
            "naive_last": ("Наивная", "оставить как в прошлом месяце"),
            "two_stage": ("Двухэтапная", "одно число и разнос долями"),
            best: ("Лучшая в среднем по панели", build_site.MODEL_LABELS[best]),
        }
        recommended = self.forward_cfg["recommended"][min(self.forward_cfg["recommended"])]
        if recommended not in expected:
            expected[recommended] = ("Рекомендуемая панельная", build_site.MODEL_LABELS[recommended])
        for model_id, (name, note) in expected.items():
            with self.subTest(expected=model_id):
                where = stand if model_id in stand else landing
                self.assertEqual(where[model_id]["name" if where is stand else "label"], name)
                self.assertEqual(where[model_id]["note"], note)
        # Страница прогноза не держит своего словаря: названия приходят из данных.
        demo_js = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        self.assertNotIn("ROLE_SHORT_LABELS", demo_js)
        self.assertIn("primary.textContent = model.name;", demo_js)
        self.assertIn("secondary.textContent = model.note;", demo_js)
        # Старых названий на главной нет.
        self.assertNotIn("Лучшая панельная", self.html)

    def test_model_names_take_the_role_not_the_identifier(self) -> None:
        self.assertEqual(build_site.model_names("two_stage", ["two_stage"]),
                         ("Двухэтапная", "одно число и разнос долями"))
        # Две роли сразу — одна строка, вторая роль со строчной; пояснение то же.
        self.assertEqual(build_site.model_names("two_stage", ["recommended", "two_stage"]),
                         ("Рекомендуемая панельная + двухэтапная", "одно число и разнос долями"))
        # У модели без записи в словаре названий — её идентификатор, но не падение.
        self.assertEqual(build_site.model_names("new_model", ["best_mean"]),
                         ("Лучшая в среднем по панели", "new_model"))
        with self.assertRaises(ValueError):
            build_site.model_names("prophet", ["champion"])

    def test_model_roles_match_docstring_rule(self) -> None:
        """`models` — пять ролей по правилу докстринга модуля (его начало): prophet —
        reference, naive_last — naive, recommended самого короткого горизонта —
        recommended, MAE.idxmin() сводки — best_mean, recommended самого длинного
        горизонта — two_stage; модель в двух ролях сразу — одной строкой, role через
        «+». Пересчитано здесь по тому же правилу своими выражениями, а не вызовом
        build_site._build_model_roles: проверяет, что список index.json следует
        правилу, а не что генератор согласен сам с собой."""
        recommended = self.forward_cfg["recommended"]
        shortest, longest = min(recommended), max(recommended)
        expected_roles: dict[str, list[str]] = {}
        for model_id, role in [
            ("prophet", "reference"),
            ("naive_last", "naive"),
            (recommended[shortest], "recommended"),
            (self.summary["MAE"].idxmin(), "best_mean"),
            (recommended[longest], "two_stage"),
        ]:
            expected_roles.setdefault(model_id, []).append(role)

        actual_by_id = {m["id"]: m["role"] for m in self.index_json["models"]}
        self.assertEqual(set(actual_by_id), set(expected_roles))
        for model_id, roles in expected_roles.items():
            with self.subTest(model=model_id):
                self.assertEqual(actual_by_id[model_id], "+".join(roles))

    def test_no_template_placeholders_left(self) -> None:
        self.assertNotIn("${", self.html)

    # -- полоса чисел на обложке: четыре числа, под каждым подпись и строка оговорки --------------

    def _cover_html(self) -> str:
        """Разметка обложки — от `<section class="hero night">` до её закрывающего `</section>` (вложенных секций
        в ней нет): всё, что тест считает «на обложке», лежит внутри этого тега, а не просто выше первого раздела."""
        start = self.html.index('<section class="hero night"')
        return self.html[start:self.html.index("</section>", start)]

    def _windows_won_and_lost(self) -> tuple[list, list, list]:
        """Окна проверки основного горизонта (фолды протокола): все, те, где лучшая модель точнее эталона, и те,
        где эталон точнее, — по per_series.csv, своими выражениями."""
        top = self.summary["MAE"].idxmin()
        fold_mae = (self.ok[self.ok["model"].isin([top, "prophet"])]
                    .groupby(["fold", "model"])["mae"].mean().unstack())
        folds = list(fold_mae.index)
        won = [f for f in folds if fold_mae.loc[f, top] < fold_mae.loc[f, "prophet"]]
        lost = [f for f in folds if fold_mae.loc[f, top] > fold_mae.loc[f, "prophet"]]
        return folds, won, lost

    def test_the_strip_has_four_numbers_inside_the_dark_cover_and_no_panel_line(self) -> None:
        # Полоса — у нижнего края тёмной обложки, а не белые карточки под ней; пятого числа (панели) и словаря
        # под карточками нет: число рядов уже стоит в подзаголовке.
        self.assertEqual(len(self.stats), 4)
        cover = self._cover_html()
        self.assertEqual(cover.count('<li class="stat'), 4)
        self.assertIn('<ul class="stats rise d5" aria-label="Ключевые числа работы">', cover)
        self.assertNotIn("stats-wrap", self.html)
        self.assertNotIn("stat-panel", self.html)
        self.assertNotIn("glossary", self.html)
        # Янтарным — только первое число.
        self.assertEqual(self.html.count("stat-lead"), 1)
        self.assertTrue(cover.index('<li class="stat stat-lead">') < cover.index("stat-number"))
        # У каждого пункта: число, подпись и одна строка оговорки; подпись целиком — это они двое.
        for stat in self.stats:
            with self.subTest(number=stat["number"]):
                self.assertTrue(stat["number"] and stat["label"] and stat["note"])
                self.assertEqual(stat["caption"], f"{stat['label']} {stat['note']}")

    def test_headline_stat_matches_summary_csv(self) -> None:
        # stat-number сверяется целиком, а подпись — в связке с горизонтом внутри ИМЕННО первого пункта
        # (self.stats[0]), а не где угодно в HTML: «3 месяца» из любой другой подписи число не подтвердило бы.
        top = self.summary["MAE"].idxmin()
        # Выигрыш без знака: на странице число стоит перед словом «точнее».
        best_gain = self.summary.loc[top, "к Prophet, %"]
        horizon_main = int(self.full_cfg["split"]["horizon"])

        stat = self.stats[0]
        self.assertEqual(stat["number"], f"{_fmt(best_gain, 1)}%")
        # Число и «месяца» — через неразрывный пробел: на 320 px подпись иначе переносится между ними («на 3 / месяца»).
        self.assertEqual(
            stat["label"],
            f"точнее эталона конкурса на {horizon_main}\u00a0{_plural(horizon_main, 'месяц', 'месяца', 'месяцев')}")
        self.assertRegex(stat["label"], r"\d\u00a0месяц")
        self.assertNotRegex(stat["label"], r"\d месяц")

    def test_windows_held_note_and_lost_windows_match_per_series_csv(self) -> None:
        folds, won, lost = self._windows_won_and_lost()
        # Оговорка к главному числу — строка того же пункта: в скольких окнах проверки выигрыш держится.
        if not won:
            held = "не держится ни в одном окне проверки"
        elif len(won) == len(folds):
            held = "держится во всех окнах проверки"
        else:
            held = f"держится в {len(won)} {'окне' if len(won) == 1 else 'окнах'} проверки из {len(folds)}"
        self.assertEqual(self.stats[0]["note"], held)
        # Какие окна эталон выиграл — в пояснении основного горизонта, рядом с полосами, где это видно по моделям.
        horizons = self.landing["horizons"]
        note = dict(zip(horizons["list"], horizons["notes"]))[horizons["main"]]
        ordinals = [{0: "первом", 1: "втором", 2: "третьем"}[f] for f in lost]
        self.assertTrue(lost, "в текущем прогоне эталон выигрывает хотя бы одно окно: проверке нужен такой случай")
        joined = ordinals[0] if len(ordinals) == 1 else ", ".join(ordinals[:-1]) + " и " + ordinals[-1]
        short_history = ", с самой короткой историей," if len(lost) == 1 and lost[0] == min(folds) else ""
        # Эталон точнее лучшей в среднем (MAE окна — выше у неё, это и есть условие `lost`), а не «точнее всех»:
        # в первом окне наивная и двухэтапная точнее эталона, и фраза без названия читалась бы неверно.
        sentence = (f"{'Во' if ordinals[0].startswith('вт') else 'В'} {joined} "
                    f"{'окне' if len(lost) == 1 else 'окнах'}{short_history} эталон точнее лучшей в среднем.")
        self.assertIn(sentence, note)
        # «Лучшая в среднем» — название третьей полосы того же раздела: читатель находит модель, о которой фраза.
        best_bar = self.landing["horizons"]["models"][2]
        self.assertEqual(best_bar["role"], "best_mean")
        self.assertTrue(best_bar["label"].startswith("Лучшая в среднем"))

    def test_the_landing_says_validation_window_and_never_fold(self) -> None:
        # «Фолд» — слово отчёта; на главной то же самое называется «окно проверки»: ни в тексте страницы, ни в данных
        # главной, ни в шаблоне слова нет. Страница прогноза (demo/) проверяется отдельно: слово отчёта стоит там один
        # раз, мостиком к отчёту (`test_demo_scripts.ErrorsCaptionTest`).
        template = (ROOT / "site" / "index.template.html").read_text(encoding="utf-8")
        for name, page in (("index.html", self.html), ("шаблон", template)):
            collector = _TextCollector()
            collector.feed(page)
            with self.subTest(page=name):
                self.assertNotRegex(" ".join(collector.chunks), r"(?i)фолд")
        self.assertNotRegex(self.landing_raw, r"(?i)фолд")
        # Проверка не слепа: слово заменено, а не вычеркнуто.
        self.assertIn("окнах проверки", self.stats[0]["note"])
        self.assertIn("окнам проверки", " ".join(self.landing["horizons"]["notes"]))

    # -- число 2: наукаст (горизонт 1) --------------------------------------

    def test_nowcast_stat_matches_horizons_summary_csv(self) -> None:
        top = self.summary["MAE"].idxmin()
        by_horizon_model = self.horizons_summary.set_index(["horizon", "model"])
        self.assertIn((1, top), by_horizon_model.index)
        h1_best = by_horizon_model.loc[(1, top), "MAE"]
        h1_prophet = by_horizon_model.loc[(1, "prophet"), "MAE"]
        h1_gain = by_horizon_model.loc[(1, top), "к Prophet, %"]

        stat = self.stats[1]
        self.assertEqual(stat["number"], f"{_fmt(h1_gain, 1)}%")
        self.assertEqual(stat["label"], "точнее эталона на текущий месяц")
        # Пара чисел — ошибки (MAE), а не расходы: без слова «ошибка» строку можно было прочесть как траты.
        # «₽» не отрывается от числа, «в месяц» держится вместе (так же, как единица различителя рядов страницы
        # прогноза): на 360 px строка иначе переносилась между «1 550» и «₽».
        self.assertEqual(
            stat["note"], f"ошибка {_fmt(h1_best, 0)} против {_fmt(h1_prophet, 0)}\u00a0₽ на человека в\u00a0месяц")

    # -- число 3: год вперёд (горизонт 12, без оракула) ---------------------

    def test_year_ahead_stat_matches_horizons_summary_and_folds_csv(self) -> None:
        year = self.horizons_summary[
            (self.horizons_summary["horizon"] == 12) & self.horizons_summary["MAE"].notna()
            & (self.horizons_summary["model"] != "two_stage_known")
        ].set_index("model")
        year_top = year["MAE"].idxmin()
        gain_prophet = year.loc[year_top, "к Prophet, %"]
        n_folds_top = int(self.horizons_folds.loc[
            (self.horizons_folds["horizon"] == 12) & (self.horizons_folds["model"] == year_top), "MAE"
        ].notna().sum())

        stat = self.stats[2]
        self.assertEqual(stat["number"], f"{_fmt(gain_prophet, 0)}%")
        self.assertEqual(stat["label"], "точнее эталона на год вперёд")
        # Оговорка — сколько окон проверки за числом, словом, и это именно оговорка, как в резюме отчёта
        # («но это один фолд»): «но это одно окно проверки» / «но это два окна проверки».
        self.assertLess(n_folds_top, 3, "при трёх окнах и больше оговорка другая: проверке нужен случай с одним-двумя")
        words = {1: "одно", 2: "два"}[n_folds_top]
        self.assertEqual(stat["note"], f"но это {words} {_plural(n_folds_top, 'окно', 'окна', 'окон')} проверки")

    # -- число 4: проверка агрегата по факту 2025 года -----------------------

    def test_aggregate_stat_matches_forecast_2025_aggregate_check_csv(self) -> None:
        abs_error = self.agg_check.assign(e=self.agg_check["error_pct"].abs())
        by_method_horizon = abs_error.groupby(["method", "horizon"])["e"].mean()
        own, rules = by_method_horizon.loc["two_stage"], by_method_horizon.drop(index="two_stage")

        def _range(series: pd.Series) -> str:
            lo, hi = series.min(), series.max()
            return _fmt(lo, 1) if _fmt(lo, 1) == _fmt(hi, 1) else f"{_fmt(lo, 1)}–{_fmt(hi, 1)}"

        origin = pd.Period(self.forward_cfg["origin"], "M")
        # «Год факта» страницы — тот же origin.year + 1, что и в отдельном годе прогноза
        # вперёд (число 3): здесь достаточно локального значения, привязанного к этому
        # же пункту подписи.
        forecast_year = origin.year + 1

        # Горизонты диапазона — те, по которым посчитаны сами числа (уровень «горизонт» их групп), а не из конфига.
        horizons = sorted(set(own.index) | set(rules.index.get_level_values("horizon")))
        self.assertEqual(sorted(own.index), horizons)
        stat = self.stats[3]
        self.assertEqual(stat["number"], f"{_range(own)}%")
        # «Средняя» и горизонты — как в резюме отчёта («в среднем на 3,8–4,5% (горизонты 1–12 мес.)»): без них
        # диапазон читался бы как помесячная ошибка, а в разделе 03 есть месяцы с ошибкой больше.
        self.assertEqual(stat["label"], f"средняя ошибка прогноза страны на факте {forecast_year}")
        self.assertEqual(
            stat["note"],
            f"горизонты {horizons[0]}–{horizons[-1]}\u00a0мес.; у\u00a0простых правил — {_range(rules)}%")
        # Диапазон — по горизонтам проверки: у простых правил он шире, чем у первого этапа.
        self.assertGreater(float(rules.max()), float(own.max()))

    # -- дата сборки -----------------------------------------------------------

    def test_built_date_is_today_in_iso_and_words(self) -> None:
        today = dt.date.today()
        self.assertEqual(self.index_json["built"], today.isoformat())
        self.assertIn(f"{today.day} {_MONTH_OF[today.month - 1]} {today.year}", self.html)

    # -- покрытие рядов панели -------------------------------------------------

    def test_all_matrix_series_are_in_index_json_series_list(self) -> None:
        listed = {row[0] for row in self.index_json["series"]}
        self.assertEqual(listed, set(self.wide.columns))

    def test_every_series_appears_exactly_once_across_mo_files(self) -> None:
        seen: dict[str, str] = {}
        for path in sorted((self.data_dir / "mo").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for series_id in payload:
                self.assertNotIn(
                    series_id, seen,
                    f"{series_id!r} встречается и в {seen.get(series_id)}, и в {path.name}",
                )
                seen[series_id] = path.name
        self.assertEqual(set(seen), set(self.wide.columns))

    def test_every_series_has_24_facts_12_forecasts_12_known(self) -> None:
        for path in sorted((self.data_dir / "mo").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for series_id, entry in payload.items():
                with self.subTest(series=series_id):
                    self.assertEqual(len(entry["fact"]), 24)
                    self.assertEqual(len(entry["forecast"]), 12)
                    self.assertEqual(len(entry["known"]), 12)

    def test_selected_mo_data_matches_independent_recomputation(self) -> None:
        """DEFAULT_MO и один из одноимённых рядов («Михайловский муниципальный район #2») —
        forecast, known, breaks и mae по фолдам пересчитаны здесь заново по
        forecast_2025.csv, cp_offline_series.csv и per_series.csv, своими выражениями,
        а не вызовом build_site._steps_frame/_build_forecast_rule/_load_breaks: тест,
        зовущий функции генератора, проверял бы только то, что код согласен сам с собой,
        а не что demo/data/mo/*.json верны."""
        targets = [build_site.DEFAULT_MO, "Михайловский муниципальный район #2"]

        forecast = read_results(ROOT / "results" / "forecast_2025.csv")
        self.assertTrue(
            set(targets) <= set(forecast["series_id"]),
            "нет одного из проверяемых рядов в forecast_2025.csv",
        )

        origin = pd.Period(self.forward_cfg["origin"], "M")
        # Ключи recommended — тот же источник истины, что у генератора
        # (build_site.build_demo_data) и у отчёта (report.qmd::_fc_steps), а не
        # forward_cfg["horizons"] отдельным списком: сейчас они совпадают, но именно
        # recommended отвечает и за прогноз, и за пунктир.
        horizons = sorted(self.forward_cfg["recommended"])
        # Шаг → наименьший горизонт, который его покрывает — правило report.qmd::_fc_steps,
        # написанное заново, а не импортом build_site._step_horizon_map.
        step_horizon = {step: min(h for h in horizons if h >= step) for step in range(1, horizons[-1] + 1)}

        cp_cfg = yaml.safe_load((ROOT / "configs" / "changepoints.yaml").read_text(encoding="utf-8"))
        protocol = f"v{cp_cfg['protocol_version']}"
        # PENALTY — константа scripts/news_event_study.py; грузим модуль по пути тем же
        # способом, что build_site._load_penalty (scripts/ — не пакет), но не вызывая
        # саму функцию генератора: если она возьмёт не ту константу, тест должен
        # остаться независимым от этой её ошибки, а не унаследовать её.
        news_event_study_spec = importlib.util.spec_from_file_location(
            "news_event_study", ROOT / "scripts" / "news_event_study.py"
        )
        news_event_study = importlib.util.module_from_spec(news_event_study_spec)
        news_event_study_spec.loader.exec_module(news_event_study)
        penalty = float(news_event_study.PENALTY)
        offline = read_results(ROOT / "results" / "cp_offline_series.csv")
        offline_hits = offline.loc[(offline["protocol"] == protocol) & np.isclose(offline["penalty"], penalty)]

        n_folds = int(self.full_cfg["split"]["n_folds"])
        # Список id и ролей моделей сверяет test_model_roles_match_docstring_rule; здесь
        # он уже готов — берём его из index.json и сверяем только значения MAE по нему.
        model_ids = [m["id"] for m in self.index_json["models"]]

        mo_files: dict[str, dict] = {}
        for path in (self.data_dir / "mo").glob("*.json"):
            mo_files.update(json.loads(path.read_text(encoding="utf-8")))

        for series_id in targets:
            with self.subTest(series=series_id):
                rows = forecast.loc[forecast["series_id"] == series_id]

                expected_forecast = []
                expected_known = []
                for step in range(1, horizons[-1] + 1):
                    month = str(origin + step)
                    horizon = step_horizon[step]
                    rec = rows.loc[
                        rows["recommended"] & (rows["horizon"] == horizon) & (rows["month"] == month), "forecast"
                    ]
                    known = rows.loc[
                        rows["uses_published_aggregate"] & (rows["horizon"] == horizon) & (rows["month"] == month),
                        "forecast",
                    ]
                    expected_forecast.append(int(round(float(rec.iloc[0]))) if len(rec) else None)
                    expected_known.append(int(round(float(known.iloc[0]))) if len(known) else None)

                expected_breaks = sorted(offline_hits.loc[offline_hits["series_id"] == series_id, "month"])

                per_series_rows = self.ok.loc[self.ok["mo"] == series_id]
                expected_mae = {}
                for model_id in model_ids:
                    model_rows = per_series_rows.loc[per_series_rows["model"] == model_id].set_index("fold")["mae"]
                    expected_mae[model_id] = [
                        int(round(float(model_rows[f]))) if f in model_rows.index and pd.notna(model_rows[f])
                        else None
                        for f in range(n_folds)
                    ]

                actual = mo_files[series_id]
                self.assertEqual(actual["forecast"], expected_forecast)
                self.assertEqual(actual["known"], expected_known)
                self.assertEqual(actual["breaks"], expected_breaks)
                for model_id in model_ids:
                    with self.subTest(model=model_id):
                        self.assertEqual(actual["mae"][model_id], expected_mae[model_id])

    # -- размер данных стенда ----------------------------------------------

    def test_demo_data_size_is_at_or_under_the_module_threshold(self) -> None:
        total = sum(p.stat().st_size for p in self.data_dir.rglob("*") if p.is_file())
        self.assertLessEqual(total, build_site.MAX_DEMO_BYTES)

    # -- ссылки страницы входа -----------------------------------------------

    def test_relative_links_resolve_to_existing_repository_files(self) -> None:
        collector = _LinkCollector()
        collector.feed(self.html)
        self.assertTrue(collector.links, "в index.html не нашлось ни одной ссылки")

        ids = _IdCollector()
        ids.feed(self.html)
        for link in collector.links:
            if link.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            with self.subTest(link=link):
                if link.startswith("#"):
                    # Якорь внутри страницы — это не файл: ищем элемент с таким id.
                    self.assertIn(link[1:], ids.ids, f"{link}: на странице нет такого id")
                    continue
                path_part, _, fragment = link.partition("#")
                target = (ROOT / unquote(path_part)).resolve()
                if target.is_dir():
                    self.assertTrue((target / "index.html").exists(), f"{link}: нет index.html внутри")
                else:
                    self.assertTrue(target.exists(), f"{link}: файла {target} нет")
                if fragment:
                    # Ссылка на раздел другой страницы («report/report.html#…»): такой id там должен быть.
                    page_ids = _IdCollector()
                    page_ids.feed(target.read_text(encoding="utf-8"))
                    self.assertIn(unquote(fragment), page_ids.ids, f"{link}: в {target.name} нет такого id")

    # -- ссылки и ресурсы стенда (demo/index.html) ----------------------------

    def test_demo_index_html_relative_links_resolve_to_existing_files(self) -> None:
        collector = _LinkCollector()
        collector.feed(self.demo_html)
        self.assertTrue(collector.links, "в demo/index.html не нашлось ни одной ссылки")

        own_ids = _IdCollector()
        own_ids.feed(self.demo_html)
        for link in collector.links:
            if link.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            with self.subTest(link=link):
                path_part, _, fragment = link.partition("#")
                if not path_part:
                    # Якорь внутри самого стенда («К содержанию»): ищем элемент с таким id.
                    self.assertIn(fragment, own_ids.ids, f"{link}: на странице нет такого id")
                    continue
                # demo/index.html лежит в demo/, относительные ссылки — от этой папки
                # (а не от корня репозитория, как у index.html на верхнем уровне).
                target = (self.demo_html_path.parent / path_part).resolve()
                if target.is_dir():
                    target_page = target / "index.html"
                    self.assertTrue(target_page.exists(), f"{link}: нет index.html внутри")
                else:
                    target_page = target
                    self.assertTrue(target.exists(), f"{link}: файла {target} нет")
                if fragment:
                    # Ссылка на раздел другой страницы («../#fact» — «Проверка фактом» на
                    # главной): раздел с таким id там должен быть.
                    ids = _IdCollector()
                    ids.feed(target_page.read_text(encoding="utf-8"))
                    self.assertIn(fragment, ids.ids, f"{link}: в {target_page.name} нет такого id")

    def test_no_page_loads_external_scripts_or_stylesheets(self) -> None:
        # Ни одна страница не грузит внешних скриптов и стилей — ни index.html
        # (страница входа), ни demo/index.html (стенд). Обычные ссылки `<a href>`
        # (репозиторий, дашборд СберИндекса) сюда не относятся — только script/link.
        for page, html in (("index.html", self.html), ("demo/index.html", self.demo_html)):
            collector = _ResourceCollector()
            collector.feed(html)
            self.assertTrue(collector.resources, f"{page}: не нашлось ни одного script/link")
            for tag, value in collector.resources:
                with self.subTest(page=page, tag=tag, value=value):
                    self.assertFalse(
                        value.startswith(("http://", "https://", "//")),
                        f"{page}: внешний ресурс <{tag}> -> {value}",
                    )

    # -- стенд: разметка, скрипт и стили -----------------------------------------------------

    def test_demo_markup_provides_every_element_id_the_script_reads(self) -> None:
        js = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        wanted = set(re.findall(r'getElementById\("([^"]+)"\)', js))
        self.assertGreater(len(wanted), 15, "проверка потеряла обращения скрипта к странице")
        ids = _IdCollector()
        ids.feed(self.demo_html)
        self.assertEqual(sorted(wanted - set(ids.ids)), [], "скрипт ждёт элементов, которых нет в разметке")

    def test_demo_page_has_landmarks_accessible_search_and_site_links(self) -> None:
        ids = _IdCollector()
        ids.feed(self.demo_html)
        links = _LinkCollector()
        links.feed(self.demo_html)
        for fragment in ('lang="ru"', 'name="viewport"', 'name="description"',
                         '<link rel="icon" href="../favicon.ico" sizes="32x32">',
                         '<link rel="icon" href="../favicon.svg" type="image/svg+xml">'):
            self.assertIn(fragment, self.demo_html)
        self.assertEqual(ids.ids["main"]["tag"], "main")
        self.assertIn("#main", links.links, "нет ссылки «К содержанию»")
        # Поле поиска — ARIA-комбобокс со списком и живой областью; быстрые кнопки — группа
        # с названием, сообщение об ошибке — alert.
        combo = ids.ids["mo-search"]
        self.assertEqual(combo["role"], "combobox")
        self.assertEqual(combo["aria-controls"], "mo-listbox")
        self.assertEqual(ids.ids["mo-listbox"]["role"], "listbox")
        self.assertEqual(ids.ids["mo-search-status"]["aria-live"], "polite")
        self.assertEqual(ids.ids["quick-chips"]["role"], "group")
        self.assertTrue(ids.ids["quick-chips"]["aria-label"])
        self.assertEqual(ids.ids["mo-invalid-note"]["role"], "alert")
        # Часть того же сайта: общая тема и шрифты, шапка с теми же ссылками, что на главной,
        # и переход к интерактивной версии блока агрегата («Проверка фактом»).
        for link in ("../site/site.css", "demo.css", "linechart.js", "demo.js", "../", "../report/report.html",
                     "https://github.com/kr1zal/sberindex_konkurs", "../#fact"):
            with self.subTest(link=link):
                self.assertIn(link, links.links)
        for licence in ("unbounded", "onest", "jetbrains-mono"):
            self.assertIn(f"../site/fonts/{licence}-OFL.txt", links.links)

    def test_demo_page_has_no_hand_typed_numbers(self) -> None:
        # Как в шаблоне главной: число на странице приходит из данных (его подставляет скрипт),
        # в разметке цифр нет. Исключение — название лицензии «CC BY-SA 4.0».
        collector = _TextCollector()
        collector.feed(self.demo_html)
        text = " ".join(collector.chunks).replace("CC BY-SA 4.0", " ")
        leftovers = re.findall(r".{0,20}\d.{0,20}", text)
        self.assertEqual(leftovers, [], "в разметке стенда вписаны числа; их должен подставлять скрипт из данных")

    def test_demo_scripts_have_no_hand_typed_data_numbers(self) -> None:
        # Те же «данные-подобные» числа, что проверяются у скрипта главной: с десятичной запятой,
        # с неразрывным пробелом разрядов или из четырёх и более цифр — число рядов, проценты
        # агрегата, средние ошибки моделей по панели. Короткие целые («8» в лимите подсказок)
        # совпали бы с константами кода и только шумели бы.
        tokens: set[str] = {str(self.index_json["n_series"])}
        tokens.update(self.aggregate_json["mape"].values())
        for row in self.index_json["panel_mae"].values():
            tokens.update(str(v) for v in row if v is not None)

        def data_like(token: str) -> bool:
            return "," in token or "\u00a0" in token or len(re.sub(r"\D", "", token)) >= 4

        forbidden = sorted(t for t in tokens if data_like(t))
        self.assertGreater(len(forbidden), 8, "проверка потеряла числа страницы")
        for name in ("demo/demo.js", "demo/linechart.js"):
            js = (ROOT / name).read_text(encoding="utf-8")
            for token in forbidden:
                for variant in _written_forms(token):
                    with self.subTest(file=name, number=variant):
                        self.assertIsNone(
                            re.search(r"(?<![\w.,])" + re.escape(variant) + r"(?![\w.,])", js),
                            f"в {name} вписано число данных {variant!r}",
                        )
            # И сами ряды: массив из шести и более чисел подряд — вписанные данные.
            self.assertIsNone(
                re.search(r"\[\s*-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*){5,}\]", js),
                f"в {name} есть числовой массив — данные должны приходить из demo/data",
            )

    def test_demo_chart_uses_only_theme_tokens_of_site_css(self) -> None:
        # Цвета рядов графика у стенда и главной одни и те же: стенд не держит своих копий --chart-*,
        # а каждый токен, на который он ссылается, объявлен в site.css дважды — для светлой темы
        # (тема одна) и для ночных блоков (.night).
        demo_css = (ROOT / "demo" / "demo.css").read_text(encoding="utf-8")
        self.assertNotRegex(demo_css, r"--chart-[a-z-]+\s*:", "demo.css объявляет свой токен графика")
        site_css = (ROOT / "site" / "site.css").read_text(encoding="utf-8")
        sources = demo_css + (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        used = set(re.findall(r"var\((--chart-[a-z-]+)\)", sources))
        self.assertTrue({"--chart-fact", "--chart-forecast", "--chart-known", "--chart-grid"} <= used)
        for token in sorted(used):
            with self.subTest(token=token):
                self.assertEqual(len(re.findall(re.escape(token) + r"\s*:", site_css)), 2,
                                 f"{token} должен быть объявлен для светлой темы и .night")

    # -- данные главной: <script id="landing-data"> --------------------------------------

    def test_landing_data_is_one_parsable_json_script(self) -> None:
        self.assertIsNotNone(self.landing_raw, "в index.html нет ровно одного <script id=landing-data>")
        # «<» экранирован: иначе «</script» внутри строки данных закрыл бы тег.
        self.assertNotIn("<", self.landing_raw)
        self.assertEqual(set(self.landing), LANDING_JSON_KEYS)
        self.assertEqual(set(self.landing["story"]), STORY_KEYS)
        self.assertEqual(set(self.landing["horizons"]), HORIZONS_KEYS)
        self.assertEqual(set(self.landing["teaser"]), {"months", "items", "reference"})
        self.assertEqual(set(self.landing["breaks"]), BREAKS_KEYS)

    def test_landing_json_size_is_under_the_module_threshold(self) -> None:
        size = len(self.landing_raw.encode("utf-8"))
        self.assertLessEqual(size, build_site.MAX_LANDING_BYTES)

    def test_landing_json_escapes_angle_brackets_and_round_trips(self) -> None:
        payload = {"name": "</script><!-- x -->", "n": [1.5, None]}
        text = build_site.landing_json(payload)
        self.assertNotIn("<", text)
        self.assertEqual(json.loads(text), payload)

    def test_landing_json_over_threshold_raises(self) -> None:
        too_big = {"x": "я" * build_site.MAX_LANDING_BYTES}
        with self.assertRaises(ValueError):
            build_site.landing_json(too_big)

    # -- story: ряды выборки и медиана по ВСЕМ рядам -----------------------------------------

    def _story_norm(self) -> pd.DataFrame:
        """Ряды матрицы, делённые на среднее своих первых 12 месяцев — написано здесь заново."""
        return self.wide / self.wide.iloc[:12].mean()

    def test_story_decomposition_recomputed_from_csv(self) -> None:
        # Четвёртый шаг: у каждой модели промах и разброс — средние по окнам проверки из
        # results/error_decomposition.csv, целыми рублями, модели по убыванию суммы (MAE); название — из словаря
        # моделей без пояснения; горизонт — основного протокола.
        decomposition = read_results(ROOT / "results" / "error_decomposition.csv")
        means = decomposition.groupby("модель", sort=False)[["смещение", "разброс", "MAE"]].mean()
        got = self.landing["story"]["decomposition"]
        self.assertEqual(got["horizon"], int(self.full_cfg["split"]["horizon"]))
        self.assertEqual(set(m["id"] for m in got["models"]), set(means.index))
        totals = [m["bias"] + m["spread"] for m in got["models"]]
        self.assertEqual(totals, sorted(totals, reverse=True))
        for model in got["models"]:
            with self.subTest(model=model["id"]):
                row = means.loc[model["id"]]
                self.assertEqual(model["bias"], round(float(row["смещение"])))
                self.assertEqual(model["spread"], round(float(row["разброс"])))
                # Смещение и разброс складываются в MAE (до округления рубля на каждом слагаемом).
                self.assertLessEqual(abs(model["bias"] + model["spread"] - float(row["MAE"])), 1.5)
                label = build_site.MODEL_LABELS[model["id"]]
                head = label.split(": ")[0] if ": " in label else label.split(" (")[0]
                self.assertEqual(model["label"], head[:1].upper() + head[1:])

    def test_story_median_is_recomputed_over_all_matrix_series(self) -> None:
        story = self.landing["story"]
        median = self._story_norm().median(axis=1)
        self.assertEqual(len(story["median"]), len(self.wide))
        for month, got, expected in zip(story["months"], story["median"], median):
            with self.subTest(month=month):
                # Три знака после запятой — округление генератора: расхождение не больше полу-единицы
                # последнего знака (плюс запас на двоичное представление).
                self.assertAlmostEqual(got, float(expected), delta=0.0005 + 1e-9)
        # Тест чувствителен к ошибке «медиана только по выборке»: медиана 30 рядов расходится
        # с медианой по всей панели больше, чем округление.
        sample_median = self._story_norm()[story["ids"]].median(axis=1)
        self.assertGreater(float((sample_median - median).abs().max()), 0.0005)

    def test_story_counts_and_months_match_the_matrix(self) -> None:
        story = self.landing["story"]
        self.assertEqual(story["n_total"], self.wide.shape[1])
        self.assertEqual(story["n_sample"], len(story["ids"]))
        self.assertEqual(story["n_sample"], build_site.LANDING_SAMPLE_SIZE)
        self.assertEqual(story["months"], [m.strftime("%Y-%m") for m in self.wide.index])
        self.assertEqual(story["base_months"], 12)

    def test_story_sample_series_are_matrix_columns_with_normalised_values(self) -> None:
        story = self.landing["story"]
        self.assertEqual(len(set(story["ids"])), len(story["ids"]), "в выборке повторяются ряды")
        self.assertEqual(len(story["series"]), len(story["ids"]))
        norm = self._story_norm()
        for series_id, row in zip(story["ids"], story["series"]):
            with self.subTest(series=series_id):
                self.assertIn(series_id, norm.columns)
                self.assertEqual(len(row), len(self.wide))
                for got, expected in zip(row, norm[series_id]):
                    self.assertAlmostEqual(got, float(expected), delta=0.0005 + 1e-9)

    def test_story_sample_is_fixed_by_the_module_seed(self) -> None:
        # Сид — константа модуля: та же выборка при повторной сборке, другой сид — другая.
        decomposition = read_results(ROOT / "results" / "error_decomposition.csv")
        again = build_site.build_story(self.wide, decomposition, int(self.full_cfg["split"]["horizon"]))
        self.assertEqual(again["ids"], self.landing["story"]["ids"])
        other = np.random.default_rng(build_site.LANDING_SAMPLE_SEED + 1).choice(
            self.wide.shape[1], size=build_site.LANDING_SAMPLE_SIZE, replace=False
        )
        self.assertNotEqual([str(self.wide.columns[i]) for i in other], self.landing["story"]["ids"])

    def test_story_spread_in_template_is_recomputed_from_all_series(self) -> None:
        # Строка третьего шага: «не менее чем у N% значений … не больше ±M%» — по всей панели. M — процентиль
        # отклонений, округлённый вверх: при 7,03% полоса ±7% вмещала бы меньше заявленной доли.
        norm = self._story_norm()
        deviation = norm.div(norm.median(axis=1), axis=0).sub(1).abs().to_numpy()
        share = build_site.STORY_SPREAD_SHARE
        self.assertTrue(50 < share < 100)
        band = math.ceil(round(float(np.percentile(deviation, share)) * 100, 6))
        line = _IdsOf(self.html).text("step-3")
        # Ряды поделены на общее движение: полоса ±M% вокруг линии на графике, и без этих слов её не от чего считать.
        self.assertIn(f"Ряды, делённые на общее движение: не менее чем у {share}% значений по всей панели", line)
        self.assertIn(f"отклонение не больше ±{band}%", line)
        # Те же числа — в данных главной: по ним скрипт рисует полосу третьего шага.
        self.assertEqual(self.landing["story"]["spread_share"], share)
        self.assertEqual(self.landing["story"]["spread_pct"], band)
        # Утверждение верно, и полоса не шире нужного: на один процент уже она бы доли не вмещала.
        self.assertGreaterEqual(float((deviation <= band / 100).mean()), share / 100)
        self.assertLess(float((deviation <= (band - 1) / 100).mean()), share / 100)

    # -- horizons: MAE четырёх моделей на горизонтах конфига --------------------------------

    def test_horizons_match_horizons_summary_csv(self) -> None:
        horizons = self.landing["horizons"]
        expected_list = sorted(int(item["horizon"]) for item in self.horizons_cfg["horizons"])
        self.assertEqual(horizons["list"], expected_list)
        self.assertEqual(horizons["main"], int(self.full_cfg["split"]["horizon"]))
        self.assertEqual(len(horizons["labels"]), len(expected_list))
        self.assertEqual(len(horizons["notes"]), len(expected_list))

        best = self.summary["MAE"].idxmin()
        self.assertEqual([m["id"] for m in horizons["models"]], ["prophet", "naive_last", best, "two_stage"])
        table = self.horizons_summary.set_index(["horizon", "model"])["MAE"]
        for model in horizons["models"]:
            expected = []
            for horizon in expected_list:
                value = table.get((horizon, model["id"]), np.nan)
                expected.append(int(round(float(value))) if pd.notna(value) else None)
            with self.subTest(model=model["id"]):
                self.assertEqual(model["mae"], expected)
                self.assertTrue(model["label"] and model["note"], "у модели нет подписи или пояснения")

    def test_horizons_missing_value_is_null_not_zero(self) -> None:
        # Лучшая модель основного протокола на годовом горизонте не обучается (MAE пуст в CSV):
        # в данных — null, а не 0, иначе полоса вышла бы «лучшей».
        year = self.horizons_summary[self.horizons_summary["horizon"] == 12].set_index("model")["MAE"]
        best = self.summary["MAE"].idxmin()
        index = self.landing["horizons"]["list"].index(12)
        model = next(m for m in self.landing["horizons"]["models"] if m["id"] == best)
        if pd.isna(year.get(best, np.nan)):
            self.assertIsNone(model["mae"][index])
        else:
            self.assertEqual(model["mae"][index], int(round(float(year[best]))))

    def test_horizons_notes_are_one_phrase_with_the_number_of_validation_windows(self) -> None:
        horizons = self.landing["horizons"]
        notes = dict(zip(horizons["list"], horizons["notes"]))
        main = horizons["main"]
        # Число окон проверки каждого горизонта — словом, в одной фразе под полосами, из двух независимых источников:
        # конфига и результатов. Среднее по двум окнам и по девяти — разные по весу утверждения.
        config_folds = {int(item["horizon"]): int(item["n_folds"]) for item in self.horizons_cfg["horizons"]}
        result_folds = self.horizons_folds.groupby("horizon")["fold"].nunique()
        self.assertEqual(sorted(config_folds), horizons["list"])
        dative = {1: "одному окну", 2: "двум окнам", 3: "трём окнам", 6: "шести окнам", 9: "девяти окнам"}
        for horizon, note in notes.items():
            with self.subTest(horizon=horizon):
                self.assertEqual(int(result_folds[horizon]), config_folds[horizon])
                self.assertIn(f"Средняя ошибка, ₽ на человека в месяц, по {dative[config_folds[horizon]]} проверки.",
                              note)
                self.assertNotRegex(note, r"\d")
        # Наукаст назван прогнозом текущего месяца: слово «Наукаст» на переключателе иначе нигде не объяснено.
        self.assertTrue(notes[1].startswith("Наукаст — прогноз текущего месяца, пока его данных ещё нет."))
        self.assertTrue(all("Наукаст" not in note for horizon, note in notes.items() if horizon != 1))
        # Основной горизонт открыт по умолчанию: где эталон выиграл окно, сказано рядом с полосами (число окон —
        # в оговорке полосы чисел, а какое окно — здесь); у других горизонтов этой фразы нет.
        self.assertIn("эталон точнее лучшей в среднем.", notes[main])
        for horizon, note in notes.items():
            if horizon != main:
                with self.subTest(lost_window_only_at_the_main_horizon=horizon):
                    self.assertNotIn("эталон точнее", note)
        # Пояснение «модель не удалось обучить» — только если на этом горизонте у какой-то из
        # четырёх моделей MAE нет: полосы нет, и читатель должен знать почему.
        year = self.horizons_summary[self.horizons_summary["horizon"] == 12].set_index("model")["MAE"]
        shown = [m["id"] for m in horizons["models"]]
        has_gap = any(pd.isna(year.get(model_id, np.nan)) for model_id in shown)
        self.assertEqual("не удалось обучить" in notes[12], has_gap)
        for horizon, note in notes.items():
            if horizon != 12:
                self.assertNotIn("не удалось обучить", note)

    # -- fact и teaser: те же значения, что в файлах стенда --------------------------------------

    def test_fact_equals_demo_aggregate_json(self) -> None:
        self.assertEqual(self.landing["fact"], self.aggregate_json)

    def test_teaser_equals_demo_mo_files(self) -> None:
        mo_files: dict[str, dict] = {}
        for path in (self.data_dir / "mo").glob("*.json"):
            mo_files.update(json.loads(path.read_text(encoding="utf-8")))
        regions = {row[0]: row[1] for row in self.index_json["series"]}

        teaser = self.landing["teaser"]
        self.assertEqual(
            teaser["months"], self.index_json["panel_months"] + self.index_json["forecast_months"]
        )
        # Три примера (TEASER_MO): МО страницы прогноза по умолчанию, Казань и сельский район — все с регионом,
        # у каждого подпись, чем он отличается от соседей.
        self.assertEqual(
            [item["id"] for item in teaser["items"]],
            [self.index_json["default_mo"], "городской округ город Казань", "Тербунский муниципальный район"],
        )
        self.assertEqual([item["kind"] for item in teaser["items"]], ["областной центр", "миллионник", "сельский район"])
        reference = next(m for m in self.index_json["models"] if "reference" in m["role"].split("+"))
        self.assertEqual(teaser["reference"], {"id": reference["id"], "name": reference["name"], "note": reference["note"]})
        for item in teaser["items"]:
            with self.subTest(series=item["id"]):
                self.assertEqual(set(item), TEASER_ITEM_KEYS)
                entry = mo_files[item["id"]]
                for key in ("fact", "forecast", "breaks"):
                    self.assertEqual(item[key], entry[key])
                self.assertEqual(item["region"], regions[item["id"]])
                self.assertTrue(regions[item["id"]], "пример главной — ряд с регионом")
                self.assertTrue(item["short"])
                self.assertNotIn(None, item["fact"] + item["forecast"] + item["baseline"])
                self.assertEqual(len(item["baseline"]), len(item["forecast"]))

    def test_teaser_baseline_is_the_reference_forward_forecast_rounded(self) -> None:
        # Линия эталона — прогноз Prophet по умолчанию от конца панели на те же 12 месяцев
        # (results/forecast_2025_baseline.csv), округлённый до рубля, как остальные значения стенда.
        base_cfg = self.forward_cfg["baseline"]
        baseline = read_results(ROOT / "results" / "forecast_2025_baseline.csv", dtype={"error": object})
        self.assertEqual(base_cfg["model"], "prophet")
        for item in self.landing["teaser"]["items"]:
            with self.subTest(series=item["id"]):
                rows = baseline.loc[
                    (baseline["series_id"] == item["id"]) & (baseline["model"] == base_cfg["model"])
                    & (baseline["horizon"] == int(base_cfg["horizon"]))
                ].set_index("month")["forecast"]
                expected = [int(round(rows.loc[month])) for month in self.index_json["forecast_months"]]
                self.assertEqual(item["baseline"], expected)

    def test_teaser_summary_numbers_recomputed_from_demo_files(self) -> None:
        # Числа раздела — из тех же целых рублей, что на графике и в таблице ошибок стенда: средний месяц
        # прогноза, его рост к среднему месяцу последнего года панели, средний месяц эталона, лучшая по
        # средней ошибке на окнах проверки модель таблицы ошибок и та же ошибка эталона.
        mo_files: dict[str, dict] = {}
        for path in (self.data_dir / "mo").glob("*.json"):
            mo_files.update(json.loads(path.read_text(encoding="utf-8")))
        models = self.index_json["models"]
        reference = next(m for m in models if "reference" in m["role"].split("+"))
        for item in self.landing["teaser"]["items"]:
            with self.subTest(series=item["id"]):
                summary = item["summary"]
                self.assertEqual(set(summary), TEASER_SUMMARY_KEYS)
                entry = mo_files[item["id"]]
                forecast_mean = float(np.mean(item["forecast"]))
                fact_mean = float(np.mean(item["fact"][-12:]))
                self.assertEqual(summary["forecast_mean"], _fmt(forecast_mean, 0))
                self.assertEqual(summary["baseline_mean"], _fmt(float(np.mean(item["baseline"])), 0))
                self.assertEqual(summary["growth"], f"{_fmt_signed((forecast_mean / fact_mean - 1) * 100, 1)}%")
                means = {}
                for model in models:
                    finite = [v for v in entry["mae"][model["id"]] if v is not None]
                    means[model["id"]] = round(float(np.mean(finite))) if finite else None
                best = min((m for m in models if means[m["id"]] is not None), key=lambda m: means[m["id"]])
                self.assertEqual(summary["best"], {"id": best["id"], "name": best["name"], "note": best["note"],
                                                   "mae": _fmt(means[best["id"]], 0)})
                self.assertEqual(summary["reference_mae"], _fmt(means[reference["id"]], 0))

    def test_teaser_rejects_missing_series_gaps_and_a_baseline_without_the_series(self) -> None:
        models = self.index_json["models"]
        base_cfg = self.forward_cfg["baseline"]
        origin = pd.Period(self.forward_cfg["origin"], "M")
        months = [str(origin + step) for step in range(1, int(base_cfg["horizon"]) + 1)]
        ids = [sid for sid, _, _ in build_site.TEASER_MO]
        baseline = pd.DataFrame({
            "series_id": np.repeat(ids, len(months)), "model": base_cfg["model"], "horizon": int(base_cfg["horizon"]),
            "month": months * len(ids), "forecast": 1.0,
        })

        empty = build_site.DemoBuild(files={}, total_bytes=0, entries={}, regions={}, models=models)
        with self.assertRaises(ValueError):
            build_site.build_teaser(empty, self.wide, self.forward_cfg, baseline)

        def entries_of():
            return {sid: {"fact": [1] * 24, "forecast": [1] * 12, "breaks": [],
                          "mae": {m["id"]: [1, 1, 1] for m in models}} for sid in ids}

        gappy_entries = entries_of()
        gappy_entries[ids[0]]["forecast"] = [None] * 12
        gappy = build_site.DemoBuild(files={}, total_bytes=0, entries=gappy_entries,
                                     regions={sid: None for sid in ids}, models=models)
        with self.assertRaises(ValueError):
            build_site.build_teaser(gappy, self.wide, self.forward_cfg, baseline)

        whole = build_site.DemoBuild(files={}, total_bytes=0, entries=entries_of(),
                                     regions={sid: None for sid in ids}, models=models)
        with self.assertRaises(ValueError):
            build_site.build_teaser(whole, self.wide, self.forward_cfg, baseline[baseline["series_id"] != ids[0]])
        with self.assertRaises(ValueError):
            build_site.build_teaser(whole, self.wide, self.forward_cfg, baseline[baseline["month"] != months[-1]])
        teaser = build_site.build_teaser(whole, self.wide, self.forward_cfg, baseline)
        self.assertEqual([item["baseline"] for item in teaser["items"]], [[1] * 12] * 3)

    # -- шрифты, стили, скрипт: файлы на месте, внешнего нет -------------------------------------

    def test_fonts_are_self_hosted_with_licences(self) -> None:
        css = (ROOT / "site" / "site.css").read_text(encoding="utf-8")
        faces = re.findall(r"@font-face\s*\{(.*?)\}", css, re.S)
        self.assertEqual(len(faces), 6, "ожидаются кириллица и латиница трёх семейств")
        families = set()
        for face in faces:
            family = re.search(r'font-family:\s*"([^"]+)"', face).group(1)
            families.add(family)
            with self.subTest(family=family):
                self.assertIn("font-display: swap", face)
                self.assertIn("unicode-range:", face)
                src = re.search(r'url\("(fonts/[^"]+\.woff2)"\)', face)
                self.assertIsNotNone(src, "шрифт должен лежать в site/fonts/")
                font_file = ROOT / "site" / src.group(1)
                self.assertTrue(font_file.exists(), f"{font_file} нет")
                self.assertEqual(font_file.read_bytes()[:4], b"wOF2", f"{font_file.name}: не woff2")
        self.assertEqual(families, {"Unbounded", "Onest", "JetBrains Mono"})
        for licence in ("unbounded-OFL.txt", "onest-OFL.txt", "jetbrains-mono-OFL.txt"):
            with self.subTest(licence=licence):
                text = (ROOT / "site" / "fonts" / licence).read_text(encoding="utf-8")
                self.assertIn("SIL OPEN FONT LICENSE Version 1.1", text)

    @staticmethod
    def _js_without_comments(code: str) -> str:
        """Код скрипта без комментариев: блочные, затем «//» до конца строки (адреса «://» не режутся)."""
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
        return "\n".join(line.split("//")[0] if "://" not in line else line for line in code.splitlines())

    @staticmethod
    def _page_scripts() -> list[str]:
        """Все скрипты страниц: файлы `site/*.js` и `demo/*.js` (список не пишется руками — новый
        файл попадает под проверки сам)."""
        names = sorted(str(path.relative_to(ROOT)) for folder in ("site", "demo") for path in (ROOT / folder).glob("*.js"))
        assert "site/landing-lib.js" in names and "demo/linechart.js" in names, names
        return names

    def test_stylesheets_and_script_load_nothing_external(self) -> None:
        for name in ("site/site.css", "site/landing.css", "demo/demo.css"):
            css = (ROOT / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertNotRegex(css, r"url\(\s*['\"]?(?:https?:)?//", "внешний url() в стилях")
                self.assertNotIn("@import", css)
        for name in self._page_scripts():
            js = (ROOT / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                # Единственный «http» в скрипте — пространство имён SVG, это не ресурс.
                urls = [u for u in re.findall(r"https?://[^\s\"'`)]+", js) if u != "http://www.w3.org/2000/svg"]
                self.assertEqual(urls, [], f"{name} не должен ходить за пределы сайта")
                for forbidden in ("XMLHttpRequest", "WebSocket", "sendBeacon", "import("):
                    self.assertNotIn(forbidden, js)
        # Стенд читает только свои файлы данных, и только по относительным путям.
        demo_js = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        fetched = re.findall(r"fetchJson\(\s*[`\"']([^`\"']+)", demo_js)
        self.assertTrue(fetched, "проверка потеряла обращения стенда к данным")
        for path in fetched:
            with self.subTest(fetch=path):
                self.assertTrue(path.startswith("data/"), f"{path}: данные стенда лежат в demo/data/")

    def test_local_assets_referenced_by_stylesheets_exist(self) -> None:
        for name in ("site/site.css", "site/landing.css", "demo/demo.css"):
            css_path = ROOT / name
            for url in re.findall(r"url\(\s*['\"]?([^'\")\s]+)", css_path.read_text(encoding="utf-8")):
                with self.subTest(file=name, url=url):
                    self.assertTrue((css_path.parent / url).resolve().exists(), f"{url}: файла нет")

    def test_landing_scripts_have_no_hand_typed_data_numbers(self) -> None:
        for name in ("site/landing.js", "site/landing-lib.js"):
            with self.subTest(file=name):
                self._check_no_data_numbers_in((ROOT / name).read_text(encoding="utf-8"), name)

    def test_landing_script_binds_each_year_error_row_to_its_own_method(self) -> None:
        # Три строки «Средняя ошибка за год» — свои методы: перепутанные `mean-r1` и `mean-r2` проходили
        # все тесты, а читатель увидел бы чужое число у правила. Порядок строк в разметке — как здесь.
        js = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        ids = _IdsOf(self.html)
        rows = {"mean-own": "two_stage", "mean-r1": "naive", "mean-r2": "seasonal_naive"}
        for node_id, method in rows.items():
            with self.subTest(row=node_id):
                self.assertIn(f'byId("{node_id}").textContent = `${{fact.mape.{method}}}%`;', js)
                self.assertEqual(ids.attr(node_id, "class"), node_id)
        for node_id, method in {"mean-r1-name": "naive", "mean-r2-name": "seasonal_naive"}.items():
            with self.subTest(row=node_id):
                self.assertIn(f'byId("{node_id}").textContent = `«${{fact.rule_names.{method}}}»`;', js)
        # Само число и подпись правила в данных — одного метода: названия правил из того же файла проверки.
        names = self.aggregate_json["rule_names"]
        self.assertEqual(names["naive"], self.agg_check.loc[self.agg_check["method"] == "naive", "aggregate_model"].iloc[0])
        self.assertEqual(
            names["seasonal_naive"],
            self.agg_check.loc[self.agg_check["method"] == "seasonal_naive", "aggregate_model"].iloc[0],
        )

    def test_fact_check_line_slider_and_rule_toggles_are_tied_to_the_chart(self) -> None:
        # Факт проверки — та же линия, что и история (кружки без линии читались как отдельный ряд): нарисована
        # по `check.actual` целиком и обрезана по выбранному месяцу. Ползунок стоит под месяцами проверки —
        # бегунок на одной вертикали с кружком месяца, — подписан названием месяца, а переключатели правил
        # стоят в легенде, где правила подписаны.
        js = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        self.assertIn('byId("fact-act").setAttribute("d", tail(check.actual));', js)
        self.assertIn("reveal.style.width = `${view.xs[view.nHistory + i].toFixed(1)}px`;", js)
        self.assertIn("scrubTrack.style.left = `calc(${left(nHistory)} - var(--scrub-thumb) / 2)`;", js)
        self.assertIn("scrubValue.textContent = MONTH_NOM[Number(month.slice(5, 7)) - 1];", js)
        ids = _IdsOf(self.html)
        self.assertEqual(ids.attr("fact-act", "clip-path"), "url(#fact-act-clip)")
        self.assertEqual(ids.attr("fact-act-reveal", "tag"), "rect")
        self.assertEqual(ids.attr("fact-k-value", "for"), "fact-k")
        self.assertEqual(ids.attr("fact-k-label", "for"), "fact-k")
        legend = re.search(r'<ul class="legend legend-fact"[^>]*>(.*?)</ul>', self.html, re.S)
        self.assertIsNotNone(legend, "нет легенды раздела 03")
        for toggle in ("fact-r1", "fact-r2"):
            with self.subTest(toggle=toggle):
                self.assertIn(f'<input type="checkbox" id="{toggle}" checked>', legend.group(1))
        # Ползунок — один на странице, со своей дорожкой и делениями под ней.
        self.assertEqual(self.html.count('type="range"'), 1)
        self.assertLess(self.html.index('id="fact-k"'), self.html.index('id="fact-scrub-ticks"'))

    def _check_no_data_numbers_in(self, js: str, name: str) -> None:
        # Числа страницы: всё, что стоит в пунктах «пяти чисел», средние ошибки года и MAE
        # горизонтов. Нужны «данные-подобные» значения: с десятичной запятой, с неразрывным
        # пробелом разрядов или из четырёх и более цифр — короткие целые («24», «3») совпали бы
        # с константами кода, и проверка шумела бы.
        tokens: set[str] = set()
        number = r"\d+(?:\u00a0\d{3})*(?:,\d+)?"
        for stat in self.stats:
            tokens.update(re.findall(number, stat["number"] + " " + stat["caption"]))
        tokens.update(self.aggregate_json["mape"].values())
        for model in self.landing["horizons"]["models"]:
            tokens.update(str(v) for v in model["mae"] if v is not None)
        tokens.add(str(self.wide.shape[1]))

        def data_like(token: str) -> bool:
            return "," in token or "\u00a0" in token or len(re.sub(r"\D", "", token)) >= 4

        forbidden = sorted(t for t in tokens if data_like(t))
        self.assertGreater(len(forbidden), 10, "проверка потеряла числа страницы")
        for token in forbidden:
            variants = {token, token.replace("\u00a0", " "), token.replace("\u00a0", ""),
                        token.replace("\u00a0", "").replace(",", ".")}
            for variant in variants:
                with self.subTest(number=variant):
                    self.assertIsNone(
                        re.search(r"(?<![\w.,])" + re.escape(variant) + r"(?![\w.,])", js),
                        f"в {name} вписано число данных {variant!r}",
                    )

        # И сами ряды: массив из шести и более чисел подряд — вписанные данные.
        self.assertIsNone(
            re.search(r"\[\s*-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*){5,}\]", js),
            f"в {name} есть числовой массив — данные должны приходить из landing-data",
        )

    def test_template_has_no_hand_typed_numbers(self) -> None:
        # В шаблоне число — только подстановка `${…}`; вне подстановок цифр в видимом тексте
        # и в подписях нет. Исключения — номера разделов и шагов («01 · Главный вывод»,
        # «Шаг 02») и названия лицензий («CC BY-SA 4.0»): это метки, а не данные.
        template = (ROOT / "site" / "index.template.html").read_text(encoding="utf-8")
        collector = _TextCollector()
        collector.feed(template)
        text = " ".join(collector.chunks)
        text = re.sub(r"\$\{[a-z_0-9]+\}", " ", text)
        text = re.sub(r"\b0[1-7]\b", " ", text)
        text = text.replace("CC BY-SA 4.0", " ")
        leftovers = re.findall(r".{0,20}\d.{0,20}", text)
        self.assertEqual(leftovers, [], "в шаблоне вписаны числа; вынесите их в подстановки генератора")

    def test_template_has_no_hand_typed_number_words(self) -> None:
        # Правило «ни одного числа руками» — и словами тоже: «Пять горизонтов» над четырьмя кнопками
        # читалось как ошибка, «девять из десяти» повторяло константу генератора, «первый год панели»
        # — константу STORY_BASE_MONTHS. Исключения — обороты, где слово числом не является.
        for name, path in (("шаблон главной", ROOT / "site" / "index.template.html"),
                           ("стенд", ROOT / "demo" / "index.html")):
            collector = _TextCollector()
            collector.feed(path.read_text(encoding="utf-8"))
            text = re.sub(r"\$\{[a-z_0-9]+\}", " ", " ".join(collector.chunks))
            with self.subTest(page=name):
                self.assertEqual(_number_words_in(text), [], f"{name}: число вписано словами; вынесите его в подстановку")

    def test_number_word_check_sees_what_it_should_and_spares_what_it_should(self) -> None:
        # Сама проверка не должна ослепнуть: формы и падежи ловятся, исключения скрывают только свои обороты.
        caught = ["Пять горизонтов", "у девяти значений из десяти", "двух тысяч рядов", "за первый год панели",
                  "двенадцатый месяц", "три месяца", "шести моделей", "один фолд", "десять", "в три раза",
                  "ни одной из шести", "одной из шести", "сорок восемь гипотез", "четвёртое число", "ровно ноль"]
        for text in caught:
            with self.subTest(text=text):
                self.assertTrue(_number_words_in(text), f"проверка пропустила: {text}")
        spared = ["Одно число", "прогноз одного числа", "каждая линия — один муниципалитет", "первый этап модели",
                  "не двигается ни одной из моделей",
                  "прогноз первого этапа", "двухэтапная модель", "двухэтапной модели", "движение", "стоимость",
                  "стенд", "второстепенный", "расходы"]
        for text in spared:
            with self.subTest(text=text):
                self.assertEqual(_number_words_in(text), [])
        # Оборот-исключение не прячет число рядом с собой.
        self.assertTrue(_number_words_in("первый этап и первый год"))
        self.assertTrue(_number_words_in("двухэтапная модель на двух горизонтах"))

    def test_substitutions_outside_the_strip_sit_in_their_own_nodes(self) -> None:
        # Подстановки вне полосы чисел привязаны к узлам по id: подмена `n_series_rub` на другое число
        # в подзаголовке обложки, в шаге 01 или в блоке «Прогноз по муниципалитету», `forecast_year` на другое
        # число в заголовке раздела 03 или в подписи ползунка тест видит, а не «число нашлось где-то на странице».
        nodes = _IdsOf(self.html)
        n_series = _fmt(self.wide.shape[1], 0)
        origin = pd.Period(self.forward_cfg["origin"], "M")
        year = origin.year + 1
        origin_label = f"{_MONTH_OF[origin.month - 1]} {origin.year}"
        sample = self.landing["story"]["n_sample"]
        sample_label = f"{sample} " + _plural(sample, "случайный муниципалитет", "случайных муниципалитета",
                                              "случайных муниципалитетов")
        start = self.wide.index[0]
        base = f"{start.year} год" if start.month == 1 else None
        self.assertIsNotNone(base, "панель начинается не с января: проверке нужна своя подпись периода")

        self.assertEqual(nodes.text("hero-sub"), f"и его разнос по {n_series} муниципалитетам")
        # Шаг 01: выборка, число рядов и период — в одной строке шага; видимой подписи под графиком нет.
        self.assertIn(f"{sample_label} из {n_series}, расходы к среднему за {base}: растут и падают", nodes.text("step-1"))
        # Прогнозирует первый этап — по федеральному ряду; медиана рядов только показывает общее движение.
        self.assertIn("это одно число — федеральный ряд, который прогнозирует первый этап модели.", nodes.text("step-2"))
        # Шаг 04: число моделей разложения словом и границы разброса по всем сочетаниям модели и окна — из CSV.
        decomposition = read_results(ROOT / "results" / "error_decomposition.csv")
        spreads = decomposition["разброс"].astype(float)
        n_models = decomposition["модель"].nunique()
        words = {2: "двух", 3: "трёх", 4: "четырёх", 5: "пяти", 6: "шести", 7: "семи", 8: "восьми", 9: "девяти"}
        self.assertIn(
            f"Разброс между муниципалитетами у всех {words[n_models]} моделей почти равный, "
            f"{_fmt(spreads.min(), 0)}–{_fmt(spreads.max(), 0)}\u00a0₽,", nodes.text("step-4"))
        self.assertIn("поэтому прогноз одного числа и разнос долями.", nodes.text("step-4"))
        self.assertEqual(nodes.text("story-unit-lines"), f"Расходы к среднему за {base}, %")
        # Раздел «Один из … крупно» — заголовок с числом рядов и одна строка: три масштаба примеров и то, что
        # муниципальный прогноз фактом не проверен (именно здесь, рядом с линией прогноза, его проще всего принять
        # за проверенный).
        self.assertEqual(nodes.text("city-title"), f"Один из {n_series} крупно")
        self.assertEqual(
            nodes.text("stand-lead"),
            f"Примеры разного масштаба: областной центр, миллионник и сельский район. Прогноз на {year} год фактом "
            f"не проверен: муниципальный разрез {year} года ещё не опубликован.")
        self.assertEqual(nodes.text("fact-title"), f"Прогноз от {origin_label} против факта {year} года")
        self.assertEqual(
            nodes.text("fact-lead"),
            f"Федеральный ряд за весь {year} год опубликован, муниципальный разрез — нет: с фактом сверяется только "
            "первый этап модели.")
        self.assertEqual(nodes.text("fact-k-label"), f"Месяц {year}")
        self.assertEqual(nodes.text("material-forecast"), f"Прогноз на {year} год")
        meta = re.search(r'<meta name="description" content="([^"]*)"', self.html).group(1)
        self.assertIn(f"Прогноз потребительских расходов {n_series} муниципальных образований", meta)
        self.assertIn(f"прогноз на {year} год", meta)

    def test_cover_has_two_buttons_the_report_in_a_new_tab_and_the_forecast_page(self) -> None:
        # Вместо поля поиска на обложке — две кнопки: основная ведёт в отчёт (в новой вкладке, со знаком «↗» и
        # скрытой подписью, как все ссылки на документы), вторая — на страницу прогноза по муниципалитету.
        ids = _IdCollector()
        ids.feed(self.html)
        anchors = {a["href"]: a for a in _anchors(self.html)}
        nodes = _IdsOf(self.html)
        report, city = ids.ids["hero-report"], ids.ids["hero-city"]
        self.assertEqual((report["tag"], report["href"], report["target"], report["rel"]),
                         ("a", "report/report.html", "_blank", "noopener"))
        self.assertEqual(nodes.text("hero-report"), "Читать отчёт\u00a0↗ (откроется в новой вкладке)")
        self.assertIn("btn-primary", report["class"])
        self.assertEqual((city["tag"], city["href"], city.get("target")), ("a", "demo/", None))
        self.assertEqual(nodes.text("hero-city"), "Найти свой город →")
        self.assertIn("btn-secondary", city["class"])
        self.assertIn("demo/", anchors)
        # Кнопки — в одном блоке под подзаголовком: сначала основная, и обе до графика.
        cover = self._cover_html()
        cover = cover[:cover.index('class="stats')]
        self.assertTrue(cover.index('id="hero-sub"') < cover.index('id="hero-report"') < cover.index('id="hero-city"')
                        < cover.index('class="hero-plot"'))

    def test_cover_has_no_search_field_and_no_text_around_the_chart(self) -> None:
        # Поле «Покажите мой город», его подсказки и подпись удалены с обложки целиком — поиск живёт на странице
        # прогноза. Абзац-лид, строка с главным числом, подписи осей и легенда графика — тоже: он — иллюстрация.
        for gone in ("mo-q", "mo-list", "mo-go", "mo-status", "mo-note", 'role="search"', "search-label", "suggestions",
                     "hero-lead", "hero-proof", "hero-axis", "hero-caption", "hero-figure"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, self.html)
        self.assertNotIn("combobox", self.html)
        self.assertIn('<div class="hero-plot" aria-hidden="true">', self.html)
        cover = self._cover_html()
        self.assertNotIn("<figcaption", cover)
        collector = _TextCollector()
        collector.feed(cover)
        text = " ".join(collector.chunks)
        # Видимый текст обложки — метка, заголовок, подзаголовок, кнопки и полоса; цифра 23 (главное число) только
        # в полосе, один раз.
        self.assertEqual(text.count(self.stats[0]["number"]), 1)
        self.assertNotIn("медиана", text)

    def test_story_base_label_names_a_year_only_when_it_is_one(self) -> None:
        january = pd.DataFrame(index=pd.date_range("2023-01-01", periods=24, freq="MS"))
        self.assertEqual(build_site.story_base_label(january), "2023 год")
        # Панель с другого месяца: «год» был бы неправдой, и подпись называет месяцы.
        march = pd.DataFrame(index=pd.date_range("2023-03-01", periods=24, freq="MS"))
        self.assertEqual(build_site.story_base_label(march), "первые 12 месяцев панели")

    def test_horizon_folds_follow_the_config_and_reject_results_of_another_one(self) -> None:
        config = {int(item["horizon"]): int(item["n_folds"]) for item in self.horizons_cfg["horizons"]}
        self.assertEqual(build_site.horizon_folds(self.horizons_cfg, self.horizons_summary), config)
        other_config = self.horizons_summary.copy()
        other_config.loc[other_config["horizon"] == 6, "фолдов"] += 1
        with self.assertRaises(ValueError):
            build_site.horizon_folds(self.horizons_cfg, other_config)
        mixed = self.horizons_summary.copy()
        mixed.loc[(mixed["horizon"] == 3) & (mixed["model"] == "prophet"), "фолдов"] = 1
        with self.assertRaises(ValueError):
            build_site.horizon_folds(self.horizons_cfg, mixed)

    def test_section_one_is_four_steps_with_a_title_and_one_sentence_each_and_no_caption_under_the_chart(self) -> None:
        # Заголовок, интерактив и одна строка текста: у каждого из четырёх шагов — название и одно предложение;
        # подписи под графиком и строки-опоры со ссылкой на разложение ошибки нет (разложение — сам четвёртый шаг).
        # Подпись графика для программ чтения с экрана — скрытая живая область, которую скрипт заполняет названием
        # и строкой выбранного шага.
        for step in (1, 2, 3, 4):
            line = re.search(rf'id="step-{step}".*?<span class="step-text">(.*?)</span>', self.html, re.S).group(1)
            with self.subTest(step=step):
                self.assertEqual(len(re.findall(r"[.!?](?:\s|$)", line.strip())), 1, line)
                self.assertTrue(line.strip().endswith("."))
        for gone in ("story-proof", "plot-caption", "data-caption", "механизм-ошибка-состоит"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, self.html)
        ids = _IdCollector()
        ids.feed(self.html)
        self.assertEqual(ids.ids["story-caption"]["class"], "visually-hidden")
        self.assertEqual(ids.ids["story-caption"]["aria-live"], "polite")
        js = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        self.assertIn("caption.textContent = label;", js)
        self.assertIn('plot.setAttribute("aria-label", label);', js)

    # -- раздел «Изломы» ---------------------------------------------------------------------------------

    @staticmethod
    def _report_summary_items() -> list[str]:
        """Пункты списка «Остальные результаты» резюме отчёта — текстом, без разметки и ссылок на разделы."""
        report = (ROOT / "report" / "report.html").read_text(encoding="utf-8")
        start = report.index("<strong>Остальные результаты.</strong>")
        block = report[start:report.index("</ol>", start)]
        return [_collapse(unescape(re.sub(r"<[^>]+>", "", item))) for item in re.findall(r"<li>(.*?)</li>", block, re.S)]

    def test_breaks_claim_is_the_eighth_summary_item_of_the_report(self) -> None:
        # Фраза раздела «Изломы» — пункт 8 резюме отчёта слово в слово, без ссылки «(об обнаружении)»:
        # условия «решает преобразование», «скромный», «раннего предупреждения не показано» генератор
        # пересчитал на тех же файлах, и если прогон или отчёт изменились, расхождение видно здесь.
        items = self._report_summary_items()
        self.assertEqual(len(items), 8)
        item = items[7]
        self.assertTrue(item.startswith("Точки структурных изменений:"))
        self.assertTrue(item.endswith(" (об обнаружении)."))
        expected = item[: -len(" (об обнаружении).")] + "."
        self.assertEqual(_IdsOf(self.html).text("breaks-claim"), expected)
        # Жирным идёт начало фразы, как в резюме.
        self.assertIn("<strong>Точки структурных изменений: ", self.html)

    def test_breaks_data_are_the_offline_shares_at_the_offline_penalty_and_protocol(self) -> None:
        cp_cfg = yaml.safe_load((ROOT / "configs" / "changepoints.yaml").read_text(encoding="utf-8"))
        protocol = f"v{cp_cfg['protocol_version']}"
        nes_spec = importlib.util.spec_from_file_location("news_event_study", ROOT / "scripts" / "news_event_study.py")
        nes = importlib.util.module_from_spec(nes_spec)
        nes_spec.loader.exec_module(nes)
        penalty = float(nes.PENALTY)
        offline = read_results(ROOT / "results" / "cp_offline.csv")
        picture = offline[(offline["protocol"] == protocol) & np.isclose(offline["penalty"], penalty)].set_index("month")["share"]

        breaks = self.landing["breaks"]
        self.assertEqual(breaks["months"], [m.strftime("%Y-%m") for m in self.wide.index])
        self.assertEqual(breaks["share"], [round(float(picture[m]), 2) for m in breaks["months"]])
        # Штраф важен: при другом штрафе сетки доли другие, и страница не должна взять чужой.
        other = offline[(offline["protocol"] == protocol) & ~np.isclose(offline["penalty"], penalty)]
        for other_penalty, group in other.groupby("penalty"):
            with self.subTest(other_penalty=other_penalty):
                self.assertNotEqual(list(group.set_index("month")["share"].reindex(breaks["months"])), list(picture))
        # Месяцы массового согласия — столько наибольших долей, сколько событий в конфиге, по порядку месяцев.
        top = sorted(picture.nlargest(len(cp_cfg["realtime"]["events"])).index)
        self.assertEqual(breaks["top"], top)
        labels = []
        for month in top:
            text = _fmt(picture[month], 1)
            labels.append(f"{text[:-2] if text.endswith(',0') else text}%")
        self.assertEqual(breaks["top_labels"], labels)

    def test_breaks_section_is_the_claim_the_bars_and_one_note_that_says_what_the_bars_are_not(self) -> None:
        nodes = _IdsOf(self.html)
        # Изломы названы точками структурных изменений — термином конкурса. Абзаца о стенде с врезками и подписи
        # из двух предложений нет: в разделе фраза резюме, столбики и одна строка оговорок.
        # Неразрывный пробел перед тире: на узком экране заголовок не переносится так, чтобы строка начиналась с тире.
        self.assertEqual(nodes.text("breaks-title"), "Изломы\u00a0— точки структурных изменений")
        for gone in ("breaks-lead", "breaks-caption", "breaks-caveats", "breaks-more-text"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, self.html)
        section = self.html[self.html.index('<section class="section" id="breaks"'):self.html.index('id="city"')]
        self.assertEqual(re.findall(r'<p class="([^"]+)"', section),
                         ["eyebrow", "breaks-claim", "chart-unit", "breaks-note", "breaks-more"])
        # Строка под столбиками: найдено задним числом, не сигнал в реальном времени — и оговорки отчёта одной фразой.
        note = nodes.text("breaks-note")
        self.assertTrue(note.startswith("Найдено задним числом, не в реальном времени. "), note)
        self.assertEqual(note.count("; "), 1, "обе оговорки — одной фразой через точку с запятой")
        self.assertTrue(note.endswith("территорий."))
        # Порядок: заголовок, фраза резюме, график, строка под ним, ссылка на отчёт.
        marks = [section.index(mark) for mark in
                 ('id="breaks-title"', 'id="breaks-claim"', 'id="breaks-plot"', 'id="breaks-note"',
                  'href="report/report.html#sec-cp"')]
        self.assertEqual(marks, sorted(marks))
        # Раздел стоит после проверки фактом и перед блоком прогноза по муниципалитету; ссылка ведёт
        # в раздел отчёта об обнаружении (якорь проверяет тест ссылок), в новой вкладке.
        self.assertLess(self.html.index('id="fact"'), self.html.index('id="breaks"'))
        self.assertLess(self.html.index('id="breaks"'), self.html.index('id="city"'))
        link = re.search(r'<a href="(report/report\.html#sec-cp)" target="_blank" rel="noopener">', self.html)
        self.assertIsNotNone(link, "нет ссылки на раздел отчёта об обнаружении")
        report_ids = _IdCollector()
        report_ids.feed((ROOT / "report" / "report.html").read_text(encoding="utf-8"))
        self.assertIn("sec-cp", report_ids.ids)

    @staticmethod
    def _report_text() -> str:
        """Весь текст отчёта без разметки: пробелы разметки схлопнуты, неразрывные остаются."""
        report = (ROOT / "report" / "report.html").read_text(encoding="utf-8")
        return _collapse(unescape(re.sub(r"<[^>]+>", "", report)))

    def test_breaks_caveats_repeat_the_reports_words_next_to_the_numbers_recomputed_from_the_csv(self) -> None:
        # Столбики 50,5 / 63,7 / 85,8% — картина при штрафе, который стенд отвергает; отчёт нигде не ставит эти числа
        # без двух оговорок: пики — концы одного отрезка с декабрём внутри, а не отдельные шоки, и при выбранном штрафе
        # излом в любой месяц находится не больше чем у малой доли территорий. Числа пересчитаны здесь из CSV своими
        # выражениями, слова сверены с отрисованным отчётом.
        cp_cfg = yaml.safe_load((ROOT / "configs" / "changepoints.yaml").read_text(encoding="utf-8"))
        protocol = f"v{cp_cfg['protocol_version']}"
        nes_spec = importlib.util.spec_from_file_location("news_event_study", ROOT / "scripts" / "news_event_study.py")
        nes = importlib.util.module_from_spec(nes_spec)
        nes_spec.loader.exec_module(nes)
        offline_penalty = float(nes.PENALTY)
        events = sorted(str(e) for e in cp_cfg["realtime"]["events"])
        detector, mode = cp_cfg["realtime"]["detector"], cp_cfg["realtime"]["mode"]

        def table(name: str, **kwargs) -> pd.DataFrame:
            frame = read_results(ROOT / "results" / f"{name}.csv", **kwargs)
            return frame[frame["protocol"] == protocol]

        stand_events = table("cp_realtime_events", dtype={"crossed_month": object})
        selected = float(stand_events.loc[stand_events["selected"].astype(str).str.lower() == "true", "penalty"].iloc[0])
        self.assertNotEqual(selected, offline_penalty, "штрафы равны: оговорки о штрафе на странице быть не должно")
        offline = table("cp_offline")
        ceiling = float(offline.loc[np.isclose(offline["penalty"], selected), "share"].max())
        summary = table("cp_summary")
        false_alarm = float(summary.loc[
            (summary["detector"] == detector) & (summary["mode"] == mode) & np.isclose(summary["penalty"], offline_penalty),
            "ложных на чистых, %"].iloc[0])
        series = table("cp_offline_series")
        series = series[np.isclose(series["penalty"], offline_penalty)]
        first, second, last = events[0], events[1], events[-1]
        with_first = set(series.loc[series["month"] == first, "series_id"])
        pair = len(with_first & set(series.loc[series["month"] == second, "series_id"]))
        self.assertGreaterEqual(pair / len(with_first), 0.9)
        self.assertEqual(self.landing["breaks"]["top"], events, "пики — те же месяцы, что события конфига")

        gap = int((pd.Period(second, "M") - pd.Period(first, "M")).n)
        segment = [str(pd.Period(first, "M") + i) for i in range(gap)]
        tail = [str(m) for m in pd.period_range(last, self.wide.index[-1].strftime("%Y-%m"), freq="M")]
        self.assertTrue(any(m.endswith("-12") for m in segment) and any(m.endswith("-12") for m in tail))
        self.assertEqual(len(segment), len(tail))
        last_ids = set(series.loc[series["month"] == last, "series_id"])
        self.assertTrue(series[series["series_id"].isin(last_ids) & (series["month"] > last)].empty)

        def pct(value: float) -> str:
            text = _fmt(value, 1)
            return f"{text[:-2] if text.endswith(',0') else text}%"

        def pen(value: float) -> str:
            return _fmt(value, 0 if float(value).is_integer() else 1)

        def nom(month: str) -> str:
            year, number = month.split("-")
            return f"{_MONTH_NOM[int(number) - 1]} {year}"

        def prepositional(month: str) -> str:
            year, number = month.split("-")
            return f"{_MONTH_IN[int(number) - 1]} {year}"

        def span(months: list[str]) -> str:
            return f"{_MONTH_NOM[int(months[0][5:]) - 1]}–{nom(months[-1])}"

        nominative = {2: "два", 3: "три", 4: "четыре"}
        genitive = {2: "двух", 3: "трёх", 4: "четырёх"}
        ceiling_up = math.ceil(ceiling * 10 - 1e-9) / 10  # верхняя граница «не больше»: округление вверх
        shocks = f"{nominative[len(events)]} независимых {_plural(len(events), 'шок', 'шока', 'шоков')}"
        evidence = (
            f"У {_fmt(pair, 0)} из {_fmt(len(with_first), 0)} {_plural(len(with_first), 'ряда', 'рядов', 'рядов')} "
            f"с изломом в {prepositional(first)} есть излом и в {prepositional(second)}, ровно через {gap} "
            f"{_plural(gap, 'месяц', 'месяца', 'месяцев')}: это начало и конец отрезка {span(segment)}. "
            f"{nom(last).capitalize()} открывает такой же отрезок в конце панели — {span(tail)}.")
        # На странице обе оговорки — одной фразой под столбиками, без доказательства (числа рядов и отрезков —
        # в отчёте): первая часть про отрезки, вторая про штраф.
        segments_clause = (
            f"Подсвеченные столбики — не {shocks}, а концы {genitive[len(segment)]}месячных отрезков с декабрём "
            "внутри")
        penalty_clause = (
            f"штраф {pen(offline_penalty)} стенд отвергает: при нём потоковый {detector.upper()} тревожит на "
            f"{pct(false_alarm)} нетронутых рядов, а при выбранном по стенду штрафе {pen(selected)} и по полному ряду "
            f"излом в любой месяц находится не больше чем у {pct(ceiling_up)} территорий")

        note = _IdsOf(self.html).text("breaks-note")
        self.assertEqual(note, f"Найдено задним числом, не в реальном времени. {segments_clause}; {penalty_clause}.")
        self.assertNotIn("<p>", self.html[self.html.index('id="breaks-note"'):self.html.index('id="city"')])

        # Слова — отчёта: тот же тезис про отрезки, та же оговорка про штраф; числа доказательства («у N из M рядов
        # … ровно через три месяца»), пересчитанные здесь из CSV, — те же, что в отчёте: фраза на странице
        # опирается на них, хотя сами они остались в отчёте.
        report = self._report_text()
        for fragment in (
            f"не {shocks}, а концы {genitive[len(segment)]}месячных отрезков с декабрём внутри",
            evidence,
            f"при штрафе, который стенд отвергает: при нём потоковый {detector.upper()} тревожит на "
            f"{pct(false_alarm)} нетронутых рядов",
            f"при штрафе {pen(selected)} и по полному ряду излом в любой месяц находится не больше чем у "
            f"{pct(ceiling_up)} территорий",
        ):
            with self.subTest(fragment=fragment[:60]):
                self.assertIn(fragment, report)

    def test_breaks_note_sits_in_the_card_under_the_bars_before_the_link_to_the_report(self) -> None:
        # Строка с оговорками стоит в карточке с графиком, сразу под столбиками, а не отдельным разделом: столбики
        # и их оговорки читаются вместе. Ссылка на отчёт идёт после карточки.
        self.assertEqual(self.html.count('<div class="plot-card breaks-card">'), 1)
        start = self.html.index('<div class="plot-card breaks-card">')
        depth, end = 0, None
        for match in re.finditer(r"<div\b|</div>", self.html[start:]):
            depth += 1 if match.group(0) == "<div" else -1
            if depth == 0:
                end = start + match.end()
                break
        axis = self.html.index('id="breaks-axis"')
        note = self.html.index('id="breaks-note"')
        link = self.html.index('href="report/report.html#sec-cp"')
        self.assertTrue(start < axis < note < end < link, (start, axis, note, end, link))

    def test_sections_are_numbered_in_page_order(self) -> None:
        # Разделы нумеруются подряд, в порядке на странице: новый раздел «Изломы» сдвинул три следующих.
        labels = re.findall(r'<p class="eyebrow">(\d\d) · ([^<]+)</p>', self.html)
        self.assertEqual([number for number, _ in labels], [f"{n:02d}" for n in range(1, len(labels) + 1)])
        self.assertEqual(len(labels), 7)
        self.assertEqual(labels[3][1], "Изломы")

    # -- названия и ссылки ------------------------------------------------------------------------------

    GITHUB = "https://github.com/kr1zal/sberindex_konkurs"

    def test_documents_and_code_open_in_a_new_tab_and_the_site_pages_in_the_same(self) -> None:
        # Отчёт, слайды, их PDF, схема метода крупно и GitHub — в новой вкладке со знаком «↗» в подписи: у
        # отчёта и слайдов нет пути обратно на сайт, и читатель, открыв отчёт на много экранов, терял бы
        # сайт из виду. Переходы внутри сайта — в той же вкладке.
        for name, page in (("главная", self.html), ("страница прогноза", self.demo_html)):
            collector = _AnchorCollector()
            collector.feed(page)
            with self.subTest(page=name):
                self.assertGreater(len(collector.anchors), 8)
                documents = 0
                for anchor in collector.anchors:
                    href = anchor["href"]
                    if href.startswith(("report/", "../report/")) or href == self.GITHUB:
                        documents += 1
                        self.assertEqual(anchor["target"], "_blank", href)
                        self.assertEqual(anchor["rel"], "noopener", href)
                        # У каждой такой ссылки программа чтения с экрана слышит, что она откроется в новой вкладке;
                        # у ссылки-картинки схемы метода имя — длинный alt, и скрытая подпись стоит рядом с ним.
                        self.assertIn("(откроется в новой вкладке)", anchor["text"], href)
                        if anchor["text"].strip() != "(откроется в новой вкладке)":  # знак «↗» — у текстовых ссылок
                            self.assertIn("↗", anchor["text"], href)
                    elif not href.startswith(("http://", "https://")):
                        self.assertIsNone(anchor["target"], f"переход внутри сайта в новой вкладке: {href}")
                self.assertGreaterEqual(documents, 3)
        # В меню главной и страницы прогноза: отчёт, слайды и код — в новой вкладке; сами страницы сайта — нет.
        landing_nav = {a["href"]: a for a in _anchors(self.html)}
        self.assertIsNone(landing_nav["demo/"]["target"])
        self.assertEqual(landing_nav["report/slides.html"]["target"], "_blank")

    def test_the_method_diagram_link_says_it_opens_in_a_new_tab_to_screen_readers_too(self) -> None:
        # Картинка-ссылка на схему метода открывается в новой вкладке, как остальные ссылки на документы: у неё нет
        # видимого текста, а имя — длинный alt, поэтому скрытая подпись стоит прямо в ссылке, после картинки.
        link = re.search(r'<a href="report/method\.svg" target="_blank" rel="noopener">\s*<img .*?>\s*(.*?)\s*</a>',
                         self.html, re.S)
        self.assertIsNotNone(link, "не нашлась ссылка-картинка схемы метода")
        self.assertEqual(link.group(1), '<span class="visually-hidden"> (откроется в новой вкладке)</span>')
        # И подпись под картинкой — со знаком и скрытым текстом, как у остальных ссылок.
        caption = re.search(r"<figcaption>(.*?)</figcaption>", self.html, re.S).group(1)
        self.assertIn("(откроется в новой вкладке)", caption)

    def test_the_word_stand_names_only_the_protocol_with_injections(self) -> None:
        # Страница прогноза по муниципалитету на сайте нигде не «стенд»: слово зарезервировано за стендом с
        # врезками известной величины, которым отчёт и схема метода проверяют детекторы изломов.
        for name, path in (("шаблон главной", ROOT / "site" / "index.template.html"), ("страница", self.demo_html_path)):
            collector = _TextCollector()
            collector.feed(path.read_text(encoding="utf-8"))
            text = " ".join(collector.chunks)
            found = list(re.finditer(r"[Сс]тенд\w*", text))
            for match in found:
                with self.subTest(page=name, at=text[max(0, match.start() - 30): match.end() + 40]):
                    self.assertIn("врезк", text[match.start(): match.end() + 40])
            if name == "шаблон главной":  # подпись схемы метода говорит о стенде с врезками — проверка её видит
                self.assertGreaterEqual(len(found), 1)
        # И в строках скриптов — тексты, которые видит читатель: ошибки, подсказки, подписи.
        for name in ("site/landing.js", "demo/demo.js"):
            code = self._js_without_comments((ROOT / name).read_text(encoding="utf-8"))
            with self.subTest(file=name):
                self.assertEqual(re.findall(r".{0,40}[Сс]тенд.{0,30}", code), [])

    def test_demo_page_is_named_forecast_by_municipality_on_both_pages(self) -> None:
        # Одно имя: «Прогноз по муниципалитету» — в меню, метке раздела и карточке материалов, в метке и заголовке
        # страницы. Приглашение «Покажите мой город» — H1 страницы прогноза; на главной кнопка обложки зовёт
        # «Найти свой город», раздел «Один из … крупно» открывает выбранный пример и зовёт найти любой другой.
        self.assertIn('<a class="nav-link" href="demo/">Прогноз по муниципалитету</a>', self.html)
        self.assertIn('<p class="eyebrow">05 · Прогноз по муниципалитету</p>', self.html)
        self.assertIn('<a class="material-link" href="demo/">Прогноз по муниципалитету</a>', self.html)
        self.assertEqual(self.html.count(">Открыть этот муниципалитет →</a>"), 1, "кнопка раздела ведёт на страницу прогноза")
        self.assertEqual(self.html.count(f">найти любой из {build_site.rub(self.wide.shape[1])} →</a>"), 1)
        self.assertEqual(self.html.count(">Найти свой город →</a>"), 1, "кнопка обложки ведёт на страницу прогноза")
        self.assertNotIn("Покажите мой город", self.html)
        self.assertIn("<title>Прогноз по муниципалитету — покажите мой город</title>", self.demo_html)
        self.assertIn('<p class="eyebrow">Прогноз по муниципалитету</p>', self.demo_html)
        self.assertIn('<h1 class="stand-title" id="stand-title">Покажите мой город</h1>', self.demo_html)
        demo_js = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        self.assertIn("document.title = `${seriesId} — прогноз по муниципалитету`;", demo_js)

    def test_demo_page_names_the_landing_page_one_way_and_has_the_slides_link(self) -> None:
        # Главная названа в навигации «Обзор работы» — так же и в кнопке агрегата, а не «на главной»; в меню
        # страницы прогноза есть «Слайды», как в меню главной (сайт навигирует одинаково).
        self.assertIn("← Обзор работы", self.demo_html)
        self.assertIn("Интерактивная версия — в обзоре работы →", self.demo_html)
        self.assertNotIn("на главной", self.demo_html)
        links = _LinkCollector()
        links.feed(self.demo_html)
        self.assertIn("../report/slides.html", links.links)
        landing_links = _LinkCollector()
        landing_links.feed(self.html)
        for href in ("report/report.html", "report/slides.html", self.GITHUB):
            self.assertIn(href, landing_links.links)
        # Те же пункты меню, что на главной, в том же порядке: отчёт, слайды, код.
        demo_nav = [a["href"] for a in _anchors(self.demo_html) if a["href"].startswith(("../report/", self.GITHUB))]
        self.assertEqual(demo_nav, ["../report/report.html", "../report/slides.html", self.GITHUB])

    def test_materials_cards_show_the_sizes_of_the_files_they_give(self) -> None:
        # PDF (по два с лишним и по полмегабайта) и архив с прогнозом скачиваются целиком: размер — в подписи
        # карточки. Считается заново по самим файлам, мегабайтами по 1 048 576 байт.
        def size(path: str) -> str:
            return f"{_fmt((ROOT / path).stat().st_size / 1024 ** 2, 1)}\u00a0МБ"

        by_href = {a["href"]: a for a in _anchors(self.html)}
        for href in ("report/report.pdf", "report/slides.pdf"):
            with self.subTest(href=href):
                self.assertTrue(by_href[href]["text"].startswith(f"PDF · {size(href)}"), by_href[href]["text"])
        note = re.search(r'<p class="material-note">(csv\.gz[^<]*)</p>', self.html).group(1)
        self.assertEqual(note, f"csv.gz · {size('results/forecast_2025.csv.gz')}")

    def test_materials_cards_are_a_title_and_a_format_or_size_without_descriptions(self) -> None:
        # Карточка материала — название и формат или размер: описаний («полный текст», «коротко о главном»,
        # «воспроизводимые прогоны») нет, что внутри — понятно по названию.
        def size(path: str) -> str:
            return f"{_fmt((ROOT / path).stat().st_size / 1024 ** 2, 1)}\u00a0МБ"

        cards = re.findall(r'<li class="card card-link material">(.*?)</li>', self.html, re.S)
        self.assertEqual(len(cards), 5)
        notes = [_collapse(unescape(re.sub(r"<[^>]+>", "", re.search(r'<p class="material-note">(.*?)</p>', card, re.S).group(1))))
                 for card in cards]
        self.assertTrue(notes[0].startswith(f"HTML и PDF · {size('report/report.pdf')}"), notes[0])
        self.assertTrue(notes[1].startswith(f"HTML и PDF · {size('report/slides.pdf')}"), notes[1])
        self.assertEqual(notes[2:], ["страница", "GitHub", f"csv.gz · {size('results/forecast_2025.csv.gz')}"])
        for gone in ("полный текст", "коротко о главном", "воспроизводимые прогоны", "все ряды панели",
                     "расходы, прогноз и изломы"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, self.html)

    def test_file_size_label_uses_megabytes_and_falls_back_to_kilobytes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            def made(size_bytes: int) -> Path:
                path = Path(folder) / f"f{size_bytes}"
                path.write_bytes(b"x" * size_bytes)
                return path

            self.assertEqual(build_site.file_size_label(made(2 * 1024 ** 2 + 400_000)), "2,4\u00a0МБ")
            self.assertEqual(build_site.file_size_label(made(600_000)), "0,6\u00a0МБ")
            self.assertEqual(build_site.file_size_label(made(53_000)), "0,1\u00a0МБ")
            # Меньше 0,05 МБ при округлении до десятых стал бы «0,0 МБ»: показывается в килобайтах.
            self.assertEqual(build_site.file_size_label(made(8_416)), "8\u00a0КБ")

    # -- разметка главной ---------------------------------------------------------------------------

    def test_page_has_landmarks_sections_and_accessible_controls(self) -> None:
        ids = _IdCollector()
        ids.feed(self.html)
        for section in ("why", "horizons", "fact", "breaks", "city", "method", "report"):
            with self.subTest(section=section):
                self.assertEqual(ids.ids[section]["tag"], "section")
                self.assertEqual(ids.ids[section]["aria-labelledby"], f"{section}-title")
                self.assertIn(f"{section}-title", ids.ids)
        # Поля поиска на главной нет (оно — на странице прогноза); смена шага и горизонта объявляется
        # живыми областями.
        self.assertNotIn("mo-q", ids.ids)
        for live in ("story-caption", "h-note"):
            with self.subTest(live=live):
                self.assertEqual(ids.ids[live]["aria-live"], "polite")
        # Графики с данными — картинки с названием; декоративный график обложки скрыт.
        for chart in ("story-plot", "fact-plot", "breaks-plot", "t-plot"):
            with self.subTest(chart=chart):
                self.assertEqual(ids.ids[chart]["role"], "img")
                self.assertTrue(ids.ids[chart]["aria-label"])
        self.assertIn('class="hero-plot" aria-hidden="true"', self.html)
        # Шаги и переключатели — кнопки с состоянием.
        self.assertEqual(len(re.findall(r'<button class="step" id="step-\d" type="button" aria-pressed=', self.html)), 4)
        self.assertIn('lang="ru"', self.html)
        self.assertIn('<link rel="icon" href="favicon.ico" sizes="32x32">', self.html)
        self.assertIn('<link rel="icon" href="favicon.svg" type="image/svg+xml">', self.html)
        self.assertIn('name="viewport"', self.html)
        self.assertIn('name="description"', self.html)

    def test_page_loads_the_helpers_before_the_page_script(self) -> None:
        # landing.js берёт помощники из LandingLib при запуске: без файла перед ним страница не оживёт.
        template = (ROOT / "site" / "index.template.html").read_text(encoding="utf-8")
        for name, page in (("index.html", self.html), ("шаблон", template)):
            with self.subTest(page=name):
                lib = page.index('<script src="site/landing-lib.js" defer></script>')
                script = page.index('<script src="site/landing.js" defer></script>')
                self.assertLess(lib, script)

    def test_page_links_to_the_expected_materials(self) -> None:
        collector = _LinkCollector()
        collector.feed(self.html)
        for link in ("report/report.html", "report/slides.html", "report/report.pdf", "report/slides.pdf",
                     "demo/", "report/method.svg", "results/forecast_2025.csv.gz",
                     "https://github.com/kr1zal/sberindex_konkurs"):
            with self.subTest(link=link):
                self.assertIn(link, collector.links)


class WindowsWordingTest(unittest.TestCase):
    """Оговорки об окнах проверки (фолдах протокола) на числах, которых нет в текущем прогоне: ни одного окна,
    все окна, одно, несколько; числа словом в формах «одно окно проверки» и «по трём окнам проверки»."""

    def test_windows_held_names_how_many_of_the_windows_the_gain_holds_in(self) -> None:
        self.assertEqual(build_site.on_windows(2, 3), "в 2 окнах проверки из 3")
        self.assertEqual(build_site.on_windows(1, 3), "в 1 окне проверки из 3")
        self.assertEqual(build_site.on_windows(21, 25), "в 21 окне проверки из 25")
        self.assertEqual(build_site.on_windows(3, 3), "во всех окнах проверки")
        self.assertEqual(build_site.on_windows(0, 3), "ни в одном окне проверки")

    def test_lost_windows_are_named_by_their_order_and_the_shortest_history_only_when_alone(self) -> None:
        sentence = lambda lost, shortest: build_site.windows_lost_sentence(lost, shortest, "best_mean")  # noqa: E731
        self.assertEqual(sentence([0], 0), "В первом окне, с самой короткой историей, эталон точнее лучшей в среднем.")
        self.assertEqual(sentence([1], 0), "Во втором окне эталон точнее лучшей в среднем.")
        self.assertEqual(sentence([2], 0), "В третьем окне эталон точнее лучшей в среднем.")
        self.assertEqual(sentence([1, 2], 0), "Во втором и третьем окнах эталон точнее лучшей в среднем.")
        # Первое окно среди нескольких — без слов о короткой истории: они про одно окно.
        self.assertEqual(sentence([0, 2], 0), "В первом и третьем окнах эталон точнее лучшей в среднем.")
        self.assertEqual(sentence([3], 0), "В 4-м окне эталон точнее лучшей в среднем.")
        self.assertEqual(sentence([], 0), "")

    def test_the_compared_model_is_named_in_the_genitive_by_its_role(self) -> None:
        # Фраза называет, кого эталон обошёл: роль даёт родительный падеж названия, а не слово, вписанное в оборот.
        sentence = build_site.windows_lost_sentence
        self.assertEqual(sentence([0], 0, "naive"), "В первом окне, с самой короткой историей, эталон точнее наивной.")
        self.assertEqual(sentence([1, 2], 0, "two_stage"), "Во втором и третьем окнах эталон точнее двухэтапной.")
        with self.assertRaises(KeyError):
            sentence([0], 0, "champion")
        # У каждой роли с названием есть и родительный падеж: роль, добавленная в ROLE_NAMES, без него не пройдёт.
        self.assertEqual(set(build_site.ROLE_NAMES_GEN), set(build_site.ROLE_NAMES))
        # Родительный падеж — того же слова, что название роли (первые буквы первого слова совпадают): оборот не
        # назовёт другую модель. У лучшей в среднем «по панели» опущено: оборот стоит под полосами с полным названием.
        for role, name in build_site.ROLE_NAMES.items():
            with self.subTest(role=role):
                self.assertEqual(build_site.ROLE_NAMES_GEN[role].split()[0][:4], name.split()[0].lower()[:4])
        self.assertEqual(build_site.ROLE_NAMES["best_mean"], "Лучшая в среднем по панели")
        self.assertEqual(build_site.ROLE_NAMES_GEN["best_mean"], "лучшей в среднем")

    def test_year_ahead_note_is_a_caveat_for_one_or_two_windows_and_a_plain_count_for_more(self) -> None:
        # «Но это одно окно проверки» — оговорка резюме отчёта («но это один фолд»); при трёх окнах и больше «но это
        # девять окон» читалось бы как довод в пользу числа, поэтому там просто счёт окон, как в разделе 02.
        note = build_site.year_windows_note
        self.assertEqual(note(1), "но это одно окно проверки")
        self.assertEqual(note(2), "но это два окна проверки")
        self.assertEqual(note(3), "по трём окнам проверки")
        self.assertEqual(note(9), "по девяти окнам проверки")
        self.assertEqual(note(12), "по 12 окнам проверки")
        self.assertEqual(note(21), "по 21 окну проверки")
        with self.assertRaises(ValueError):
            note(0)

    def test_small_numbers_in_the_site_forms_are_words_and_larger_ones_digits(self) -> None:
        self.assertEqual([build_site.in_words(n, "им_с") for n in (1, 2, 3, 10, 11)], ["одно", "два", "три", "десять", "11"])
        self.assertEqual([build_site.in_words(n, "дат") for n in (1, 2, 3, 9, 12)], ["одному", "двум", "трём", "девяти", "12"])
        self.assertEqual([build_site.plural(n, "окну", "окнам", "окнам") for n in (1, 2, 5, 21)],
                         ["окну", "окнам", "окнам", "окну"])


class ChangepointClaimTest(unittest.TestCase):
    """Условия фразы об изломах на таблицах, которые подставляют ветки отчёта, не встречающиеся в текущем
    прогоне: тезис не держится, лучший J выше пятой части шкалы, пороги переходят не все события или только
    в декабре. Таблицы — синтетические, генератор не запускается."""

    DETECTORS = ["pelt", "binseg", "window", "bottomup", "kernel_rbf", "cusum"]
    CFG = {
        "realtime": {"detector": "pelt", "mode": "ratio", "events": ["2023-10", "2024-01", "2024-10"]},
        "bench": {"detectors": DETECTORS},
    }

    def frames(self, *, ratio_fa=0.0, raw_fa=60.0, ratio_j=12.0, raw_j=-50.0, crossed=None, delay=0.0,
               ratio_overrides=None):
        """cp_summary и cp_realtime_events: у методов со штрафом (все, кроме CUSUM) — выбранный штраф."""
        rows = []
        for detector in self.DETECTORS:
            own = detector == "cusum"
            for mode, fa, j in (("raw", raw_fa, raw_j), ("ratio", ratio_fa, ratio_j), ("deseason", 90.0, -80.0)):
                if mode == "ratio" and ratio_overrides and detector in ratio_overrides:
                    fa, j = ratio_overrides[detector]
                rows.append({"protocol": "v2", "detector": detector, "mode": mode,
                             "penalty": float("nan") if own else 5.0, "J Юдена": j, "ложных на чистых, %": fa})
        events = []
        crossed = crossed or {}
        for event in self.CFG["realtime"]["events"]:
            month = crossed.get(event)
            events.append({"protocol": "v2", "penalty": 5.0, "selected": True, "event": event,
                           "crossed_month": month if month else float("nan"),
                           "delay": delay if month else float("nan")})
        return pd.DataFrame(rows), pd.DataFrame(events)

    def claim(self, **kwargs) -> str:
        head, body = build_site.changepoint_claim(*self.frames(**kwargs), self.CFG)
        return f"{head} {body}"

    def test_thesis_holds_when_every_method_gains_and_the_spread_is_small(self) -> None:
        text = self.claim()
        self.assertEqual(text, (
            "Точки структурных изменений: решает преобразование ряда, а не алгоритм. На сырых значениях методы "
            "со штрафом тревожат на 60% нетронутых рядов, на темпах роста — ни разу. Уровень обнаружения "
            "скромный, и раннего предупреждения на двух годах данных не показано."))

    def test_false_alarms_on_ratio_are_named_when_nonzero(self) -> None:
        text = self.claim(ratio_fa=3.3)
        self.assertIn("На сырых значениях методы со штрафом тревожат на 60% нетронутых рядов, "
                      "на темпах роста — на 3,3%.", text)

    def test_thesis_names_only_the_unmet_condition(self) -> None:
        # У одного метода преобразование не снижает ложные тревоги — тезис не подтверждён, и фраза говорит именно это.
        text = self.claim(ratio_overrides={"window": (70.0, 12.0)})
        self.assertTrue(text.startswith("Точки структурных изменений: преобразование ряда решает не всё. "))
        self.assertIn("Тезис «решает преобразование, а не алгоритм» этим прогоном не подтверждается: "
                      "у Window на темпах роста ложных тревог не меньше или J не выше, чем на сырых значениях.", text)
        # Разброс J между методами со штрафом не меньше наименьшего выигрыша от смены преобразования — второе условие.
        spread = self.claim(ratio_overrides={"window": (0.0, 100.0)})
        self.assertTrue(spread.startswith("Точки структурных изменений: преобразование ряда решает не всё. "))
        self.assertIn("разброс J между методами со штрафом на темпах роста, 88,0 пункта, "
                      "не меньше наименьшего выигрыша от смены преобразования, 62,0", spread)

    def test_level_is_modest_until_the_best_j_reaches_a_fifth_of_the_scale(self) -> None:
        self.assertIn("Уровень обнаружения скромный", self.claim(ratio_j=19.9))
        self.assertIn("Уровень обнаружения заметный", self.claim(ratio_j=20.0, ratio_overrides={"pelt": (0.0, 20.0)}))

    def test_early_warning_depends_on_where_the_threshold_was_crossed(self) -> None:
        none = self.claim()
        self.assertIn("и раннего предупреждения на двух годах данных не показано", none)
        # Все переходы — в декабре позже месяца события: сам декабрьский скачок, раннего предупреждения нет.
        december = self.claim(crossed={"2023-10": "2023-12", "2024-01": "2024-12"}, delay=2.0)
        self.assertIn("и раннего предупреждения на двух годах данных не показано", december)
        # Переход в месяц события или не в декабре — предупреждение есть: видны все события или часть.
        every = self.claim(crossed={"2023-10": "2023-11", "2024-01": "2024-02", "2024-10": "2024-10"}, delay=1.0)
        self.assertIn("а в реальном времени видны все события", every)
        part = self.claim(crossed={"2023-10": "2023-11", "2024-01": "2024-02"}, delay=1.0)
        self.assertIn("а в реальном времени видно 2 из 3 событий", part)

    def test_claim_refuses_a_panel_of_another_length_instead_of_repeating_the_old_words(self) -> None:
        # «На двух годах данных» — слова отчёта, написанные руками: панель другой длины сделала бы их неверными.
        # Сборка обязана остановиться, а не напечатать прежнюю фразу.
        tables = {"cp_summary": self.frames()[0], "cp_realtime_events": self.frames()[1]}
        months = [f"2023-{m:02d}" for m in range(1, 13)]
        tables["cp_realtime"] = pd.DataFrame({"protocol": "v2", "penalty": 5.0, "month": months, "share": 0.0})
        tables["cp_offline"] = pd.DataFrame({"protocol": "v2", "penalty": 1.0, "month": months, "share": 0.0})
        with self.assertRaises(ValueError) as caught:
            build_site.build_breaks(tables, self.CFG, 1.0)
        self.assertIn("двух годах данных", str(caught.exception))


class ChangepointCaveatsTest(unittest.TestCase):
    """Оговорки к столбикам раздела «Изломы» на таблицах, которые подставляют ветки отчёта, не встречающиеся в
    текущем прогоне: хвост без декабря, пики не одного отрезка, равные штрафы, верхняя граница «не больше».
    Оговорка — часть предложения со строчной буквы и без точки; на странице обе стоят одной фразой.
    Таблицы синтетические, генератор не запускается."""

    MONTHS = [f"{2023 + i // 12}-{i % 12 + 1:02d}" for i in range(24)]
    EVENTS = ["2023-10", "2024-01", "2024-10"]
    CFG = {"realtime": {"detector": "pelt", "mode": "ratio", "events": EVENTS}, "bench": {"detectors": ["pelt"]}}

    SEGMENTS = "подсвеченные столбики — не три независимых шока, а концы трёхмесячных отрезков с декабрём внутри"
    ONE_SEGMENT = (
        "подсвеченные столбики — не три независимых шока: октябрь 2023 и январь 2024 — начало и конец одного "
        "отрезка, а не два независимых события")

    def tables(self, *, offline=1.0, selected=5.0, ceiling=0.14, together=10, alone=0, tail_breaks=(),
               false_alarm=100.0):
        """Пять таблиц стенда: картина при штрафе `offline`, выбранный стендом штраф `selected`; у `together` рядов
        излом и в октябре 2023, и в январе 2024, ещё у `alone` — только в октябре 2023; `tail_breaks` — месяцы,
        в которые у ряда с изломом в октябре 2024 есть ещё излом."""
        grid = sorted({offline, selected})
        shares, summary, realtime, stand_events, rows = [], [], [], [], []
        for penalty in grid:
            for month in self.MONTHS:
                if penalty == selected:
                    share = ceiling if month == "2024-10" else 0.0
                else:
                    share = {"2023-10": 50.5, "2024-01": 63.7, "2024-10": 85.8}.get(month, 1.0)
                shares.append({"protocol": "v2", "penalty": penalty, "month": month, "share": share})
                realtime.append({"protocol": "v2", "penalty": penalty, "month": month, "share": 0.0})
            summary.append({"protocol": "v2", "detector": "pelt", "mode": "ratio", "penalty": penalty,
                            "ложных на чистых, %": false_alarm if penalty == offline else 13.9})
            for event in self.EVENTS:
                stand_events.append({"protocol": "v2", "penalty": penalty, "selected": penalty == selected,
                                     "event": event, "crossed_month": float("nan"), "delay": float("nan")})
        summary.append({"protocol": "v2", "detector": "cusum", "mode": "ratio", "penalty": float("nan"),
                        "ложных на чистых, %": 0.0})

        def add(series_id: str, *months: str) -> None:
            rows.extend({"protocol": "v2", "penalty": offline, "series_id": series_id, "month": m} for m in months)

        for i in range(together):
            add(f"вместе {i}", "2023-10", "2024-01")
        for i in range(alone):
            add(f"один {i}", "2023-10")
        for i in range(5):
            add(f"хвост {i}", "2024-10", *(tail_breaks if i == 0 else ()))
        return {"cp_summary": pd.DataFrame(summary), "cp_realtime": pd.DataFrame(realtime),
                "cp_realtime_events": pd.DataFrame(stand_events), "cp_offline": pd.DataFrame(shares),
                "cp_offline_series": pd.DataFrame(rows)}

    def caveats(self, offline=1.0, cfg=None, **kwargs) -> list[str]:
        return build_site.changepoint_caveats(self.tables(offline=offline, **kwargs), cfg or self.CFG, offline)

    def test_both_caveats_when_the_peaks_are_ends_of_one_segment_and_the_stand_rejects_the_penalty(self) -> None:
        got = self.caveats()
        self.assertEqual(got, [
            self.SEGMENTS,
            "штраф 1 стенд отвергает: при нём потоковый PELT тревожит на 100% нетронутых рядов, а при выбранном по "
            "стенду штрафе 5 и по полному ряду излом в любой месяц находится не больше чем у 0,2% территорий"])

    def test_ceiling_is_rounded_up_so_that_not_more_than_stays_true(self) -> None:
        # Максимум 0,14%: обычное округление дало бы «0,1%» — меньше самого максимума.
        self.assertIn("не больше чем у 0,2% территорий", self.caveats(ceiling=0.14)[-1])
        self.assertIn("не больше чем у 0,1% территорий", self.caveats(ceiling=0.0986)[-1])
        self.assertIn("не больше чем у 0,1% территорий", self.caveats(ceiling=0.1)[-1])
        self.assertIn("не больше чем у 0% территорий", self.caveats(ceiling=0.0)[-1])

    def test_false_alarm_rate_comes_from_the_summary_at_the_offline_penalty(self) -> None:
        self.assertIn("тревожит на 53,3% нетронутых рядов", self.caveats(false_alarm=53.3)[-1])

    def test_short_wording_when_the_tail_has_no_december_of_its_own(self) -> None:
        # У ряда с изломом в октябре 2024 есть ещё излом в ноябре: хвост — не такой же отрезок, и фраза про
        # «концы трёхмесячных отрезков с декабрём» не подходит; остаётся про начало и конец одного отрезка.
        got = self.caveats(tail_breaks=("2024-11",))
        self.assertEqual(got[0], self.ONE_SEGMENT)
        self.assertEqual(len(got), 2)

    def test_no_segment_caveat_when_the_peaks_are_not_the_ends_of_one_segment(self) -> None:
        # Из рядов с изломом в октябре 2023 в январе 2024 он есть у половины, меньше девяти десятых: отрезка нет.
        got = self.caveats(together=5, alone=5)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].startswith("штраф 1 стенд отвергает"))
        # Девять десятых — граница включительно.
        self.assertEqual(self.caveats(together=9, alone=1)[0], self.SEGMENTS)
        self.assertEqual(len(self.caveats(together=9, alone=1)), 2)

    def test_no_penalty_caveat_when_the_picture_is_taken_at_the_penalty_the_stand_chose(self) -> None:
        got = self.caveats(offline=5.0, selected=5.0)
        self.assertEqual(got, [self.SEGMENTS])

    def test_nothing_to_say_when_both_conditions_fail(self) -> None:
        self.assertEqual(self.caveats(offline=5.0, selected=5.0, together=3, alone=7), [])

    def test_unknown_events_and_a_missing_summary_row_stop_the_build(self) -> None:
        other_events = {**self.CFG, "realtime": {**self.CFG["realtime"], "events": ["2022-01", "2024-01", "2024-10"]}}
        with self.assertRaises(ValueError) as caught:
            self.caveats(cfg=other_events)
        self.assertIn("2022-01", str(caught.exception))
        tables = self.tables()
        summary = tables["cp_summary"]
        tables["cp_summary"] = summary[~np.isclose(summary["penalty"], 1.0)]
        with self.assertRaises(ValueError):
            build_site.changepoint_caveats(tables, self.CFG, 1.0)

    def test_note_joins_the_caveats_into_one_phrase(self) -> None:
        # Одна фраза: части через «;», первая с заглавной, в конце точка; оговорок нет — пустая строка.
        note = build_site.changepoint_note(self.caveats())
        self.assertEqual(note, (
            "Подсвеченные столбики — не три независимых шока, а концы трёхмесячных отрезков с декабрём внутри; "
            "штраф 1 стенд отвергает: при нём потоковый PELT тревожит на 100% нетронутых рядов, а при выбранном по "
            "стенду штрафе 5 и по полному ряду излом в любой месяц находится не больше чем у 0,2% территорий."))
        self.assertEqual(note.count(". "), 0)
        self.assertEqual(build_site.changepoint_note([self.SEGMENTS]), self.SEGMENTS[:1].upper() + self.SEGMENTS[1:] + ".")
        self.assertTrue(build_site.changepoint_note(self.caveats(together=5, alone=5)).startswith("Штраф 1 стенд"))
        self.assertEqual(build_site.changepoint_note([]), "")

    def test_build_breaks_puts_the_phrase_into_the_page_block(self) -> None:
        tables = self.tables()
        tables["cp_summary"] = self._summary_for_the_claim(tables["cp_summary"])
        built = build_site.build_breaks(tables, {**self.CFG, "bench": {"detectors": ["pelt", "cusum"]}}, 1.0)
        self.assertNotIn("<p>", built.caveats_html)
        self.assertTrue(built.caveats_html.startswith("Подсвеченные столбики"))
        self.assertTrue(built.caveats_html.endswith("территорий."))

    @staticmethod
    def _summary_for_the_claim(summary: pd.DataFrame) -> pd.DataFrame:
        """К строкам оговорок — строки, которые читает фраза резюме: raw и ratio у PELT и CUSUM, J Юдена."""
        rows = []
        for _, row in summary.iterrows():
            for mode, fa, j in (("raw", 60.0, -50.0), ("ratio", 0.0, 12.0)):
                rows.append({**row.to_dict(), "mode": mode, "ложных на чистых, %":
                             fa if mode == "raw" else row["ложных на чистых, %"], "J Юдена": j})
        return pd.DataFrame(rows)


def _functions_without_caller(tree: ast.Module) -> list[str]:
    """Функции верхнего уровня модуля, на которые не ссылается ни одно имя и ни один атрибут в самом модуле."""
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return sorted(defined - used)


def _unread_record_fields(tree: ast.Module) -> list[str]:
    """Поля записей `NamedTuple` модуля, которые в самом модуле никто не читает (`запись.поле`)."""
    read = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    unread = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(getattr(base, "id", None) == "NamedTuple" for base in node.bases):
            unread += [f"{node.name}.{item.target.id}" for item in node.body
                       if isinstance(item, ast.AnnAssign) and item.target.id not in read]
    return unread


class GeneratorHygieneTest(unittest.TestCase):
    """Мёртвый код генератора: помощник, чью единственную вызывающую фразу удалили, и поле записи, которое никто не
    читает, остаются в модуле незамеченными, пока не мешают. Проверка смотрит на сам модуль, а не на страницу."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse((ROOT / "scripts" / "build_site.py").read_text(encoding="utf-8"))

    def test_every_function_of_the_generator_has_a_caller_in_the_module(self) -> None:
        self.assertEqual(_functions_without_caller(self.tree), [])

    def test_every_field_of_the_generator_records_is_read_somewhere_in_the_module(self) -> None:
        self.assertEqual(_unread_record_fields(self.tree), [])

    def test_the_checks_see_what_they_should(self) -> None:
        # Проверки не слепы: на модуле с помощником без вызова и с полем, которого никто не читает, они их называют.
        module = ast.parse(
            "from typing import NamedTuple\n\n\nclass Pair(NamedTuple):\n    used: int\n    spare: int\n\n\n"
            "def kept(pair):\n    return pair.used\n\n\ndef _orphan():\n    return 2\n\n\nprint(kept(Pair(1, 2)))\n")
        self.assertEqual(_functions_without_caller(module), ["_orphan"])
        self.assertEqual(_unread_record_fields(module), ["Pair.spare"])


if __name__ == "__main__":
    unittest.main()
