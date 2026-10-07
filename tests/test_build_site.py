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
    "default_mo", "quick", "forecast_rule", "known_model", "breaks", "folds", "models",
    "panel_mae", "series",
}
AGGREGATE_JSON_KEYS = {
    "unit", "origin", "horizon", "model", "model_label",
    "history", "check", "rule_names", "mape",
}
LANDING_JSON_KEYS = {"story", "horizons", "fact", "teaser", "breaks"}
BREAKS_KEYS = {"months", "share", "top", "top_labels"}
STORY_KEYS = {"months", "base_months", "ids", "series", "median", "n_total", "n_sample"}
HORIZONS_KEYS = {"list", "main", "labels", "unit", "models", "notes"}
TEASER_ITEM_KEYS = {"id", "short", "region", "fact", "forecast", "known", "breaks"}

_MONTH_OF = ["января", "февраля", "марта", "апреля", "мая", "июня",
             "июля", "августа", "сентября", "октября", "ноября", "декабря"]
_MONTH_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь",
              "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
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
    """Пять пунктов `<li class="stat">` главной, по порядку — текст `.stat-number` и
    `.stat-caption` каждого, не текст «где-то в HTML»: связь «число ↔ его подпись»
    проверяется, только если число и текст сверяются в границах одного и того же пункта.
    `short` — сокращённая подпись для телефона (`span.stat-note-short`): она входит и в
    `caption` (подпись целиком — всё, что внутри `p.stat-caption`), и отдельно."""

    def __init__(self) -> None:
        super().__init__()
        self.stats: list[dict[str, str]] = []
        self._depth_in_li = 0
        self._current: dict[str, str] | None = None
        self._capture: str | None = None  # "number" | "caption" | None
        self._in_short = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "li" and "stat" in classes:
            self._current = {"number": "", "caption": "", "short": ""}
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
        elif tag == "span" and "stat-note-short" in classes:
            self._in_short = True

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag == "span":
            self._in_short = False
        if tag == "p" and self._capture:
            self._capture = None
        if tag == "li":
            self._depth_in_li -= 1
            if self._depth_in_li <= 0:
                self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture:
            self._current[self._capture] += data
            if self._in_short:
                self._current["short"] += data


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

    TEXT_ATTRS = ("alt", "aria-label", "title", "data-caption")

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
    """Видимый текст узлов по их `id` — вместе с вложенными тегами. Подстановки вне пяти карточек
    (подзаголовок обложки, шаг 01, блок «Стенд», заголовок раздела 03, ползунок) сверяются в
    границах своего узла: «где-то на странице» этим значениям не подходит — подмена одной
    подстановки другой (`n_series_rub` на `n_months`) оставалась бы незамеченной."""

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
# линию графика, название этапа модели, прилагательное «двухэтапная» и отрицание «ни одной из
# моделей». Исключения — целыми оборотами, а не основами: «первый год» или «двух месяцев» они
# не скрывают.
_NOT_NUMBERS = (
    "Одно число",
    "одного числа",
    "один муниципалитет",
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

        # Пять пунктов <li class="stat"> по порядку — число и подпись каждого отдельно,
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
            {"number": s["number"].strip(), "caption": collapse(s["caption"]), "short": collapse(s["short"])}
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
        # пятый элемент записи — средние расходы за последний год панели, тыс. ₽ на человека в месяц,
        # готовой строкой формата отчёта (пересчитано здесь по матрице панели); у рядов с регионом
        # записи остаются из четырёх элементов.
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
        # Общие с блоком «Стенд» главной ряды подписаны там так же.
        for series_id, short in build_site.TEASER_MO:
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
        # Старых названий на главной нет, подпись горизонта называет лучшую так же, как полоса.
        self.assertNotIn("Лучшая панельная", self.html)
        notes = dict(zip(self.landing["horizons"]["list"], self.landing["horizons"]["notes"]))
        self.assertIn("Лучшая в среднем по панели точнее эталона на", notes[1])

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

    def test_recommended_model_phrase_matches_the_forecast_rule_of_the_config(self) -> None:
        # Прогноз на год вперёд строит рекомендуемая модель, а не «лучшая в среднем»: фраза на главной называет,
        # какая именно и на какие месяцы, и повторяет формулировку отчёта. Месяцы и модели пересчитаны здесь по
        # configs/forecast_forward.yaml: шаг s берёт наименьший горизонт, который его покрывает.
        recommended = self.forward_cfg["recommended"]
        horizons = sorted(recommended)
        origin = pd.Period(self.forward_cfg["origin"], "M")
        by_month = [(origin + step, recommended[min(h for h in horizons if h >= step)])
                    for step in range(1, horizons[-1] + 1)]
        groups: list[list] = []
        for month, model in by_month:
            if groups and groups[-1][0] == model:
                groups[-1][2] = month
            else:
                groups.append([model, month, month])
        parts = [f"с {_MONTH_OF[a.month - 1]} по {_MONTH_NOM[b.month - 1]} — «{build_site.MODEL_LABELS[m]}»"
                 for m, a, b in groups]
        text = _IdsOf(self.html).text("stand-recommended")
        self.assertEqual(text, (
            f"Прогноз на {origin.year + 1} год строит рекомендуемая модель: {', '.join(parts)}. "
            "Рекомендуемая модель выбирается по длине истории, а не по верхней строке таблицы."))
        self.assertGreaterEqual(len(groups), 2)

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

    # -- число 1: горизонт основного протокола ------------------------------

    def test_headline_stat_matches_summary_csv(self) -> None:
        # stat-number сверяется целиком, а MAE/R² — в связке с горизонтом внутри подписи
        # ИМЕННО первого пункта (self.stats[0]), а не где угодно в HTML: раньше
        # assertIn(str(horizon), html) проходил на любой «3» где угодно на странице,
        # а число и подпись разных пунктов не были связаны.
        top = self.summary["MAE"].idxmin()
        prophet_mae = self.summary.loc["prophet", "MAE"]
        best_mae = self.summary.loc[top, "MAE"]
        # Выигрыш без знака: на странице число стоит перед словом «точнее».
        best_gain = self.summary.loc[top, "к Prophet, %"]
        r2_prophet = self.summary.loc["prophet", "R² пул"]
        r2_best = self.summary.loc[top, "R² пул"]
        horizon_main = int(self.full_cfg["split"]["horizon"])

        stat = self.stats[0]
        self.assertEqual(stat["number"], f"{_fmt(best_gain, 1)}%")
        # Горизонт — в названии числа, MAE — в первой строке пояснения: они идут подряд,
        # и «3 мес.» с MAE другого пункта связать нельзя.
        self.assertIn(
            f"на горизонте {horizon_main} мес. MAE {_fmt(prophet_mae, 0)} ₽ → {_fmt(best_mae, 0)} ₽",
            stat["caption"],
        )
        self.assertIn(f"R² по пулу {_fmt(r2_prophet, 3)} → {_fmt(r2_best, 3)}", stat["caption"])

    def test_folds_caveat_matches_per_series_csv(self) -> None:
        top = self.summary["MAE"].idxmin()
        fold_mae = (self.ok[self.ok["model"].isin([top, "prophet"])]
                    .groupby(["fold", "model"])["mae"].mean().unstack())
        folds = list(fold_mae.index)
        won = [f for f in folds if fold_mae.loc[f, top] < fold_mae.loc[f, "prophet"]]
        lost = [f for f in folds if fold_mae.loc[f, top] > fold_mae.loc[f, "prophet"]]

        # Оговорка о фолдах — часть подписи ИМЕННО первого пункта (folds_caveat в её
        # тексте, см. site/index.template.html): сверяем в границах self.stats[0],
        # а не где угодно на странице.
        caption = self.stats[0]["caption"]
        won_phrase = f"{len(won)} {'фолде' if len(won) == 1 else 'фолдах'} из {len(folds)}"
        self.assertIn(won_phrase, caption)
        for fold in lost:
            with self.subTest(fold=fold):
                self.assertIn(_ORDINAL.get(fold, str(fold)), caption)
        # На телефоне подпись сокращена до одного оборота — это то же число фолдов.
        self.assertEqual(self.stats[0]["short"], f"на {won_phrase}")

    def test_cover_proof_line_repeats_the_headline_number_with_its_caveat(self) -> None:
        # На обложке — главное число из тех же подстановок, что первая карточка, и та же оговорка о фолдах;
        # «проверено на факте» к нему не приписано: фактом проверен только прогноз федерального ряда.
        top = self.summary["MAE"].idxmin()
        best_gain = self.summary.loc[top, "к Prophet, %"]
        fold_mae = (self.ok[self.ok["model"].isin([top, "prophet"])]
                    .groupby(["fold", "model"])["mae"].mean().unstack())
        folds = list(fold_mae.index)
        won = [f for f in folds if fold_mae.loc[f, top] < fold_mae.loc[f, "prophet"]]
        won_phrase = f"{len(won)} {'фолде' if len(won) == 1 else 'фолдах'} из {len(folds)}"
        horizon_main = int(self.full_cfg["split"]["horizon"])
        proof = _IdsOf(self.html).text("hero-proof")
        self.assertEqual(
            proof, f"{_fmt(best_gain, 1)}% точнее эталона конкурса на горизонте {horizon_main} мес. — на {won_phrase}")
        self.assertNotIn("факт", proof)
        # То же число, что в первой карточке, — не второе, посчитанное отдельно.
        self.assertTrue(proof.startswith(self.stats[0]["number"]))
        self.assertIn(self.stats[0]["short"], proof)
        # Строка стоит в тексте обложки, до поля поиска и графика: на первом экране ноутбука она видна.
        self.assertLess(self.html.index('id="hero-lead"'), self.html.index('id="hero-proof"'))
        self.assertLess(self.html.index('id="hero-proof"'), self.html.index('id="mo-q"'))

    def test_glossary_under_the_cards_explains_fold_nowcast_and_origin(self) -> None:
        # «Фолд» стоит в главной оговорке («на 2 фолдах из 3»), и без словаря под карточками его нигде не
        # объясняют: длина окна — из index.json::folds, а не из головы.
        fold = self.index_json["folds"][0]
        first, last = pd.Period(fold["test_from"], "M"), pd.Period(fold["test_to"], "M")
        window = int((last - first).n) + 1
        glossary = _IdsOf(self.html).text("glossary")
        self.assertIn(f"фолд — проверочное окно в {window} мес.: модель учится на месяцах до него и прогнозирует его", glossary)
        self.assertIn("наукаст — прогноз текущего месяца до выхода его данных", glossary)
        self.assertIn("origin — месяц, от которого строится прогноз", glossary)
        # Словарь стоит сразу под строкой панели, до первого раздела.
        self.assertLess(self.html.index('class="stat stat-panel"'), self.html.index('id="glossary"'))
        self.assertLess(self.html.index('id="glossary"'), self.html.index('id="why"'))

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
        self.assertIn(f"MAE {_fmt(h1_best, 0)} против {_fmt(h1_prophet, 0)} ₽", stat["caption"])
        self.assertEqual(stat["short"], f"{_fmt(h1_best, 0)} против {_fmt(h1_prophet, 0)} ₽")

    # -- число 3: год вперёд (горизонт 12, без оракула) ---------------------

    def test_year_ahead_stat_matches_horizons_summary_and_folds_csv(self) -> None:
        year = self.horizons_summary[
            (self.horizons_summary["horizon"] == 12) & self.horizons_summary["MAE"].notna()
            & (self.horizons_summary["model"] != "two_stage_known")
        ].set_index("model")
        year_top = year["MAE"].idxmin()
        gain_naive = year.loc[year_top, "к наивной, %"]
        gain_prophet = year.loc[year_top, "к Prophet, %"]
        n_folds_top = int(self.horizons_folds.loc[
            (self.horizons_folds["horizon"] == 12) & (self.horizons_folds["model"] == year_top), "MAE"
        ].notna().sum())

        stat = self.stats[2]
        self.assertEqual(stat["number"], f"{_fmt(gain_prophet, 0)}%")
        self.assertIn(f"наивной — на {_fmt(gain_naive, 0)}%", stat["caption"])
        # «лучшая — …» — либо «двухэтапная» без кавычек, либо название из MODEL_LABELS
        # в кавычках (build_site.compute_placeholders::h12_model) — какая именно из двух
        # форм, зависит от того, какая модель победила, но подпись обязана содержать
        # одну из них, а не молчать, если это не двухэтапная (было именно так раньше).
        if year_top == "two_stage":
            self.assertIn("лучшая — двухэтапная", stat["caption"])
        else:
            label = build_site.MODEL_LABELS.get(year_top, year_top)
            self.assertIn(f"лучшая — «{label}»", stat["caption"])
        if 0 < n_folds_top < 3:
            # Оговорка целиком, числом словом: «один фолд» / «два фолда».
            words = {1: "один", 2: "два"}[n_folds_top]
            fold = _plural(n_folds_top, "фолд", "фолда", "фолдов")
            self.assertIn(f"но это {words} {fold}", stat["caption"])
            self.assertIn(f"но это {words} {fold}", stat["short"])

    # -- число 4: проверка агрегата по факту 2025 года -----------------------

    def test_aggregate_stat_matches_forecast_2025_aggregate_check_csv(self) -> None:
        abs_error = self.agg_check.assign(e=self.agg_check["error_pct"].abs())
        by_method_horizon = abs_error.groupby(["method", "horizon"])["e"].mean()
        own, rules = by_method_horizon.loc["two_stage"], by_method_horizon.drop(index="two_stage")

        def _range(series: pd.Series) -> str:
            lo, hi = series.min(), series.max()
            return _fmt(lo, 1) if _fmt(lo, 1) == _fmt(hi, 1) else f"{_fmt(lo, 1)}–{_fmt(hi, 1)}"

        horizons = sorted(int(h) for h in self.agg_check["horizon"].unique())
        origin = pd.Period(self.forward_cfg["origin"], "M")
        # «Год факта» страницы — тот же origin.year + 1, что и в отдельном годе прогноза
        # вперёд (число 3): здесь достаточно локального значения, привязанного к этому
        # же пункту подписи.
        forecast_year = origin.year + 1

        stat = self.stats[3]
        self.assertEqual(stat["number"], f"{_range(own)}%")
        self.assertIn(f"— {_range(rules)}%.", stat["caption"])
        self.assertEqual(stat["short"], f"у правил {_range(rules)}%")
        self.assertIn(f"на факте {forecast_year} года", stat["caption"])
        self.assertIn(f"горизонты {horizons[0]}–{horizons[-1]} мес.", stat["caption"])
        self.assertIn(f"{_MONTH_OF[origin.month - 1]} {origin.year}", stat["caption"])
        # Названия простых правил — из колонки aggregate_model того же файла.
        for name in self.agg_check.loc[self.agg_check["method"] != "two_stage", "aggregate_model"].unique():
            self.assertIn(f"«{name}»", stat["caption"])

    # -- число 5: форма панели ------------------------------------------------

    def test_panel_stat_matches_build_matrix(self) -> None:
        n_series, n_months = self.wide.shape[1], self.wide.shape[0]
        expected_shape = (
            f"{_fmt(n_series, 0)} {_plural(n_series, 'ряд', 'ряда', 'рядов')} × "
            f"{n_months} {_plural(n_months, 'месяц', 'месяца', 'месяцев')}"
        )
        start, end = self.wide.index[0], self.wide.index[-1]
        # panel_span целиком — с месяцами, а не только годы: «январь 2023 — декабрь
        # 2024», подменённое на «март 2023 — октябрь 2024», раньше проходило зелёным —
        # оба года встречались на странице и так.
        expected_span = f"{_MONTH_NOM[start.month - 1]} {start.year} — {_MONTH_NOM[end.month - 1]} {end.year}"

        stat = self.stats[4]
        self.assertEqual(stat["number"], expected_shape)
        self.assertIn(expected_span, stat["caption"])

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
        # Цвета рядов графика у стенда и главной одни и те же в обеих темах: стенд не держит
        # своих копий --chart-*, а каждый токен, на который он ссылается, объявлен в site.css
        # трижды — для светлой темы, для тёмной и для ночных блоков (.night).
        demo_css = (ROOT / "demo" / "demo.css").read_text(encoding="utf-8")
        self.assertNotRegex(demo_css, r"--chart-[a-z-]+\s*:", "demo.css объявляет свой токен графика")
        site_css = (ROOT / "site" / "site.css").read_text(encoding="utf-8")
        sources = demo_css + (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        used = set(re.findall(r"var\((--chart-[a-z-]+)\)", sources))
        self.assertTrue({"--chart-fact", "--chart-forecast", "--chart-known", "--chart-grid"} <= used)
        for token in sorted(used):
            with self.subTest(token=token):
                self.assertEqual(len(re.findall(re.escape(token) + r"\s*:", site_css)), 3,
                                 f"{token} должен быть объявлен для светлой, тёмной темы и .night")

    # -- данные главной: <script id="landing-data"> --------------------------------------

    def test_landing_data_is_one_parsable_json_script(self) -> None:
        self.assertIsNotNone(self.landing_raw, "в index.html нет ровно одного <script id=landing-data>")
        # «<» экранирован: иначе «</script» внутри строки данных закрыл бы тег.
        self.assertNotIn("<", self.landing_raw)
        self.assertEqual(set(self.landing), LANDING_JSON_KEYS)
        self.assertEqual(set(self.landing["story"]), STORY_KEYS)
        self.assertEqual(set(self.landing["horizons"]), HORIZONS_KEYS)
        self.assertEqual(set(self.landing["teaser"]), {"months", "items"})
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
        again = build_site.build_story(self.wide)
        self.assertEqual(again["ids"], self.landing["story"]["ids"])
        other = np.random.default_rng(build_site.LANDING_SAMPLE_SEED + 1).choice(
            self.wide.shape[1], size=build_site.LANDING_SAMPLE_SIZE, replace=False
        )
        self.assertNotEqual([str(self.wide.columns[i]) for i in other], self.landing["story"]["ids"])

    def test_story_spread_in_template_is_recomputed_from_all_series(self) -> None:
        # Третий шаг: «не менее чем у N% значений … не больше ±M%» — по всей панели. M — процентиль
        # отклонений, округлённый вверх: при 7,03% полоса ±7% вмещала бы меньше заявленной доли.
        norm = self._story_norm()
        deviation = norm.div(norm.median(axis=1), axis=0).sub(1).abs().to_numpy()
        share = build_site.STORY_SPREAD_SHARE
        self.assertTrue(50 < share < 100)
        band = math.ceil(round(float(np.percentile(deviation, share)) * 100, 6))
        caption = _IdsOf(self.html).attr("step-3", "data-caption")
        self.assertIn(f"не менее чем у {share}% значений", caption)
        self.assertIn(f"не больше ±{band}%", caption)
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

    def test_horizons_notes_carry_the_generators_numbers_and_caveats(self) -> None:
        horizons = self.landing["horizons"]
        notes = dict(zip(horizons["list"], horizons["notes"]))
        top = self.summary["MAE"].idxmin()
        by_horizon_model = self.horizons_summary.set_index(["horizon", "model"])
        # Наукаст: выигрыш лучшей модели к эталону — число из CSV.
        h1_gain = by_horizon_model.loc[(1, top), "к Prophet, %"]
        self.assertIn(f"{_fmt(h1_gain, 1)}%", notes[1])
        # Основной горизонт: тот же выигрыш и та же оговорка о фолдах, что в первой карточке.
        main = horizons["main"]
        self.assertIn(f"{_fmt(self.summary.loc[top, 'к Prophet, %'], 1)}%", notes[main])
        sentence = re.search(r"(?:Но выигрыш|Выигрыш) держится [^.]*\.", self.stats[0]["caption"])
        self.assertIsNotNone(sentence, "в первой карточке нет оговорки о фолдах")
        self.assertIn(sentence.group(0), notes[main])
        # Подпись основного горизонта открыта по умолчанию, то есть читатель видит её первой: в ней определены
        # фолд (с длиной окна) и скользящий origin, а в остальных подписях этих определений нет.
        self.assertIn(f"Фолд — проверочное окно в {main} мес.: модель учится на месяцах до него и прогнозирует его.",
                      notes[main])
        self.assertIn("Origin — месяц, от которого строится прогноз", notes[main])
        self.assertIn("(скользящий origin)", notes[main])
        for horizon, note in notes.items():
            if horizon != main:
                with self.subTest(definitions_only_in_the_first=horizon):
                    self.assertNotIn("Фолд — проверочное окно", note)
                    self.assertNotIn("Origin — месяц", note)
        # Год: оговорка про фолд — число из horizons_folds.csv, как у третьей карточки.
        n_folds = int(self.horizons_folds.loc[
            (self.horizons_folds["horizon"] == 12) & (self.horizons_folds["model"] == "two_stage"), "MAE"
        ].notna().sum())
        if 0 < n_folds < 3:
            words = {1: "один", 2: "два"}[n_folds]
            self.assertIn(f"но это {words} {_plural(n_folds, 'фолд', 'фолда', 'фолдов')}", notes[12])
        # Число фолдов каждого горизонта — в его пояснении, из двух независимых источников: конфига
        # и результатов. Среднее по двум фолдам и по девяти — разные по весу утверждения.
        config_folds = {int(item["horizon"]): int(item["n_folds"]) for item in self.horizons_cfg["horizons"]}
        result_folds = self.horizons_folds.groupby("horizon")["fold"].nunique()
        self.assertEqual(sorted(config_folds), horizons["list"])
        for horizon, note in notes.items():
            with self.subTest(horizon=horizon):
                self.assertEqual(int(result_folds[horizon]), config_folds[horizon])
                self.assertEqual(note.count("Фолдов:"), 1)
                self.assertIn(f"Фолдов: {config_folds[horizon]}.", note)
        # Пояснение «модель не удалось обучить» — только если на этом горизонте у какой-то из
        # четырёх моделей MAE нет: полосы нет, и читатель должен знать почему.
        year = self.horizons_summary[self.horizons_summary["horizon"] == 12].set_index("model")["MAE"]
        shown = [m["id"] for m in horizons["models"]]
        has_gap = any(pd.isna(year.get(model_id, np.nan)) for model_id in shown)
        self.assertEqual("не удалось обучить" in notes[12], has_gap)

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
        # Три МО блока «Стенд» (TEASER_MO): МО стенда по умолчанию, Казань и ряд-омоним с номером «#2».
        self.assertEqual(
            [item["id"] for item in teaser["items"]],
            [self.index_json["default_mo"], "городской округ город Казань", "Михайловский муниципальный район #2"],
        )
        for item in teaser["items"]:
            with self.subTest(series=item["id"]):
                self.assertEqual(set(item), TEASER_ITEM_KEYS)
                entry = mo_files[item["id"]]
                for key in ("fact", "forecast", "known", "breaks"):
                    self.assertEqual(item[key], entry[key])
                self.assertEqual(item["region"], regions[item["id"]])
                self.assertTrue(item["short"])
                self.assertNotIn(None, item["fact"] + item["forecast"] + item["known"])

    def test_teaser_rejects_missing_series_and_gaps(self) -> None:
        empty = build_site.DemoBuild(files={}, total_bytes=0, entries={}, regions={})
        with self.assertRaises(ValueError):
            build_site.build_teaser(empty, self.wide, self.forward_cfg)

        entries = {sid: {"fact": [1], "forecast": [1], "known": [1], "breaks": []}
                   for sid, _ in build_site.TEASER_MO}
        entries[build_site.TEASER_MO[0][0]]["known"] = [None]
        gappy = build_site.DemoBuild(
            files={}, total_bytes=0, entries=entries, regions={sid: None for sid in entries}
        )
        with self.assertRaises(ValueError):
            build_site.build_teaser(gappy, self.wide, self.forward_cfg)

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

    def test_substitutions_outside_the_stat_cards_sit_in_their_own_nodes(self) -> None:
        # Подстановки вне пяти карточек привязаны к узлам по id: подмена `n_series_rub` на `n_months`
        # в подзаголовке обложки, в шаге 01 или в блоке «Стенд», `forecast_year` на другое число в
        # заголовке раздела 03 или в подписи ползунка тест видит, а не «число нашлось где-то на странице».
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
        # Пустое поле ведёт на пример: его имя называет подпись под полем, а не догадка пользователя.
        self.assertIn(f"Пустой запрос откроет пример — {self.index_json['default_mo']}.", nodes.text("mo-note"))
        self.assertIn(f"ещё и на факте {year} года", nodes.text("hero-lead"))
        self.assertIn("прогноз федерального ряда", nodes.text("hero-lead"))
        # Линия подписана не только цветом: «толстая линия», а не «жёлтая».
        self.assertEqual(
            nodes.text("hero-caption"),
            f"{sample_label}\u00a0· расходы к среднему за {base}\u00a0· толстая линия — общее движение (медиана)",
        )
        # Шаг 01: число рядов и период — и в тексте шага, и в подписи под графиком.
        self.assertIn(f"{sample_label} из {n_series}: расходы к среднему за {base}.", nodes.text("step-1"))
        self.assertIn(f"от его среднего за {base}.", nodes.attr("step-1", "data-caption"))
        # Прогнозирует первый этап — по федеральному ряду; медиана рядов только показывает общее движение.
        self.assertIn("Его прогнозирует первый этап модели", nodes.text("step-2"))
        self.assertIn(f"Любой из {n_series} муниципалитетов", nodes.text("stand-lead"))
        # Изломы — это точки структурных изменений (термин конкурса), а пунктир объяснён в лиде, а не только в легенде.
        self.assertIn("изломы ряда (точки структурных изменений)", nodes.text("stand-lead"))
        self.assertIn("Пунктир — доли муниципалитета по прошлым месяцам, умноженные на опубликованный федеральный индекс",
                      nodes.text("stand-lead"))
        self.assertIn(f"прогноз на {year} год", nodes.text("stand-lead"))
        self.assertEqual(nodes.text("fact-title"), f"Прогноз от {origin_label} против факта {year} года")
        self.assertIn(f"за весь {year} год", nodes.text("fact-lead"))
        self.assertEqual(nodes.text("fact-k-label"), f"Месяц {year}")
        self.assertEqual(nodes.text("material-forecast"), f"Прогноз на {year} год")
        meta = re.search(r'<meta name="description" content="([^"]*)"', self.html).group(1)
        self.assertIn(f"Прогноз потребительских расходов {n_series} муниципальных образований", meta)
        self.assertIn(f"прогноз на {year} год", meta)

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

    def test_story_proof_links_to_the_error_decomposition_section_of_the_report(self) -> None:
        nodes = _IdsOf(self.html)
        text = nodes.text("story-proof")
        self.assertIn("межрядовая составляющая ошибки не двигается ни одной из моделей", text)
        self.assertIn("для которых посчитано разложение", text)
        # В строке нет числа ни цифрой, ни словом: «ни одной из шести моделей» пишет отчёт, страница — без числа.
        self.assertEqual(re.findall(r"\d", text), [])
        self.assertEqual(_number_words_in(text), [])
        link = re.search(r'<a href="(report/report\.html#[^"]+)"[^>]*>', self.html[self.html.index('id="story-proof"'):])
        self.assertIsNotNone(link)
        # Якорь — раздел отчёта о механизме: в отрендеренном отчёте такой id есть.
        anchor = unquote(link.group(1).split("#", 1)[1])
        report_html = (ROOT / "report" / "report.html").read_text(encoding="utf-8")
        report = _IdCollector()
        report.feed(report_html)
        self.assertIn(anchor, report.ids)
        self.assertIn(f'<h3 class="anchored" data-anchor-id="{anchor}">Механизм: ошибка состоит из двух частей', report_html)

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

    def test_breaks_section_says_what_the_bars_are_and_what_they_are_not(self) -> None:
        nodes = _IdsOf(self.html)
        caption = nodes.text("breaks-caption")
        # Подпись: найдено задним числом, не сигнал в реальном времени; метод и штраф — из конфига и PENALTY.
        self.assertIn("Найдено задним числом, по полному ряду (PELT на темпах роста, штраф 1)", caption)
        self.assertIn("это не сигнал в реальном времени", caption)
        self.assertIn("Штраф — настройка детектора: чем он выше, тем меньше изломов находится.", caption)
        # Изломы названы точками структурных изменений — термином конкурса.
        self.assertEqual(nodes.text("breaks-title"), "Изломы — точки структурных изменений")
        self.assertIn("точки структурных изменений — изломы", nodes.text("breaks-lead"))
        # Раздел стоит после проверки фактом и перед блоком прогноза по муниципалитету; ссылка ведёт
        # в раздел отчёта об обнаружении (якорь проверяет тест ссылок), в новой вкладке.
        self.assertLess(self.html.index('id="fact"'), self.html.index('id="breaks"'))
        self.assertLess(self.html.index('id="breaks"'), self.html.index('id="city"'))
        link = re.search(r'<a href="(report/report\.html#sec-cp)" target="_blank" rel="noopener">', self.html)
        self.assertIsNotNone(link, "нет ссылки на раздел отчёта об обнаружении")
        report_ids = _IdCollector()
        report_ids.feed((ROOT / "report" / "report.html").read_text(encoding="utf-8"))
        self.assertIn("sec-cp", report_ids.ids)

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
                        if anchor["text"].strip():  # у ссылки-картинки текста нет
                            self.assertIn("↗", anchor["text"], href)
                            self.assertIn("(откроется в новой вкладке)", anchor["text"], href)
                    elif not href.startswith(("http://", "https://")):
                        self.assertIsNone(anchor["target"], f"переход внутри сайта в новой вкладке: {href}")
                self.assertGreaterEqual(documents, 3)
        # В меню главной и страницы прогноза: отчёт, слайды и код — в новой вкладке; сами страницы сайта — нет.
        landing_nav = {a["href"]: a for a in _anchors(self.html)}
        self.assertIsNone(landing_nav["demo/"]["target"])
        self.assertEqual(landing_nav["report/slides.html"]["target"], "_blank")

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
            if name == "шаблон главной":  # раздел «Изломы» и схема метода говорят о стенде с врезками — проверка их видит
                self.assertGreaterEqual(len(found), 2)
        # И в строках скриптов — тексты, которые видит читатель: ошибки, подсказки, подписи.
        for name in ("site/landing.js", "demo/demo.js"):
            code = (ROOT / name).read_text(encoding="utf-8")
            code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
            code = "\n".join(line.split("//")[0] if "://" not in line else line for line in code.splitlines())
            with self.subTest(file=name):
                self.assertEqual(re.findall(r".{0,40}[Сс]тенд.{0,30}", code), [])

    def test_demo_page_is_named_forecast_by_municipality_on_both_pages(self) -> None:
        # Одно имя: «Прогноз по муниципалитету» — в меню, метке раздела и кнопках главной, в карточке материалов,
        # в метке и заголовке страницы. Приглашение «Покажите мой город» остаётся: поле обложки, H1 страницы.
        self.assertIn('<a class="nav-link" href="demo/">Прогноз по муниципалитету</a>', self.html)
        self.assertIn('<p class="eyebrow">05 · Прогноз по муниципалитету</p>', self.html)
        self.assertIn('<a class="material-link" href="demo/">Прогноз по муниципалитету</a>', self.html)
        self.assertEqual(self.html.count(">Открыть прогноз →</a>"), 2, "кнопки обложки и блока ведут на страницу прогноза")
        self.assertIn('<label class="search-label" for="mo-q">Покажите мой город</label>', self.html)
        self.assertIn('<h2 class="section-title" id="city-title">Покажите мой город</h2>', self.html)
        self.assertIn("<title>Прогноз по муниципалитету — покажите мой город</title>", self.demo_html)
        self.assertIn('<p class="eyebrow">Прогноз по муниципалитету</p>', self.demo_html)
        self.assertIn('<h1 class="stand-title" id="stand-title">Покажите мой город</h1>', self.demo_html)
        demo_js = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        self.assertIn("document.title = `${seriesId} — прогноз по муниципалитету`;", demo_js)
        landing_js = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        self.assertIn('"открыть прогноз →"', landing_js)

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
        note = re.search(r'<p class="material-note">(все ряды панели[^<]*)</p>', self.html).group(1)
        self.assertEqual(note, f"все ряды панели · csv.gz · {size('results/forecast_2025.csv.gz')}, архив gzip")

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
        # Поиск — ARIA-комбобокс со списком и живой областью; подписи под графиками — живые.
        combo = ids.ids["mo-q"]
        self.assertEqual(combo["role"], "combobox")
        self.assertEqual(combo["aria-controls"], "mo-list")
        self.assertEqual(ids.ids["mo-list"]["role"], "listbox")
        for live in ("mo-status", "story-caption", "h-note"):
            with self.subTest(live=live):
                self.assertEqual(ids.ids[live]["aria-live"], "polite")
        # Графики с данными — картинки с названием; декоративный график обложки скрыт.
        for chart in ("story-plot", "fact-plot", "breaks-plot", "t-plot"):
            with self.subTest(chart=chart):
                self.assertEqual(ids.ids[chart]["role"], "img")
                self.assertTrue(ids.ids[chart]["aria-label"])
        self.assertIn('class="hero-plot" aria-hidden="true"', self.html)
        # Шаги и переключатели — кнопки с состоянием.
        self.assertEqual(len(re.findall(r'<button class="step" id="step-\d" type="button" aria-pressed=', self.html)), 3)
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


if __name__ == "__main__":
    unittest.main()
