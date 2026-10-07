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
import re
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

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
    "default_mo", "forecast_rule", "known_model", "breaks", "folds", "models",
    "panel_mae", "series",
}
AGGREGATE_JSON_KEYS = {
    "unit", "origin", "horizon", "model", "model_label",
    "history", "check", "rule_names", "mape",
}
LANDING_JSON_KEYS = {"story", "horizons", "fact", "teaser"}
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
                target = (ROOT / link).resolve()
                if target.is_dir():
                    self.assertTrue((target / "index.html").exists(), f"{link}: нет index.html внутри")
                else:
                    self.assertTrue(target.exists(), f"{link}: файла {target} нет")

    # -- ссылки и ресурсы стенда (demo/index.html) ----------------------------

    def test_demo_index_html_relative_links_resolve_to_existing_files(self) -> None:
        collector = _LinkCollector()
        collector.feed(self.demo_html)
        self.assertTrue(collector.links, "в demo/index.html не нашлось ни одной ссылки")

        for link in collector.links:
            if link.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            with self.subTest(link=link):
                # demo/index.html лежит в demo/, относительные ссылки — от этой папки
                # (а не от корня репозитория, как у index.html на верхнем уровне).
                target = (self.demo_html_path.parent / link).resolve()
                if target.is_dir():
                    self.assertTrue((target / "index.html").exists(), f"{link}: нет index.html внутри")
                else:
                    self.assertTrue(target.exists(), f"{link}: файла {target} нет")

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


    # -- данные главной: <script id="landing-data"> --------------------------------------

    def test_landing_data_is_one_parsable_json_script(self) -> None:
        self.assertIsNotNone(self.landing_raw, "в index.html нет ровно одного <script id=landing-data>")
        # «<» экранирован: иначе «</script» внутри строки данных закрыл бы тег.
        self.assertNotIn("<", self.landing_raw)
        self.assertEqual(set(self.landing), LANDING_JSON_KEYS)
        self.assertEqual(set(self.landing["story"]), STORY_KEYS)
        self.assertEqual(set(self.landing["horizons"]), HORIZONS_KEYS)
        self.assertEqual(set(self.landing["teaser"]), {"months", "items"})

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
        # Третий шаг: «у девяти значений из десяти … не больше ±N%» — N по всей панели.
        norm = self._story_norm()
        deviation = norm.div(norm.median(axis=1), axis=0).sub(1).abs().to_numpy()
        expected = _fmt(float(np.percentile(deviation, 90)) * 100, 0)
        self.assertIn(f"не больше ±{expected}%", self.html)

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
        # Год: оговорка про фолд — число из horizons_folds.csv, как у третьей карточки.
        n_folds = int(self.horizons_folds.loc[
            (self.horizons_folds["horizon"] == 12) & (self.horizons_folds["model"] == "two_stage"), "MAE"
        ].notna().sum())
        if 0 < n_folds < 3:
            words = {1: "один", 2: "два"}[n_folds]
            self.assertIn(f"но это {words} {_plural(n_folds, 'фолд', 'фолда', 'фолдов')}", notes[12])
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
        # Три МО из задания: МО стенда по умолчанию, Казань и ряд-омоним с номером «#2».
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

    def test_stylesheets_and_script_load_nothing_external(self) -> None:
        for name in ("site/site.css", "site/landing.css", "demo/demo.css"):
            css = (ROOT / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertNotRegex(css, r"url\(\s*['\"]?(?:https?:)?//", "внешний url() в стилях")
                self.assertNotIn("@import", css)
        js = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        # Единственный «http» в скрипте — пространство имён SVG, это не ресурс.
        urls = [u for u in re.findall(r"https?://[^\s\"'`)]+", js) if u != "http://www.w3.org/2000/svg"]
        self.assertEqual(urls, [], "landing.js не должен ходить за пределы сайта")
        for forbidden in ("XMLHttpRequest", "WebSocket", "sendBeacon", "import("):
            self.assertNotIn(forbidden, js)

    def test_local_assets_referenced_by_stylesheets_exist(self) -> None:
        for name in ("site/site.css", "site/landing.css"):
            css_path = ROOT / name
            for url in re.findall(r"url\(\s*['\"]?([^'\")\s]+)", css_path.read_text(encoding="utf-8")):
                with self.subTest(file=name, url=url):
                    self.assertTrue((css_path.parent / url).resolve().exists(), f"{url}: файла нет")

    def test_landing_script_has_no_hand_typed_data_numbers(self) -> None:
        js = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")

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
                        f"в landing.js вписано число данных {variant!r}",
                    )

        # И сами ряды: массив из шести и более чисел подряд — вписанные данные.
        self.assertIsNone(
            re.search(r"\[\s*-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*){5,}\]", js),
            "в landing.js есть числовой массив — данные должны приходить из landing-data",
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
        text = re.sub(r"\b0[1-6]\b", " ", text)
        text = text.replace("CC BY-SA 4.0", " ")
        leftovers = re.findall(r".{0,20}\d.{0,20}", text)
        self.assertEqual(leftovers, [], "в шаблоне вписаны числа; вынесите их в подстановки генератора")

    # -- разметка главной ---------------------------------------------------------------------------

    def test_page_has_landmarks_sections_and_accessible_controls(self) -> None:
        ids = _IdCollector()
        ids.feed(self.html)
        for section in ("why", "horizons", "fact", "city", "method", "report"):
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
        for chart in ("story-plot", "fact-plot", "t-plot"):
            with self.subTest(chart=chart):
                self.assertEqual(ids.ids[chart]["role"], "img")
                self.assertTrue(ids.ids[chart]["aria-label"])
        self.assertIn('class="hero-plot" aria-hidden="true"', self.html)
        # Шаги и переключатели — кнопки с состоянием.
        self.assertEqual(self.html.count('class="step" type="button" aria-pressed='), 3)
        self.assertIn('lang="ru"', self.html)
        self.assertIn('<link rel="icon" href="data:,">', self.html)
        self.assertIn('name="viewport"', self.html)
        self.assertIn('name="description"', self.html)

    def test_page_links_to_the_expected_materials(self) -> None:
        collector = _LinkCollector()
        collector.feed(self.html)
        for link in ("report/report.html", "report/slides.html", "report/report.pdf", "report/slides.pdf",
                     "demo/", "report/method.svg", "results/forecast_2025.csv.gz",
                     "https://github.com/kr1zal/sberindex_konkurs"):
            with self.subTest(link=link):
                self.assertIn(link, collector.links)


if __name__ == "__main__":
    unittest.main()
