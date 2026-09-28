"""Генератор страницы входа и данных стенда: `scripts/build_site.py`.

Сборка (единственная дорогая часть — чтение `per_series.csv` и матрицы панели, доли
секунды) идёт один раз в `setUpClass`, во временный каталог (`--out`), чтобы не трогать
закоммиченные `index.html`/`demo/data/`. Числа страницы входа этот файл считает заново,
короткими выражениями pandas по тем же `results/*.csv` — независимо от генератора: тест,
вызывающий функции самого генератора, проверял бы только то, что код совпадает сам
с собой, а не то, что число на странице верное.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

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
    "n_no_region", "n_homonym_names", "n_homonym_series", "default_mo",
    "forecast_rule", "known_model", "breaks", "folds", "models", "panel_mae", "series",
}
AGGREGATE_JSON_KEYS = {
    "unit", "origin", "horizon", "model", "history", "check", "rule_names", "mape", "mape_by_horizon",
}

_MONTH_OF = ["января", "февраля", "марта", "апреля", "мая", "июня",
             "июля", "августа", "сентября", "октября", "ноября", "декабря"]
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
    """Ресурсы, которые браузер грузит сам (скрипты и стили) — не обычные ссылки
    `<a href>`: те вправе вести куда угодно (репозиторий, дашборд СберИндекса)."""

    RESOURCE_ATTR = {"script": "src", "link": "href"}

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

        # demo/index.html — рукописный файл этой задачи, генератор его не пишет
        # (в отличие от index.html и demo/data/**), поэтому читаем закоммиченный
        # файл репозитория, а не то, что build_site.main() положил во временный out_dir.
        cls.demo_html_path = ROOT / "demo" / "index.html"
        cls.demo_html = cls.demo_html_path.read_text(encoding="utf-8")

        # Источники чисел — те же results/*.csv и конфиги, что читает генератор,
        # но независимо: свои pandas-выражения, не вызов build_site.compute_placeholders.
        cls.full_cfg = yaml.safe_load((ROOT / "configs" / "full.yaml").read_text(encoding="utf-8"))
        cls.forward_cfg = yaml.safe_load(
            (ROOT / "configs" / "forecast_forward.yaml").read_text(encoding="utf-8")
        )
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

    def test_aggregate_json_top_level_keys(self) -> None:
        self.assertEqual(set(self.aggregate_json), AGGREGATE_JSON_KEYS)

    def test_no_template_placeholders_left(self) -> None:
        self.assertNotIn("${", self.html)

    # -- число 1: горизонт основного протокола ------------------------------

    def test_headline_gain_and_r2_match_summary_csv(self) -> None:
        top = self.summary["MAE"].idxmin()
        prophet_mae = self.summary.loc["prophet", "MAE"]
        best_mae = self.summary.loc[top, "MAE"]
        best_gain = -self.summary.loc[top, "к Prophet, %"]
        r2_prophet = self.summary.loc["prophet", "R² пул"]
        r2_best = self.summary.loc[top, "R² пул"]

        self.assertIn(str(int(self.full_cfg["split"]["horizon"])), self.html)
        self.assertIn(f"{_fmt(prophet_mae, 0)} ₽", self.html)
        self.assertIn(f"{_fmt(best_mae, 0)} ₽", self.html)
        self.assertIn(f"{_fmt(best_gain, 1)}%", self.html)
        self.assertIn(_fmt(r2_prophet, 3), self.html)
        self.assertIn(_fmt(r2_best, 3), self.html)

    def test_folds_caveat_matches_per_series_csv(self) -> None:
        top = self.summary["MAE"].idxmin()
        fold_mae = (self.ok[self.ok["model"].isin([top, "prophet"])]
                    .groupby(["fold", "model"])["mae"].mean().unstack())
        folds = list(fold_mae.index)
        won = [f for f in folds if fold_mae.loc[f, top] < fold_mae.loc[f, "prophet"]]
        lost = [f for f in folds if fold_mae.loc[f, top] > fold_mae.loc[f, "prophet"]]

        self.assertIn(f"{len(won)} {'фолде' if len(won) == 1 else 'фолдах'} из {len(folds)}", self.html)
        for fold in lost:
            with self.subTest(fold=fold):
                self.assertIn(_ORDINAL.get(fold, str(fold)), self.html)

    # -- число 2: наукаст (горизонт 1) --------------------------------------

    def test_nowcast_matches_horizons_summary_csv(self) -> None:
        top = self.summary["MAE"].idxmin()
        by_horizon_model = self.horizons_summary.set_index(["horizon", "model"])
        self.assertIn((1, top), by_horizon_model.index)
        h1_best = by_horizon_model.loc[(1, top), "MAE"]
        h1_prophet = by_horizon_model.loc[(1, "prophet"), "MAE"]
        h1_gain = by_horizon_model.loc[(1, top), "к Prophet, %"]

        self.assertIn(f"{_fmt(h1_best, 0)} против {_fmt(h1_prophet, 0)} ₽", self.html)
        self.assertIn(f"{_fmt(h1_gain, 1)}%", self.html)

    # -- число 3: год вперёд (горизонт 12, без оракула) ---------------------

    def test_year_ahead_matches_horizons_summary_and_folds_csv(self) -> None:
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

        self.assertIn(f"{_fmt(gain_naive, 0)}%", self.html)
        self.assertIn(f"{_fmt(gain_prophet, 0)}%", self.html)
        if 0 < n_folds_top < 3:
            self.assertIn(_plural(n_folds_top, "фолд", "фолда", "фолдов"), self.html)

    # -- число 4: проверка агрегата по факту 2025 года -----------------------

    def test_aggregate_check_matches_forecast_2025_aggregate_check_csv(self) -> None:
        abs_error = self.agg_check.assign(e=self.agg_check["error_pct"].abs())
        by_method_horizon = abs_error.groupby(["method", "horizon"])["e"].mean()
        own, rules = by_method_horizon.loc["two_stage"], by_method_horizon.drop(index="two_stage")

        def _range(series: pd.Series) -> str:
            lo, hi = series.min(), series.max()
            return _fmt(lo, 1) if _fmt(lo, 1) == _fmt(hi, 1) else f"{_fmt(lo, 1)}–{_fmt(hi, 1)}"

        horizons = sorted(int(h) for h in self.agg_check["horizon"].unique())
        origin = pd.Period(self.forward_cfg["origin"], "M")

        self.assertIn(f"{_range(own)}%", self.html)
        self.assertIn(f"{_range(rules)}%", self.html)
        self.assertIn(f"{horizons[0]}–{horizons[-1]}", self.html)
        self.assertIn(f"{_MONTH_OF[origin.month - 1]} {origin.year}", self.html)

    def test_forecast_year_is_origin_year_plus_one(self) -> None:
        # «2025» на странице — не хардкод, а origin.year + 1: страница входа ссылается
        # на прогноз ИМЕННО этого года (forecast_2025.csv), а не любого следующего.
        origin = pd.Period(self.forward_cfg["origin"], "M")
        self.assertIn(str(origin.year + 1), self.html)

    # -- число 5: форма панели ------------------------------------------------

    def test_panel_shape_matches_build_matrix(self) -> None:
        n_series, n_months = self.wide.shape[1], self.wide.shape[0]
        expected = (
            f"{_fmt(n_series, 0)} {_plural(n_series, 'ряд', 'ряда', 'рядов')} × "
            f"{n_months} {_plural(n_months, 'месяц', 'месяца', 'месяцев')}"
        )
        self.assertIn(expected, self.html)

        start, end = self.wide.index[0], self.wide.index[-1]
        self.assertIn(str(start.year), self.html)
        self.assertIn(str(end.year), self.html)

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

    # -- размер данных стенда ----------------------------------------------

    def test_demo_data_size_is_at_or_under_the_module_threshold(self) -> None:
        total = sum(p.stat().st_size for p in self.data_dir.rglob("*") if p.is_file())
        self.assertLessEqual(total, build_site.MAX_DEMO_BYTES)

    # -- ссылки страницы входа -----------------------------------------------

    def test_relative_links_resolve_to_existing_repository_files(self) -> None:
        collector = _LinkCollector()
        collector.feed(self.html)
        self.assertTrue(collector.links, "в index.html не нашлось ни одной ссылки")

        for link in collector.links:
            if link.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            with self.subTest(link=link):
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
        # Бриф: «ни одна страница не грузит внешних скриптов и стилей» — ни index.html
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


if __name__ == "__main__":
    unittest.main()
