"""Новостные признаки в `GlobalGBM` (`global_gbm_news`) и `scripts/news_panel.py`.

Синтетика, без файлов реального размера. Четыре группы:

- причинность и состав новостных признаков (`_load_news` в `src/models/global_model.py`);
- ленивость загрузки — файл читает только модель с `news_path`, и только раз на контекст;
- форма и арифметика `results/news_panel.csv` (парное сравнение моделей по MAE);
- форма `results/news_breaks_corr.csv` и дизайн `within_month` против `pooled`
  на подсаженном сигнале — ключевая проверка конструкции 2.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.global_model import (  # noqa: E402
    GlobalGBM, NEWS_TOPICS as MODEL_NEWS_TOPICS, PanelContext, _load_news, _news_row,
)

# scripts/ — не пакет: скрипт грузится по пути, как в tests/test_news.py.
_spec = importlib.util.spec_from_file_location("news_panel", ROOT / "scripts" / "news_panel.py")
news_panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(news_panel)


def _monthly_frame(rows: list[dict]) -> pd.DataFrame:
    """Кадр вида data/news/monthly.parquet: региона, месяц и одиннадцать колонок."""
    base = {"intensity": 0.0, "n_articles": 1, "n_outlets": 1}
    for topic in MODEL_NEWS_TOPICS:
        base.setdefault(topic, 0.0)
    frame = pd.DataFrame([{**base, **row} for row in rows])
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


class NewsTopicsConsistencyTest(unittest.TestCase):
    """Список тем не должен расходиться между моделью и скриптом сравнения."""

    def test_model_and_script_agree_on_topics(self):
        self.assertEqual(MODEL_NEWS_TOPICS, news_panel.NEWS_TOPICS)
        self.assertEqual(len(MODEL_NEWS_TOPICS), 6)


class LoadNewsCausalityTest(unittest.TestCase):
    """`_load_news`: причинность и состав признаков."""

    def test_feature_at_month_t_unaffected_by_corrupting_later_months(self):
        months = pd.period_range("2023-01", periods=4, freq="M")
        regions = {"a": "Регион"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monthly.parquet"
            rows = [
                {"region_name": "Регион", "month": m.to_timestamp(), "t_ceny": 0.1 * (i + 1)}
                for i, m in enumerate(months)
            ]
            _monthly_frame(rows).to_parquet(path)
            before = _load_news(path, regions, months)

            # t=1 (февраль) — опорный месяц. Портим всё, что после него (t=2,3).
            corrupted = pd.read_parquet(path)
            mask = pd.to_datetime(corrupted["month"]) > months[1].to_timestamp()
            for topic in MODEL_NEWS_TOPICS:
                corrupted.loc[mask, topic] = 999.0
            corrupted.to_parquet(path)
            after = _load_news(path, regions, months)

        self.assertEqual(before[("Регион", 0)], after[("Регион", 0)])
        self.assertEqual(before[("Регион", 1)], after[("Регион", 1)])
        # Проверка на валидность самой порчи: то, что после опорного месяца, обязано отличаться.
        self.assertNotEqual(before[("Регион", 2)], after[("Регион", 2)])
        self.assertNotEqual(before[("Регион", 3)], after[("Регион", 3)])

    def test_exactly_six_topic_features_no_volume_or_intensity(self):
        months = pd.period_range("2023-01", periods=1, freq="M")
        regions = {"a": "Регион"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monthly.parquet"
            row = {"region_name": "Регион", "month": months[0].to_timestamp(),
                   "intensity": 5.0, "n_articles": 100, "n_outlets": 7}
            row |= {topic: 0.5 for topic in MODEL_NEWS_TOPICS}
            _monthly_frame([row]).to_parquet(path)
            loaded = _load_news(path, regions, months)

        keys = set(loaded[("Регион", 0)])
        self.assertEqual(keys, {f"news_{t}" for t in MODEL_NEWS_TOPICS})
        self.assertEqual(len(keys), 6)
        self.assertTrue(all(not k.endswith("intensity") for k in keys))

    def test_region_absent_from_the_file_gets_no_entry(self):
        months = pd.period_range("2023-01", periods=1, freq="M")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monthly.parquet"
            _monthly_frame([{"region_name": "Регион", "month": months[0].to_timestamp()}]).to_parquet(path)
            loaded = _load_news(path, {"a": "Регион", "b": "Другой регион"}, months)
        self.assertIn(("Регион", 0), loaded)
        self.assertEqual({r for r, _ in loaded}, {"Регион"})


class LazyLoadingTest(unittest.TestCase):
    """`GlobalGBM.set_context`: файл читает только модель с `news_path`."""

    def _context(self, regions: dict[str, str]) -> PanelContext:
        return PanelContext(index=pd.period_range("2023-01", periods=4, freq="M"), regions=regions)

    def test_default_model_never_touches_read_parquet(self):
        model = GlobalGBM()  # news_path не задан — как у всех остальных вариантов
        with mock.patch("pandas.read_parquet") as reader:
            model.set_context(self._context({"a": "Регион"}))
        reader.assert_not_called()
        self.assertIsNone(model.news)

    def test_news_model_reads_file_once_even_across_repeated_set_context(self):
        frame = _monthly_frame([{"region_name": "Регион", "month": "2023-01-01"}])
        model = GlobalGBM(news_path="data/news/monthly.parquet")
        context = self._context({"a": "Регион"})
        with mock.patch("pandas.read_parquet", return_value=frame) as reader:
            model.set_context(context)
            model.set_context(context)
        reader.assert_called_once()

    def test_series_without_a_region_gets_no_news_features(self):
        frame = _monthly_frame([{"region_name": "Регион", "month": "2023-01-01", "t_ceny": 0.7}])
        model = GlobalGBM(news_path="data/news/monthly.parquet")
        with mock.patch("pandas.read_parquet", return_value=frame):
            model.set_context(self._context({"a": "Регион"}))  # "b" останется без региона

        with_region = _news_row(model.news, model.regions.get("a"), 0)
        without_region = _news_row(model.news, model.regions.get("b"), 0)
        expected = {f"news_{t}": 0.0 for t in MODEL_NEWS_TOPICS} | {"news_t_ceny": 0.7}
        self.assertEqual(with_region, expected)
        self.assertEqual(without_region, {})


# ---------------------------------------------------------------------------
# Конструкция 1: results/news_panel.csv
# ---------------------------------------------------------------------------


def _per_series_rows(mae: dict[tuple[str, str, int], float | None]) -> pd.DataFrame:
    """`mae[(model, series, fold)]`; значение None — отказ (та же строка, что пишет run.py)."""
    rows = []
    for (model, series, fold), value in mae.items():
        failed = value is None
        rows.append({
            "model": model, "fold": fold, "mo": series,
            "mae": np.nan if failed else value,
            "error": "ValueError: нет обучающих примеров" if failed else np.nan,
        })
    return pd.DataFrame(rows)


# Ручной пример: s1 и s3 — новости помогают на большинстве фолдов, s2 — без изменений
# в среднем. Числа продуманы так, чтобы среднее по фолдам совпадало со средним по парам
# (в каждом фолде поровну рядов), и результат можно свести на бумаге.
_BASE = {
    ("global_gbm", "s1", 0): 10, ("global_gbm", "s1", 1): 12, ("global_gbm", "s1", 2): 14,
    ("global_gbm", "s2", 0): 20, ("global_gbm", "s2", 1): 18, ("global_gbm", "s2", 2): 22,
    ("global_gbm", "s3", 0): 30, ("global_gbm", "s3", 1): 28, ("global_gbm", "s3", 2): 26,
    ("global_gbm_news", "s1", 0): 8, ("global_gbm_news", "s1", 1): 11, ("global_gbm_news", "s1", 2): 15,
    ("global_gbm_news", "s2", 0): 19, ("global_gbm_news", "s2", 1): 20, ("global_gbm_news", "s2", 2): 21,
    ("global_gbm_news", "s3", 0): 32, ("global_gbm_news", "s3", 1): 25, ("global_gbm_news", "s3", 2): 24,
}


class PairedRowsTest(unittest.TestCase):
    def test_refusal_drops_only_that_pair_not_the_whole_series(self):
        mae = dict(_BASE)
        # s4: global_gbm отказывает на фолде 0, оба считаются на фолдах 1 и 2.
        mae[("global_gbm", "s4", 0)] = None
        mae[("global_gbm_news", "s4", 0)] = 5.0
        mae[("global_gbm", "s4", 1)] = 5.0
        mae[("global_gbm_news", "s4", 1)] = 4.0
        mae[("global_gbm", "s4", 2)] = 5.0
        mae[("global_gbm_news", "s4", 2)] = 4.0
        frame = _per_series_rows(mae)

        pairs = news_panel.paired_rows(frame)
        self.assertEqual(len(pairs), 11)  # 9 из s1..s3 + 2 из s4 (фолд 0 выпал)
        self.assertFalse(((pairs["mo"] == "s4") & (pairs["fold"] == 0)).any())
        s4_folds = set(pairs.loc[pairs["mo"] == "s4", "fold"])
        self.assertEqual(s4_folds, {1, 2})

    def test_subset_filter_keeps_only_named_series(self):
        frame = _per_series_rows(_BASE)
        pairs = news_panel.paired_rows(frame, subset={"s1", "s2"})
        self.assertEqual(set(pairs["mo"]), {"s1", "s2"})
        self.assertEqual(len(pairs), 6)


class PairStatsArithmeticTest(unittest.TestCase):
    """Проверка на бумаге посчитанного примера — числа сверены руками (см. _BASE)."""

    def setUp(self):
        frame = _per_series_rows(_BASE)
        self.pairs = news_panel.paired_rows(frame)
        self.stats = news_panel.pair_stats(self.pairs, horizon=3, subset_name="all")

    def test_counts(self):
        self.assertEqual(self.stats["n_series"], 3)
        self.assertEqual(self.stats["n_pairs"], 9)
        self.assertEqual(self.stats["n_folds"], 3)

    def test_means_and_diff(self):
        self.assertAlmostEqual(self.stats["mae_base"], 20.0, places=6)
        self.assertAlmostEqual(self.stats["mae_news"], 175 / 9, places=6)
        self.assertAlmostEqual(self.stats["diff_mean"], -5 / 9, places=6)
        # По построению фолды сбалансированы: среднее по фолдам совпадает со средним по парам.
        self.assertAlmostEqual(self.stats["fold_diff_mean"], self.stats["diff_mean"], places=6)

    def test_fold_ci_uses_t_with_folds_minus_one_degrees_of_freedom(self):
        # Средние по фолдам: -1/3, -2/3, -2/3 (см. докстринг модуля теста).
        low, high = news_panel._t_ci(np.array([-1 / 3, -2 / 3, -2 / 3]))
        self.assertAlmostEqual(self.stats["fold_ci_low"], low, places=6)
        self.assertAlmostEqual(self.stats["fold_ci_high"], high, places=6)
        self.assertLess(low, self.stats["fold_diff_mean"])
        self.assertGreater(high, self.stats["fold_diff_mean"])

    def test_share_improved_and_folds_won(self):
        # s1: news 11.33 < base 12 — улучшился. s2: 20 == 20 — нет (строго "меньше").
        # s3: news 27 < base 28 — улучшился. Итого 2 из 3.
        self.assertAlmostEqual(self.stats["share_series_improved"], 2 / 3, places=6)
        # На всех трёх фолдах среднее по рядам MAE с новостями ниже (см. docstring теста).
        self.assertEqual(self.stats["folds_won"], 3)


class BuildNewsPanelTest(unittest.TestCase):
    def test_columns_and_row_count_and_horizon_filter(self):
        per_series = _per_series_rows(_BASE)
        horizons_rows = []
        for (model, series, fold), value in _BASE.items():
            horizons_rows.append({"model": model, "horizon": 1, "fold": fold, "mo": series,
                                   "mae": value, "error": np.nan})
            # Мусорная строка на другом горизонте: build_news_panel не должна её видеть.
            horizons_rows.append({"model": model, "horizon": 6, "fold": fold, "mo": series,
                                   "mae": 10_000.0, "error": np.nan})
        horizons_per_series = pd.DataFrame(horizons_rows)

        table = news_panel.build_news_panel(per_series, horizons_per_series, covered_series={"s1", "s2", "s3"})

        self.assertEqual(list(table.columns), news_panel.NEWS_PANEL_COLUMNS)
        self.assertEqual(len(table), 4)  # 2 горизонта × 2 подмножества
        self.assertEqual(set(table["horizon"]), {1, 3})
        self.assertEqual(set(table["subset"]), {"covered", "all"})
        # h=1 взят из горизонтного файла, а не из мусорной строки horizon=6.
        h1_all = table.loc[(table["horizon"] == 1) & (table["subset"] == "all")].iloc[0]
        self.assertLess(h1_all["mae_base"], 1000)


# ---------------------------------------------------------------------------
# Конструкция 2: results/news_breaks_corr.csv
# ---------------------------------------------------------------------------


class AllowedBreakMonthsTest(unittest.TestCase):
    def test_matches_worked_example_min_size_3_on_24_months(self):
        index = pd.period_range("2023-01", periods=24, freq="M")
        allowed = news_panel.allowed_break_months(index, min_size=3)
        self.assertEqual(len(allowed), 19)
        self.assertEqual(allowed.min(), pd.Period("2023-04", "M"))
        self.assertEqual(allowed.max(), pd.Period("2024-10", "M"))

    def test_smaller_min_size_widens_the_range(self):
        index = pd.period_range("2023-01", periods=24, freq="M")
        allowed = news_panel.allowed_break_months(index, min_size=1)
        self.assertEqual(len(allowed), 23)  # [1, 24-1] включительно


class CheckBreaksWithinRangeTest(unittest.TestCase):
    def _cp_series(self, months: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "protocol": ["v2"] * len(months), "penalty": [1.0] * len(months),
            "series_id": [f"s{i}" for i in range(len(months))], "month": months,
        })

    def test_passes_when_all_breaks_inside(self):
        allowed = pd.period_range("2023-04", "2024-10", freq="M")
        news_panel.check_breaks_within_range(self._cp_series(["2023-05", "2024-10"]), allowed)  # не падает

    def test_raises_when_a_break_is_outside(self):
        allowed = pd.period_range("2023-04", "2024-10", freq="M")
        with self.assertRaises(ValueError):
            news_panel.check_breaks_within_range(self._cp_series(["2023-05", "2023-01"]), allowed)

    def test_ignores_other_penalties(self):
        allowed = pd.period_range("2023-04", "2024-10", freq="M")
        cp_series = self._cp_series(["2023-05"])
        extra = pd.DataFrame({"protocol": ["v2"], "penalty": [3.0], "series_id": ["sx"], "month": ["2023-01"]})
        news_panel.check_breaks_within_range(pd.concat([cp_series, extra]), allowed)  # штраф 3.0 не проверяется


class CorrRowsShapeTest(unittest.TestCase):
    def test_shape_columns_and_bonferroni_alpha(self):
        rng = np.random.default_rng(1)
        regions = [f"r{i}" for i in range(5)]
        months = pd.period_range("2023-04", periods=6, freq="M")
        rows = []
        for region in regions:
            for month in months:
                row = {"region_name": region, "month": month, "break_share": rng.random()}
                for topic in news_panel.NEWS_TOPICS:
                    for lag in news_panel.LAGS:
                        row[f"{topic}_lag{lag}"] = rng.random()
                rows.append(row)
        panel = pd.DataFrame(rows)

        table = news_panel.corr_rows(panel)
        self.assertEqual(list(table.columns), news_panel.BREAKS_CORR_COLUMNS)
        self.assertEqual(len(table), 24)  # 2 дизайна × 6 признаков × 2 лага
        self.assertEqual(set(table["design"]), {"within_month", "pooled"})
        self.assertTrue((table["alpha"] == 0.05 / 12).all())

        within = table.loc[table["design"] == "within_month"]
        pooled = table.loc[table["design"] == "pooled"]
        self.assertTrue(within["passed"].map(lambda v: isinstance(v, (bool, np.bool_))).all())
        self.assertTrue(pooled["passed"].isna().all())
        # within теряет на центрировании по месяцу больше степеней свободы, чем pooled.
        self.assertTrue((within["df"].to_numpy() < pooled["df"].to_numpy()).all())


class WithinMonthDesignTest(unittest.TestCase):
    """Ключевая проверка дизайна: календарное совпадение не должно выглядеть региональной
    связью внутри месяца, а подсаженная региональная связь должна находиться. Числа
    зафиксированы одним прогоном (seed=12345) — см. комментарии внизу."""

    def setUp(self):
        rng = np.random.default_rng(12345)
        g, m = 12, 8
        month_trend = rng.normal(size=m)
        region_trait = rng.normal(size=g)
        months = np.repeat(np.arange(m), 1)  # заполнится ниже в порядке region-then-month

        confound_x, confound_y, signal_x, signal_y, group_months = [], [], [], [], []
        for gi in range(g):
            for mi in range(m):
                confound_x.append(month_trend[mi] + rng.normal())
                confound_y.append(month_trend[mi] + rng.normal())
                signal_x.append(region_trait[gi] + rng.normal())
                signal_y.append(region_trait[gi] + rng.normal())
                group_months.append(mi)
        self.months = np.array(group_months)
        self.confound_x, self.confound_y = np.array(confound_x), np.array(confound_y)
        self.signal_x, self.signal_y = np.array(signal_x), np.array(signal_y)
        self.alpha = news_panel.BONFERRONI_ALPHA

    def test_calendar_only_confound_vanishes_within_month_but_not_pooled(self):
        within = news_panel.design_stats(self.confound_x, self.confound_y, self.months, "within_month", self.alpha)
        pooled = news_panel.design_stats(self.confound_x, self.confound_y, self.months, "pooled", self.alpha)

        # Пул видит общий календарный узор как связь; within — почти ничего, и не проходит порог.
        self.assertGreater(abs(pooled["r"]), 0.4)
        self.assertLess(abs(within["r"]), 0.3)
        self.assertLess(abs(within["r"]), abs(pooled["r"]))
        self.assertGreater(within["p"], self.alpha)

    def test_planted_regional_signal_is_found_within_month(self):
        within = news_panel.design_stats(self.signal_x, self.signal_y, self.months, "within_month", self.alpha)

        self.assertGreater(abs(within["r"]), 0.4)
        self.assertLess(within["p"], self.alpha)
        self.assertGreater(abs(within["r"]), within["r_critical"])


if __name__ == "__main__":
    unittest.main()
