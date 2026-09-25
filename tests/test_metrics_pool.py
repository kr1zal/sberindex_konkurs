"""R² по пулу и медиане, выигрыш к эталонам: `src/metrics.py`.

Кадры синтетические, прогнозы известны: R² пула сверяется с R², посчитанным
напрямую по объединённым фактическим значениям и прогнозам всех строк модели.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    GAINS, R2_MEDIAN, R2_POOL, ROW_METRICS, r2, r2_and_gains, row_metrics,
)

PROPHET, NAIVE = "к Prophet, %", "к наивной, %"
# Уровни рядов различаются на порядки, как у муниципалитетов: пул держится
# на разбросе уровней, а R² пары ряд × фолд — на трёх точках вокруг своего среднего.
LEVELS = [100.0, 1_000.0, 5_000.0, 20_000.0]
NOISE = {"best": 0.01, "prophet": 0.05, "naive_last": 0.08}


def windows(horizon: int = 3, n_folds: int = 3, seed: int = 7) -> list[dict]:
    """Строки ряд × фолд трёх моделей с фактом и прогнозом: y — тест ряда, p — прогноз."""
    rng = np.random.default_rng(seed)
    out = []
    for s, level in enumerate(LEVELS):
        for fold in range(n_folds):
            y = level * (1 + 0.1 * rng.standard_normal(horizon))
            y_train = level * (1 + 0.1 * rng.standard_normal(12))
            for model, noise in NOISE.items():
                p = y + level * noise * rng.standard_normal(horizon)
                out.append({"model": model, "fold": fold, "mo": f"мо_{s}", "y": y, "p": p,
                            "y_train": y_train})
    return out


def rows(items: list[dict]) -> pd.DataFrame:
    """Строки в раскладке файла результатов: метрики — той же функцией, что в прогонах."""
    return pd.DataFrame([
        {"model": w["model"], "fold": w["fold"], "mo": w["mo"],
         **row_metrics(w["y"], w["p"], w["y_train"]), "error": None}
        for w in items
    ])


def direct_pool(items: list[dict], model: str) -> float:
    """R² напрямую: все точки модели одним массивом, SST вокруг среднего этого массива."""
    y = np.concatenate([w["y"] for w in items if w["model"] == model])
    p = np.concatenate([w["p"] for w in items if w["model"] == model])
    return 1.0 - float(np.sum((y - p) ** 2)) / float(np.sum((y - y.mean()) ** 2))


def by_model(ok: pd.DataFrame) -> pd.DataFrame:
    """Колонки сводки по моделям — как в `src/run.py::summarise`."""
    return r2_and_gains(ok, ok.groupby("model")["mae"].mean())


class RowMetricsTest(unittest.TestCase):
    def test_row_carries_metrics_and_pool_parts_in_file_order(self):
        y, p = np.array([10.0, 12.0, 17.0]), np.array([11.0, 11.0, 15.0])
        row = row_metrics(y, p, np.array([8.0, 9.0, 10.0]))
        self.assertEqual(list(row), list(ROW_METRICS))
        self.assertEqual(row["sse"], 1.0 + 1.0 + 4.0)
        self.assertAlmostEqual(row["sst"], float(np.sum((y - 13.0) ** 2)), places=12)
        self.assertEqual(row["n"], 3)
        self.assertEqual(row["y_sum"], 39.0)
        self.assertEqual(row["y_sq"], 100.0 + 144.0 + 289.0)
        self.assertEqual(row["r2"], r2(y, p))
        self.assertAlmostEqual(row["r2"], 1.0 - row["sse"] / row["sst"], places=12)


class PoolTest(unittest.TestCase):
    def setUp(self):
        self.items = windows()
        self.ok = rows(self.items)

    def test_pool_equals_r2_over_all_points_of_the_model(self):
        table = by_model(self.ok)
        self.assertEqual(list(table.columns), [R2_POOL, R2_MEDIAN, *GAINS])
        for model in NOISE:
            with self.subTest(model):
                self.assertAlmostEqual(table.loc[model, R2_POOL], direct_pool(self.items, model), places=12)

    def test_median_is_the_median_of_row_r2_not_the_mean(self):
        # Одна строка с почти ровным тестом: её R² уходит в минус тысячи, тянет
        # за собой среднее по строкам, а медиану не сдвигает.
        items = self.items + [{
            "model": "best", "fold": 0, "mo": "мо_ровный", "y": np.array([500.0, 500.01, 500.0]),
            "p": np.array([501.0, 499.0, 501.0]), "y_train": np.full(12, 500.0) + np.arange(12),
        }]
        ok = rows(items)
        r2_rows = ok.loc[ok["model"] == "best", "r2"]
        self.assertLess(r2_rows.min(), -1000)
        median = by_model(ok).loc["best", R2_MEDIAN]
        self.assertEqual(median, float(np.median(r2_rows)))
        self.assertGreater(median, r2_rows.mean() + 100)

    def test_rows_without_sse_are_left_out_of_every_sum(self):
        # Сумма без строки и сумма с NaN вместо её sse должны совпасть: её точки
        # не входят ни в SSE, ни в SST пула, хотя n, y_sum и y_sq у неё есть.
        hole = (self.ok["model"] == "best") & (self.ok["mo"] == "мо_3") & (self.ok["fold"] == 1)
        ok = self.ok.copy()
        ok.loc[hole, ["sse", "r2"]] = np.nan
        self.assertTrue(ok.loc[hole, ["n", "y_sum", "y_sq"]].notna().all().all())
        kept = [w for w, drop in zip(self.items, hole) if not drop]
        table = by_model(ok)
        self.assertAlmostEqual(table.loc["best", R2_POOL], direct_pool(kept, "best"), places=12)
        self.assertEqual(table.loc["best", R2_MEDIAN], float(ok.loc[ok["model"] == "best", "r2"].median()))

    def test_pool_of_identical_points_is_nan_like_row_r2(self):
        ok = rows([{"model": "m", "fold": f, "mo": "мо_0", "y": np.full(3, 7.0), "p": np.full(3, 6.0),
                    "y_train": np.arange(12, dtype=float)} for f in range(2)])
        self.assertTrue(np.isnan(by_model(ok).loc["m", R2_POOL]))

    def test_file_without_r2_and_parts_gives_nan_r2_and_keeps_gains(self):
        # Раскладка старого horizons_per_series.csv: ни r2, ни слагаемых.
        old = self.ok.drop(columns=["r2", "sse", "sst", "n", "y_sum", "y_sq"])
        table = by_model(old)
        self.assertTrue(table[[R2_POOL, R2_MEDIAN]].isna().all().all())
        self.assertTrue(table[[PROPHET, NAIVE]].notna().all().all())

    def test_median_comes_from_r2_even_without_pool_parts(self):
        # Раскладка per_series.csv до слагаемых: r2 пар ряд × фолд есть, пула не собрать.
        # Медиана от слагаемых не зависит и считается; NaN — только пул.
        old = self.ok.drop(columns=["sse", "sst", "n", "y_sum", "y_sq"])
        table = by_model(old)
        self.assertTrue(table[R2_POOL].isna().all())
        for model in NOISE:
            with self.subTest(model):
                self.assertEqual(table.loc[model, R2_MEDIAN], old.loc[old["model"] == model, "r2"].median())


class GainsTest(unittest.TestCase):
    def test_gain_is_share_of_reference_mae(self):
        mae = pd.Series({"best": 80.0, "prophet": 100.0, "naive_last": 125.0}, name="MAE")
        mae.index.name = "model"
        table = r2_and_gains(pd.DataFrame(columns=["model", "mae"]), mae)
        self.assertAlmostEqual(table.loc["best", PROPHET], 20.0)
        self.assertEqual(table.loc["prophet", PROPHET], 0.0)
        self.assertAlmostEqual(table.loc["naive_last", PROPHET], -25.0)
        self.assertAlmostEqual(table.loc["best", NAIVE], 36.0)
        self.assertAlmostEqual(table.loc["prophet", NAIVE], 20.0)

    def test_without_reference_the_gain_is_nan(self):
        mae = pd.Series({"best": 80.0, "naive_last": 125.0}, name="MAE")
        mae.index.name = "model"
        table = r2_and_gains(pd.DataFrame(columns=["model", "mae"]), mae)
        self.assertTrue(table[PROPHET].isna().all())
        self.assertAlmostEqual(table.loc["best", NAIVE], 36.0)

    def test_reference_is_taken_within_the_group(self):
        # Горизонт 1 со своим эталоном, горизонт 3 без эталона: чужой горизонт
        # эталоном не становится.
        mae = pd.Series(
            [50.0, 40.0, 100.0, 90.0],
            index=pd.MultiIndex.from_tuples(
                [(1, "prophet"), (1, "best"), (3, "naive_last"), (3, "best")], names=["horizon", "model"],
            ),
            name="MAE",
        )
        table = r2_and_gains(pd.DataFrame(columns=["horizon", "model", "mae"]), mae)
        self.assertAlmostEqual(table.loc[(1, "best"), PROPHET], 20.0)
        self.assertTrue(np.isnan(table.loc[(3, "best"), PROPHET]))
        self.assertAlmostEqual(table.loc[(3, "best"), NAIVE], 10.0)
        self.assertTrue(np.isnan(table.loc[(1, "best"), NAIVE]))

    def test_model_without_mae_gets_nan_gain(self):
        # Модель из одних отказов: MAE — NaN, выигрыш — NaN, а не ошибка.
        mae = pd.Series({"prophet": 100.0, "refused": np.nan}, name="MAE")
        mae.index.name = "model"
        table = r2_and_gains(pd.DataFrame(columns=["model", "mae"]), mae)
        self.assertTrue(np.isnan(table.loc["refused", PROPHET]))
        self.assertTrue(np.isnan(table.loc["refused", R2_POOL]))


if __name__ == "__main__":
    unittest.main()
