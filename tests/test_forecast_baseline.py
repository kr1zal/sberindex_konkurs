"""Прогноз эталона на 2025 год по рядам панели: scripts/forecast_baseline.py.
Синтетика, настоящие данные не читаются; сама модель подменена: проверяется сборка
таблицы (месяцы от origin, строки ряда блоком, регион приклеен, отказ — строками с
причиной и пустым прогнозом), а не Prophet.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("forecast_baseline", ROOT / "scripts" / "forecast_baseline.py")
forecast_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(forecast_baseline)

SERIES = ["мо_а", "мо_б", "мо_в"]
CFG = {"origin": "2024-12", "baseline": {"model": "prophet", "horizon": 12, "output": "x.csv"}}


def make_wide() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=24, freq="MS")
    return pd.DataFrame({s: 1000.0 + 100 * i + np.arange(24) for i, s in enumerate(SERIES)}, index=index)


def make_regions() -> pd.DataFrame:
    return pd.DataFrame({
        "series_id": SERIES, "region": ["Регион А", None, "Регион В"], "oktmo": ["01", None, "03"],
    })


class _LastValue:
    """Подмена модели: прогноз — последнее значение ряда; ряд «мо_б» роняет подгонку."""

    def __init__(self) -> None:
        self.index = None
        self.last = None

    def set_index(self, index) -> "_LastValue":
        self.index = index
        return self

    def fit(self, y) -> "_LastValue":
        if np.isclose(y[0], 1100.0):  # мо_б
            raise RuntimeError("не сошлось")
        self.last = float(y[-1])
        return self

    def predict(self, horizon: int):
        return np.full(horizon, self.last)


class ForecastBaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(forecast_baseline, "_build_model", lambda name: _LastValue())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_table_shape_months_regions_and_failures(self) -> None:
        frame = forecast_baseline.forecast_baseline(make_wide(), make_regions(), CFG, workers=1)
        self.assertEqual(list(frame.columns), forecast_baseline.BASELINE_COLUMNS)
        self.assertEqual(len(frame), len(SERIES) * 12)
        months = [str(pd.Period("2024-12", "M") + k) for k in range(1, 13)]
        for series_id, group in frame.groupby("series_id", sort=False):
            with self.subTest(series=series_id):
                self.assertEqual(list(group["month"]), months)
                self.assertTrue((group["model"] == "prophet").all())
                self.assertTrue((group["horizon"] == 12).all())
        ok = frame[frame["series_id"] == "мо_а"]
        self.assertTrue(np.allclose(ok["forecast"], 1023.0))
        self.assertTrue(ok["error"].isna().all())
        self.assertEqual(ok["region"].iloc[0], "Регион А")
        failed = frame[frame["series_id"] == "мо_б"]
        self.assertTrue(failed["forecast"].isna().all())
        self.assertTrue(failed["error"].str.contains("не сошлось").all(), failed["error"].iloc[0])
        self.assertTrue(failed["region"].isna().all(), "ряд без региона не теряется при склейке")

    def test_origin_mismatch_is_an_error(self) -> None:
        wide = make_wide().iloc[:-1]
        with self.assertRaises(ValueError):
            forecast_baseline.forecast_baseline(wide, make_regions(), CFG, workers=1)

    def test_parallel_and_sequential_runs_agree(self) -> None:
        wide, regions = make_wide(), make_regions()
        one = forecast_baseline.forecast_baseline(wide, regions, CFG, workers=1)
        # Пул процессов импортирует модуль заново и подмены модели не видит, поэтому здесь
        # он подменён на последовательное выполнение: проверяется разбиение на задачи и сборка.
        class _Pool:
            def __init__(self, max_workers=None): pass
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def map(self, fn, tasks, chunksize=1): return map(fn, tasks)
        with mock.patch.object(forecast_baseline, "ProcessPoolExecutor", _Pool):
            many = forecast_baseline.forecast_baseline(wide, regions, CFG, workers=3)
        pd.testing.assert_frame_equal(one, many)


if __name__ == "__main__":
    unittest.main()
