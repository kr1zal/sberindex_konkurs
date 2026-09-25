"""Прогон и сводка `src/run.py` без прогона моделей: синтетические кадры,
файлы — только во временном каталоге."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import run  # noqa: E402
from src.split import rolling_origin  # noqa: E402

SERIES = ["мо_0", "мо_1", "мо_2", "мо_3", "мо_4"]


def frame(models: list[str], series: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"model": m, "fold": f, "mo": s, "mae": 1.0, "error": None}
        for m in models for s in series for f in (0, 1, 2)
    ])


def printed(per_series: pd.DataFrame) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run._warn_uneven(per_series)
    return out.getvalue()


class WarnUnevenTest(unittest.TestCase):
    def test_uneven_file_prints_series_counts_per_model(self):
        # Раскладка инцидента 21.09: одна модель на части рядов, остальные на всех.
        per_series = pd.concat(
            [frame(["drift", "ets_damped"], SERIES), frame(["naive_last"], SERIES[:3])],
            ignore_index=True,
        )
        text = printed(per_series)
        self.assertTrue(text.startswith("ВНИМАНИЕ:"))
        self.assertIn("3: naive_last", text)
        self.assertIn("5: drift, ets_damped", text)

    def test_even_file_prints_nothing(self):
        self.assertEqual(printed(frame(["drift", "naive_last"], SERIES)), "")


class _Silent:
    """Модель, падающая исключением без текста: так падает голый `assert` в библиотеке."""

    def fit(self, *args):
        raise AssertionError()


class FailureTextTest(unittest.TestCase):
    """Пустой текст отказа уходит в CSV пустым полем и читается обратно как успех."""

    def test_series_model_failure_names_exception_type(self):
        folds = rolling_origin(24, 3, 1)
        task = ("silent", "мо_0", np.arange(24, dtype=float), pd.RangeIndex(24), folds, 3)
        with mock.patch.object(run, "_build_model", return_value=_Silent()):
            rows = run._score_series(task)
        self.assertEqual([row["error"] for row in rows], ["AssertionError: "])


class SummariseTest(unittest.TestCase):
    def test_row_without_mae_counts_as_refusal_even_with_empty_error(self):
        frame = pd.DataFrame({
            "model": ["m", "m"], "fold": [0, 0], "mo": ["мо_0", "мо_1"],
            "mae": [10.0, np.nan], "r2": [0.5, np.nan], "smape": [1.0, np.nan],
            "mase": [0.5, np.nan], "error": [None, ""],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_series.csv"
            frame.to_csv(path, index=False)
            read_back = pd.read_csv(path)  # пустой текст отказа возвращается как NaN
        row = run.summarise(read_back).loc["m"]
        self.assertEqual(row["отказов"], 1)
        self.assertEqual(row["серий"], 1)
        self.assertEqual(row["MAE"], 10.0)


if __name__ == "__main__":
    unittest.main()
