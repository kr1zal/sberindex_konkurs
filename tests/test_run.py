"""Сигнал о разном числе рядов у моделей в `src/run.py`.

Модели не гоняются: проверяется только то, что печатается по готовому кадру.
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import run  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
