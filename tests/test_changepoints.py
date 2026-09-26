"""Детекторы в штрафном режиме: `src/changepoints.py`. Ряды синтетические."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.changepoints import DETECTORS  # noqa: E402

PENALIZED = ["pelt", "binseg", "window", "bottomup", "kernel_rbf"]
# Штраф, при котором чистая синусоида целиком не окупает ни одного излома, а сдвиг в 4σ
# окупает. У методов l2 штраф в единицах дисперсии ряда; у ядра стоимость в пространстве
# ядра, и масштаб другой — поэтому на стенде его штраф калибруется, а не берётся как есть.
TEST_PENALTY = {"pelt": 10.0, "binseg": 10.0, "window": 10.0, "bottomup": 10.0, "kernel_rbf": 4.0}


def sine(n: int = 24) -> np.ndarray:
    """Чистая сезонность без шума и без изломов: период 12, как у месячных расходов."""
    t = np.arange(n, dtype=float)
    return 100.0 + 10.0 * np.sin(2 * np.pi * t / 12)


class DetectorPenaltyModeTest(unittest.TestCase):
    """Все пять методов со штрафом решают обнаружение: число изломов не задано заранее."""

    def test_clean_sine_has_no_breaks(self):
        # С одним заданным заранее изломом он находился всегда — сегментация, а не обнаружение.
        for name in PENALIZED:
            with self.subTest(name):
                found = DETECTORS[name](sine(), penalty=TEST_PENALTY[name]).breakpoints
                self.assertEqual(found, [])

    def test_level_shift_of_four_sigma_is_found_near_it(self):
        y = sine()
        y[12:] += 4 * np.std(y)
        for name in PENALIZED:
            with self.subTest(name):
                found = DETECTORS[name](y, penalty=TEST_PENALTY[name]).breakpoints
                self.assertTrue(found)
                self.assertTrue(all(abs(b - 12) <= 1 for b in found), found)


if __name__ == "__main__":
    unittest.main()
