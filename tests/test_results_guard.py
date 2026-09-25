"""Страховка от тихой перезаписи результатов: `src/results_guard.py`.

Кадры синтетические. Живые файлы из `results/` тесты не читают и не пишут:
путь нужен проверке только для текста сообщения.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.results_guard import check_same_panel, uneven_series  # noqa: E402

PATH = Path("results/per_series.csv")
SERIES = ["мо_0", "мо_1", "мо_2", "мо_3", "мо_4"]
HINT = (
    "партия считана на другом протоколе или выборке (например, `configs/baseline.yaml` "
    "с 300 рядами); в файл ничего не записано"
)


def frame(
    models: list[str], series: list[str], folds: list[int],
    horizon: int | None = None, error: str | None = None,
) -> pd.DataFrame:
    """Строки ряд × фолд для каждой модели, в раскладке `per_series.csv`."""
    out = pd.DataFrame([
        {"model": m, "fold": f, "mo": s, "mae": 1.0, "error": error}
        for m in models for s in series for f in folds
    ])
    if horizon is not None:
        out.insert(1, "horizon", horizon)
    return out


class CheckSamePanelTest(unittest.TestCase):
    def test_same_series_and_folds_pass(self):
        previous = frame(["naive_last", "drift"], SERIES, [0, 1, 2])
        check_same_panel(previous, frame(["naive_last"], SERIES, [0, 1, 2]), path=PATH)

    def test_batch_on_subset_of_series_is_rejected(self):
        # Инцидент 21.09 в миниатюре: 300 рядов из 2 028 против полнопанельного файла.
        previous = frame(["naive_last", "drift"], SERIES, [0, 1, 2])
        fresh = frame(["naive_last"], SERIES[:3], [0, 1, 2])
        with self.assertRaises(ValueError) as caught:
            check_same_panel(previous, fresh, path=PATH)
        message = str(caught.exception)
        self.assertIn("рядов: в файле 5, в партии 3", message)
        self.assertIn("только в файле 2, только в партии 0", message)
        self.assertIn(str(PATH), message)
        self.assertIn(HINT, message)

    def test_other_folds_without_horizon_are_rejected(self):
        previous = frame(["naive_last"], SERIES, [0, 1, 2])
        fresh = frame(["drift"], SERIES, [0, 1])
        with self.assertRaises(ValueError) as caught:
            check_same_panel(previous, fresh, path=PATH)
        self.assertIn("фолды в файле: {0, 1, 2}", str(caught.exception))
        self.assertIn("фолды в партии: {0, 1}", str(caught.exception))

    def test_folds_are_compared_per_horizon(self):
        # У горизонтов разное число фолдов по построению: сверять множество целиком нельзя.
        previous = pd.concat(
            [frame(["naive_last"], SERIES, [0, 1, 2, 3], horizon=1),
             frame(["naive_last"], SERIES, [0, 1, 2], horizon=3)],
            ignore_index=True,
        )
        with self.subTest("горизонт уже в файле, фолды те же"):
            check_same_panel(previous, frame(["drift"], SERIES, [0, 1, 2], horizon=3), path=PATH)
        with self.subTest("горизонт уже в файле, фолды другие"):
            fresh = frame(["drift"], SERIES, [0, 1], horizon=3)
            with self.assertRaises(ValueError) as caught:
                check_same_panel(previous, fresh, path=PATH)
            self.assertIn("другие фолды на горизонте 3", str(caught.exception))
            self.assertIn("фолды в партии: h3 {0, 1}", str(caught.exception))
        with self.subTest("горизонта в файле ещё нет"):
            check_same_panel(previous, frame(["drift"], SERIES, [0, 1], horizon=6), path=PATH)

    def test_empty_previous_passes(self):
        previous = frame(["naive_last"], SERIES, [0, 1, 2]).iloc[0:0]
        check_same_panel(previous, frame(["naive_last"], SERIES[:3], [0, 1]), path=PATH)

    def test_empty_batch_is_rejected(self):
        # Партия без строк — аномалия сама по себе: прогон выглядел бы успешным.
        previous = frame(["naive_last"], SERIES, [0, 1, 2])
        with self.assertRaises(ValueError) as caught:
            check_same_panel(previous, pd.DataFrame([]), path=PATH)
        self.assertIn(str(PATH), str(caught.exception))
        self.assertIn("в файл ничего не записано", str(caught.exception))

    def test_series_of_file_include_other_models_and_refusals(self):
        # Ряд, на котором другая модель только отказывала, — всё равно ряд файла.
        previous = pd.concat(
            [frame(["naive_last", "prophet"], SERIES[:4], [0, 1, 2]),
             frame(["prophet"], SERIES[4:], [0, 1, 2], error="не сошлось")],
            ignore_index=True,
        )
        with self.assertRaises(ValueError):
            check_same_panel(previous, frame(["naive_last"], SERIES[:4], [0, 1, 2]), path=PATH)
        check_same_panel(previous, frame(["naive_last"], SERIES, [0, 1, 2]), path=PATH)


class UnevenSeriesTest(unittest.TestCase):
    def test_equal_counts_give_none(self):
        self.assertIsNone(uneven_series(frame(["naive_last", "drift"], SERIES, [0, 1, 2])))

    def test_model_with_fewer_series_is_reported(self):
        per_series = pd.concat(
            [frame(["drift"], SERIES, [0, 1, 2]), frame(["naive_last"], SERIES[:3], [0, 1, 2])],
            ignore_index=True,
        )
        uneven = uneven_series(per_series)
        self.assertIsNotNone(uneven)
        self.assertEqual(uneven.to_dict(), {"drift": 5, "naive_last": 3})

    def test_refusals_count_toward_series(self):
        # Модель, отказавшая на двух рядах, гонялась на тех же пяти, что и остальные.
        per_series = pd.concat(
            [frame(["drift"], SERIES, [0, 1, 2]),
             frame(["prophet"], SERIES[:3], [0, 1, 2]),
             frame(["prophet"], SERIES[3:], [0, 1, 2], error="не сошлось")],
            ignore_index=True,
        )
        self.assertIsNone(uneven_series(per_series))


if __name__ == "__main__":
    unittest.main()
