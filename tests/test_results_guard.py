"""Страховка от тихой перезаписи результатов: `src/results_guard.py`.

Кадры синтетические. Живые файлы из `results/` тесты не читают и не пишут:
путь нужен проверке только для текста сообщения.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import results_guard  # noqa: E402
from src.results_guard import check_plan, check_same_panel, uneven_series  # noqa: E402

PATH = Path("results/per_series.csv")
SERIES = ["мо_0", "мо_1", "мо_2", "мо_3", "мо_4"]
INTENTIONAL = (
    "Если протокол сменён намеренно, задайте в конфиге другой `output.dir` "
    "(для horizons.py — другой `output.prefix`), а не убирайте из каталога один файл "
    "по рядам: у `horizons_steps.csv` своей сверки нет, и новые шаги слились бы со старыми"
)
HINT = (
    "партия считана на другом протоколе или выборке (например, `configs/baseline.yaml` "
    "с 300 рядами); в файл ничего не записано. " + INTENTIONAL
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


class CheckPlanTest(unittest.TestCase):
    """Та же сверка, что для партии, но до прогона: план — ряды матрицы и номера фолдов."""

    def test_same_plan_passes(self):
        check_plan(frame(["naive_last", "drift"], SERIES, [0, 1, 2]), SERIES, [0, 1, 2], path=PATH)

    def test_plan_on_subset_of_series_is_rejected(self):
        # 300 рядов пилота против 2 028 в файле — в миниатюре.
        previous = frame(["naive_last", "drift"], SERIES, [0, 1, 2])
        with self.assertRaises(ValueError) as caught:
            check_plan(previous, SERIES[:3], [0, 1, 2], path=PATH)
        message = str(caught.exception)
        self.assertIn(str(PATH), message)
        self.assertIn("рядов: в файле 5, в плане 3", message)
        self.assertIn("фолды в плане: {0, 1, 2}", message)
        self.assertIn("configs/baseline.yaml", message)
        self.assertIn("модели не запускались", message)
        self.assertIn(INTENTIONAL, message)

    def test_plan_with_other_folds_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            check_plan(frame(["naive_last"], SERIES, [0, 1, 2]), SERIES, [0, 1], path=PATH)
        self.assertIn("фолды в плане: {0, 1}", str(caught.exception))

    def test_plan_folds_are_compared_per_horizon(self):
        previous = pd.concat(
            [frame(["naive_last"], SERIES, [0, 1, 2, 3], horizon=1),
             frame(["naive_last"], SERIES, [0, 1, 2], horizon=3)],
            ignore_index=True,
        )
        with self.subTest("горизонта в файле ещё нет"):
            check_plan(previous, SERIES, {12: [0]}, path=PATH)
        with self.subTest("горизонты файла с теми же фолдами и новый горизонт"):
            check_plan(previous, SERIES, {1: [0, 1, 2, 3], 3: [0, 1, 2], 12: [0]}, path=PATH)
        with self.subTest("горизонт уже в файле, фолды другие"):
            with self.assertRaises(ValueError) as caught:
                check_plan(previous, SERIES, {3: [0, 1]}, path=PATH)
            self.assertIn("другие фолды на горизонте 3", str(caught.exception))
            self.assertIn("фолды в плане: h3 {0, 1}", str(caught.exception))

    def test_empty_file_passes(self):
        previous = frame(["naive_last"], SERIES, [0, 1, 2]).iloc[0:0]
        check_plan(previous, SERIES[:3], [0, 1], path=PATH)


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


class RefusalRuleTest(unittest.TestCase):
    """Правило отказа живёт в лёгком модуле: отчёт читает по нему файлы результатов,
    и импорт правила не должен поднимать `src.run` со всеми моделями."""

    def test_rule_imports_without_run_and_models(self):
        # В отдельном процессе: в этом `src.run` уже загружен другими тестами.
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from src.results_guard import failure_reason, refused; "
            "heavy = ('src.run', 'src.models', 'torch'); "
            "print(' '.join(sorted(m for m in sys.modules if m.startswith(heavy))))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "")

    def test_run_keeps_exporting_the_same_rule(self):
        # `src.run.refused` и импорт в scripts/horizons.py должны работать как раньше.
        from src import run

        self.assertIs(run.refused, results_guard.refused)
        self.assertIs(run.failure_reason, results_guard.failure_reason)

    def test_row_without_mae_or_with_text_is_a_refusal(self):
        frame = pd.DataFrame({"mae": [1.0, float("nan"), 2.0], "error": [None, None, "ValueError: x"]})
        self.assertEqual(results_guard.refused(frame).tolist(), [False, True, True])

    def test_failure_reason_names_exception_type(self):
        self.assertEqual(results_guard.failure_reason(AssertionError()), "AssertionError: ")
        self.assertEqual(results_guard.failure_reason(ValueError("мало точек")), "ValueError: мало точек")


class ReadResultsTest(unittest.TestCase):
    """Файлы результатов читаются числами ровно такими, какими они записаны."""

    def test_numbers_come_back_exactly_as_written(self):
        # Разбор pandas по умолчанию читает примерно каждое восьмое такое число
        # на единицу последнего разряда не тем, что записано.
        values = np.concatenate([[1217.032436222938, 0.6207988257547953],
                                 np.random.default_rng(1).uniform(0, 5000, 200)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_series.csv"
            pd.DataFrame({"mae": values}).to_csv(path, index=False)
            back = results_guard.read_results(path)
        self.assertTrue(np.array_equal(back["mae"].to_numpy(), values))

    def test_runs_and_backfill_read_results_only_through_it(self):
        # Правило одно на все чтения: прямое чтение в одном месте разошлось бы с ним молча.
        for name in ("src/run.py", "scripts/horizons.py", "scripts/backfill_r2.py"):
            with self.subTest(name):
                direct = (ROOT / name).read_text(encoding="utf-8").count("read_csv(")
                self.assertEqual(direct, 0, f"{name}: прямых чтений CSV — {direct}")


if __name__ == "__main__":
    unittest.main()
