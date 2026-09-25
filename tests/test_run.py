"""Прогон и сводка `src/run.py` без прогона моделей: синтетические кадры,
файлы — только во временном каталоге."""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import run  # noqa: E402
from src.split import rolling_origin  # noqa: E402

SERIES = ["мо_0", "мо_1", "мо_2", "мо_3", "мо_4"]


def frame(models: list[str], series: list[str], mae: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame([
        {"model": m, "fold": f, "mo": s, "mae": mae, "error": None}
        for m in models for s in series for f in (0, 1, 2)
    ])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


class MergeResultsTest(unittest.TestCase):
    """Слияние партии с `per_series.csv`: 5 рядов × 2 модели в файле."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "per_series.csv"
        frame(["naive_last", "drift"], SERIES).to_csv(self.path, index=False)

    def test_batch_on_other_series_is_rejected_and_file_untouched(self):
        before = digest(self.path)
        with self.assertRaises(ValueError):
            run.merge_results(self.path, frame(["naive_last"], SERIES[:3], mae=2.0))
        self.assertEqual(digest(self.path), before)

    def test_batch_replaces_rows_of_its_model_and_keeps_the_rest(self):
        merged = run.merge_results(self.path, frame(["naive_last"], SERIES, mae=2.0))
        on_disk = pd.read_csv(self.path)
        self.assertEqual(len(on_disk), 2 * len(SERIES) * 3)
        self.assertEqual(on_disk.loc[on_disk["model"] == "naive_last", "mae"].unique().tolist(), [2.0])
        self.assertEqual(on_disk.loc[on_disk["model"] == "drift", "mae"].unique().tolist(), [1.0])
        self.assertEqual(len(merged), len(on_disk))


class _Report:
    def as_text(self) -> str:
        return ""


def _model_started(*args, **kwargs):
    raise AssertionError("модель запущена до сверки плана с файлом")


class MainStopsBeforeModelsTest(unittest.TestCase):
    """`main()` целиком: загрузка подменена, модели — ловушки, которые падают при вызове."""

    def test_plan_on_other_series_stops_before_any_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            per_path = root / "results" / "per_series.csv"
            per_path.parent.mkdir()
            frame(["naive_last", "drift"], SERIES).to_csv(per_path, index=False)
            before = digest(per_path)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "data": {"path": "panel.parquet", "category": "Все категории", "max_gap": 2},
                "split": {"horizon": 3, "n_folds": 3},
                "sample": {"n_series": None, "seed": 1},
                "models": ["naive_last"],
                "output": {"dir": "results"},
                "compute": {"workers": 1},
            }, allow_unicode=True), encoding="utf-8")
            # Матрица на 3 рядах из 5, что лежат в файле: пилот против полной панели.
            wide = pd.DataFrame(np.ones((24, 3)), columns=SERIES[:3])
            with (
                mock.patch.object(run, "ROOT", root),
                mock.patch.object(run, "load_panel", return_value=None),
                mock.patch.object(run, "build_matrix", return_value=(wide, _Report())),
                mock.patch.object(run, "build_context", return_value=None),
                mock.patch.object(run, "evaluate", side_effect=_model_started) as evaluate,
                mock.patch.object(run, "evaluate_global", side_effect=_model_started) as panel,
                mock.patch.object(sys, "argv", ["run.py", "--config", str(config)]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(ValueError) as caught:
                    run.main()
            self.assertIn("модели не запускались", str(caught.exception))
            evaluate.assert_not_called()
            panel.assert_not_called()
            self.assertEqual(digest(per_path), before)


if __name__ == "__main__":
    unittest.main()
