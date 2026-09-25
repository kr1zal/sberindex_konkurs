"""Слияние, отказы и сводка прогона по горизонтам: `scripts/horizons.py`.

Кадры синтетические, файлы пишутся только во временный каталог: живые
`results/horizons_*.csv` тесты не трогают.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.split import rolling_origin  # noqa: E402

# scripts/ — не пакет: скрипт грузится по пути, без правки sys.path под все тесты.
_spec = importlib.util.spec_from_file_location("horizons", ROOT / "scripts" / "horizons.py")
horizons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(horizons)

SERIES = ["мо_0", "мо_1", "мо_2"]
PAIR = {"model": "m", "horizon": 3}


def per_series(model: str, horizon: int, folds: list[int], series=SERIES, mae=1.0) -> pd.DataFrame:
    """Строки ряд × фолд в раскладке `horizons_per_series.csv`."""
    return pd.DataFrame([
        {"model": model, "horizon": horizon, "fold": f, "train_end": 12 + f * horizon, "mo": s,
         "mae": mae, "smape": 1.0, "mase": 1.0, "error": np.nan}
        for s in series for f in folds
    ])


def steps(model: str, horizon: int, folds: list[int], mae=1.0) -> pd.DataFrame:
    """Строки фолд × шаг в раскладке `horizons_steps.csv`: колонки `mo` в нём нет."""
    return pd.DataFrame([
        {"model": model, "horizon": horizon, "fold": f, "train_end": 12 + f * horizon,
         "step": k + 1, "mae": mae, "серий": len(SERIES)}
        for f in folds for k in range(horizon)
    ])


def pairs(frame: pd.DataFrame) -> list[tuple]:
    return sorted(set(zip(frame["model"], frame["horizon"])))


class MergeIntoTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_empty_fresh_drops_stale_rows_of_the_pair(self):
        # Модель отказала на всех фолдах горизонта: кадр шагов пуст, а старые шаги
        # этой пары из файла должны уйти — в per_series у неё теперь одни отказы.
        path = self.dir / "horizons_steps.csv"
        pd.concat([steps("m", 3, [0, 1]), steps("m", 1, [0, 1]), steps("other", 3, [0, 1])],
                  ignore_index=True).to_csv(path, index=False)
        horizons.merge_into(path, pd.DataFrame([]), PAIR)
        after = pd.read_csv(path)
        self.assertEqual(pairs(after), [("m", 1), ("other", 3)])
        self.assertEqual(len(after), len(steps("m", 1, [0, 1])) + len(steps("other", 3, [0, 1])))

    def test_file_keeps_header_when_its_only_pair_is_replaced_by_nothing(self):
        path = self.dir / "horizons_steps.csv"
        steps("m", 3, [0, 1]).to_csv(path, index=False)
        horizons.merge_into(path, pd.DataFrame([]), PAIR)
        after = pd.read_csv(path)
        self.assertTrue(after.empty)
        self.assertIn("mae", after.columns)

    def test_fresh_rows_replace_the_pair_and_keep_the_rest(self):
        path = self.dir / "horizons_per_series.csv"
        pd.concat([per_series("m", 3, [0, 1, 2]), per_series("other", 3, [0, 1, 2], mae=5.0)],
                  ignore_index=True).to_csv(path, index=False)
        horizons.merge_into(path, per_series("m", 3, [0, 1, 2], mae=2.0), PAIR, same_panel=True)
        after = pd.read_csv(path)
        self.assertEqual(len(after), 2 * len(SERIES) * 3)
        self.assertEqual(after.loc[after["model"] == "m", "mae"].unique().tolist(), [2.0])
        self.assertEqual(after.loc[after["model"] == "other", "mae"].unique().tolist(), [5.0])

    def test_batch_on_other_series_is_rejected_and_file_untouched(self):
        path = self.dir / "horizons_per_series.csv"
        per_series("m", 3, [0, 1, 2]).to_csv(path, index=False)
        before = path.read_bytes()
        fresh = per_series("m", 3, [0, 1, 2], series=SERIES[:2], mae=2.0)
        with self.assertRaises(ValueError):
            horizons.merge_into(path, fresh, PAIR, same_panel=True)
        self.assertEqual(path.read_bytes(), before)


class _Silent:
    """Модель, падающая исключением без текста: так падает голый `assert` в библиотеке."""

    def fit(self, *args):
        raise AssertionError()


class FailureTextTest(unittest.TestCase):
    """Пустой текст отказа уходит в CSV пустым полем и читается обратно как успех."""

    def test_series_model_failure_names_exception_type(self):
        folds = rolling_origin(24, 3, 1)
        task = ("silent", "мо_0", np.arange(24, dtype=float), pd.RangeIndex(24), folds, 3)
        with mock.patch.object(horizons, "_build_model", return_value=_Silent()):
            rows, _ = horizons._score_series(task)
        self.assertEqual([row["error"] for row in rows], ["AssertionError: "])

    def test_panel_model_failure_names_exception_type(self):
        folds = rolling_origin(24, 3, 1)
        wide = pd.DataFrame(np.ones((24, 2)), columns=["мо_0", "мо_1"])
        with mock.patch.dict(horizons.EXTRA_GLOBAL, {"silent": _Silent}), \
                contextlib.redirect_stdout(io.StringIO()):
            part, _ = horizons.evaluate_panel_model(wide, wide, "silent", 3, folds, context=None)
        self.assertEqual(part["error"].tolist(), ["AssertionError: ", "AssertionError: "])


class SummariseTest(unittest.TestCase):
    def test_row_without_mae_counts_as_refusal_even_with_empty_error(self):
        frame = pd.DataFrame({
            "model": ["m", "m"], "horizon": [3, 3], "fold": [0, 0], "train_end": [21, 21],
            "mo": ["мо_0", "мо_1"], "mae": [10.0, np.nan], "smape": [1.0, np.nan],
            "mase": [0.5, np.nan], "error": [None, ""],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "horizons_per_series.csv"
            frame.to_csv(path, index=False)
            read_back = pd.read_csv(path)  # пустой текст отказа возвращается как NaN
        by_fold, by_horizon = horizons.summarise(read_back)
        for table in (by_fold, by_horizon):
            row = table.iloc[0]
            self.assertEqual(row["отказов"], 1)
            self.assertEqual(row["серий"], 1)
            self.assertEqual(row["MAE"], 10.0)


class _Report:
    def as_text(self) -> str:
        return ""


def _model_started(*args, **kwargs):
    raise AssertionError("модель запущена до сверки плана с файлом")


class MainStopsBeforeModelsTest(unittest.TestCase):
    """`main()` целиком: загрузка подменена, модели — ловушки, которые падают при вызове."""

    def test_plan_on_other_series_stops_before_any_model(self):
        five = [f"мо_{i}" for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            per_path = root / "results" / "horizons_per_series.csv"
            per_path.parent.mkdir()
            per_series("m", 3, [0, 1, 2], series=five).to_csv(per_path, index=False)
            before = per_path.read_bytes()
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "data": {"path": "panel.parquet", "category": "Все категории", "max_gap": 2},
                "horizons": [{"horizon": 3, "n_folds": 3}],
                "sample": {"n_series": None, "seed": 1},
                "models": ["naive_last"],
                "output": {"dir": "results", "prefix": "horizons"},
                "compute": {"workers": 1},
            }, allow_unicode=True), encoding="utf-8")
            wide = pd.DataFrame(np.ones((24, 3)), columns=five[:3])
            with (
                warnings.catch_warnings(),  # main() глушит предупреждения глобально
                mock.patch.object(horizons, "ROOT", root),
                mock.patch.object(horizons, "load_panel", return_value=None),
                mock.patch.object(horizons, "build_matrix", return_value=(wide, _Report())),
                mock.patch.object(horizons, "build_context", return_value=None),
                mock.patch.object(horizons, "evaluate_series_models", side_effect=_model_started) as series,
                mock.patch.object(horizons, "evaluate_panel_model", side_effect=_model_started) as panel,
                mock.patch.object(sys, "argv", ["horizons.py", "--config", str(config)]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(ValueError) as caught:
                    horizons.main()
            self.assertIn("модели не запускались", str(caught.exception))
            series.assert_not_called()
            panel.assert_not_called()
            self.assertEqual(per_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
