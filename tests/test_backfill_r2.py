"""Бэкфилл слагаемых R² пула: `scripts/backfill_r2.py`.

Панель синтетическая — пять рядов с трендом и колебанием и один с ровным тестом
в первом фолде, — файлы результатов пишутся только во временный каталог. Метрики
строк посчитаны `src.metrics` по известным прогнозам, как их посчитал бы прогон:
бэкфилл должен восстановить из панели и r2 те же суммы, что дал бы прямой счёт.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import subprocess
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
from src.metrics import mae, mase, r2, smape  # noqa: E402
from src.models.naive import NaiveLast, SeasonalNaive  # noqa: E402
from src.results_guard import read_results  # noqa: E402
from src.split import Fold, rolling_origin  # noqa: E402

# scripts/ — не пакет: скрипт грузится по пути, как в tests/test_horizons.py.
_spec = importlib.util.spec_from_file_location("backfill_r2", ROOT / "scripts" / "backfill_r2.py")
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)

SERIES = ["мо_0", "мо_1", "мо_2", "мо_3", "мо_4", "мо_ровный"]
PARTS = ["sse", "sst", "n", "y_sum", "y_sq"]
BASE = ["model", "fold", "mo", "mae", "r2", "smape", "mase", "error"]
SUMMARY = ["MAE", "MASE", "sMAPE", "R² пул", "R² медиана", "к Prophet, %", "к наивной, %",
           "серий", "отказов"]
NEW_IN_SUMMARY = ["R² пул", "R² медиана", "к Prophet, %", "к наивной, %"]
FILES = ["per_series.csv", "summary.csv", "horizons_summary.csv", "horizons_folds.csv"]
FOLDS = rolling_origin(24, 3, 3)


def matrix() -> pd.DataFrame:
    """24 месяца × 6 рядов. У последнего тест первого фолда (месяцы 16..18) ровный:
    SST ряда ноль, r2 не определён, и sse из него не восстановить."""
    t = np.arange(24, dtype=float)
    wide = pd.DataFrame({s: 100 + (i + 1) * t + 5 * np.sin(t) for i, s in enumerate(SERIES[:-1])})
    flat = 100 + 2 * t
    flat[15:18] = 150.0
    wide[SERIES[-1]] = flat
    return wide


def forecasts(wide: pd.DataFrame) -> dict[tuple[str, str, int], np.ndarray]:
    """Прогнозы четырёх моделей на каждую пару ряд × фолд: две наивные — настоящими
    классами, две — случайные, их sse бэкфилл восстанавливает только из r2."""
    rng = np.random.default_rng(11)
    out = {}
    for col in wide.columns:
        series = wide[col].to_numpy()
        for fold in FOLDS:
            y_train = series[: fold.train_end]
            y_test = series[fold.test_start : fold.test_end]
            out[("naive_last", col, fold.index)] = NaiveLast().fit(y_train).predict(3)
            out[("seasonal_naive", col, fold.index)] = SeasonalNaive(season=12).fit(y_train).predict(3)
            for model, scale in (("prophet", 5.0), ("drift", 12.0)):
                out[(model, col, fold.index)] = y_test + scale * rng.standard_normal(3)
    return out


def per_series(wide: pd.DataFrame, predicted: dict) -> pd.DataFrame:
    """`per_series.csv` старой раскладки: без слагаемых пула. Один отказ у prophet."""
    rows = []
    for (model, col, fold_index), y_pred in predicted.items():
        fold = FOLDS[fold_index]
        series = wide[col].to_numpy()
        y_train, y_test = series[: fold.train_end], series[fold.test_start : fold.test_end]
        if (model, col, fold_index) == ("prophet", "мо_2", 1):
            rows.append({"model": model, "fold": fold_index, "mo": col, "mae": np.nan, "r2": np.nan,
                         "smape": np.nan, "mase": np.nan, "error": "ValueError: не сошлось"})
            continue
        rows.append({"model": model, "fold": fold_index, "mo": col, "mae": mae(y_test, y_pred),
                     "r2": r2(y_test, y_pred), "smape": smape(y_test, y_pred),
                     "mase": mase(y_test, y_pred, y_train, season=1), "error": None})
    return pd.DataFrame(rows, columns=BASE)


def horizons_per_series() -> pd.DataFrame:
    """`horizons_per_series.csv` старой раскладки: колонок r2 и слагаемых нет вовсе."""
    rows = []
    for horizon, n_folds in ((1, 9), (3, 3)):
        for fold in rolling_origin(24, horizon, n_folds):
            for s in SERIES:
                for model, level in (("prophet", 50.0), ("naive_last", 60.0), ("m", 40.0)):
                    rows.append({"model": model, "horizon": horizon, "fold": fold.index,
                                 "train_end": fold.train_end, "mo": s, "mae": level * horizon + fold.index,
                                 "smape": 1.0, "mase": 1.0, "error": np.nan})
    return pd.DataFrame(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Report:
    def as_text(self) -> str:
        return ""


class BackfillTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.wide = matrix()
        self.predicted = forecasts(self.wide)
        old = per_series(self.wide, self.predicted)
        old.to_csv(self.results / "per_series.csv", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run.summarise(old)
        summary.drop(columns=NEW_IN_SUMMARY).to_csv(self.results / "summary.csv")
        hz = horizons_per_series()
        hz.to_csv(self.results / "horizons_per_series.csv", index=False)
        # Сводки горизонтов прогон считает по файлу, прочитанному обратно, — как здесь.
        by_fold, by_horizon = backfill.horizons.summarise(read_results(self.results / "horizons_per_series.csv"))
        by_fold.drop(columns=NEW_IN_SUMMARY).to_csv(self.results / "horizons_folds.csv", index=False)
        by_horizon.drop(columns=NEW_IN_SUMMARY).to_csv(self.results / "horizons_summary.csv", index=False)
        self.config = {"data": {"path": "panel.parquet", "category": "Все категории", "max_gap": 2},
                       "split": {"horizon": 3, "n_folds": 3},
                       "sample": {"n_series": None, "seed": 1},
                       "output": {"dir": "results"}}

    def run_backfill(self, *flags: str) -> str:
        """`main()` во временном каталоге: загрузка панели подменена синтетической."""
        config, horizons_config = self.root / "config.yaml", self.root / "horizons.yaml"
        config.write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")
        horizons_config.write_text(yaml.safe_dump({"output": {"dir": "results", "prefix": "horizons"}}),
                                   encoding="utf-8")
        self.build_matrix = mock.Mock(return_value=(self.wide, _Report()))
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for name, value in {"ROOT": self.root, "load_panel": mock.Mock(return_value=None),
                                "build_matrix": self.build_matrix}.items():
                stack.enter_context(mock.patch.object(backfill, name, value))
            stack.enter_context(mock.patch.object(sys, "argv", [
                "backfill_r2.py", "--config", str(config), "--horizons-config", str(horizons_config), *flags,
            ]))
            stack.enter_context(contextlib.redirect_stdout(out))
            backfill.main()
        return out.getvalue()

    def digests(self) -> dict[str, str]:
        return {name: digest(self.results / name) for name in FILES}

    def direct(self, model: str, col: str, fold_index: int) -> dict:
        fold = FOLDS[fold_index]
        y = self.wide[col].to_numpy()[fold.test_start : fold.test_end]
        p = self.predicted[(model, col, fold_index)]
        return {"sse": float(np.sum((y - p) ** 2)), "sst": float(np.sum((y - y.mean()) ** 2)),
                "n": 3, "y_sum": float(y.sum()), "y_sq": float(np.sum(y ** 2))}

    def test_restores_pool_parts_from_panel_and_r2(self):
        self.run_backfill()
        after = pd.read_csv(self.results / "per_series.csv")
        self.assertEqual(list(after.columns), BASE[:-1] + PARTS + ["error"])
        flat = 0
        for row in after.itertuples():
            with self.subTest(model=row.model, mo=row.mo, fold=row.fold):
                if isinstance(row.error, str):
                    self.assertTrue(np.isnan([row.sse, row.sst, row.n, row.y_sum, row.y_sq]).all())
                    continue
                expected = self.direct(row.model, row.mo, row.fold)
                for key in ("sst", "n", "y_sum", "y_sq"):
                    self.assertAlmostEqual(getattr(row, key), expected[key], delta=1e-9 * abs(expected[key]))
                if np.isnan(row.r2):
                    flat += 1
                    self.assertTrue(np.isnan(row.sse))
                    self.assertEqual(row.sst, 0.0)
                else:
                    self.assertAlmostEqual(row.sse, expected["sse"], delta=1e-9 * expected["sse"])
        self.assertEqual(flat, 4)  # ровный тест: по строке на каждую из четырёх моделей

    def test_prints_nan_sse_count_and_reconstruction_deviation(self):
        out = self.run_backfill()
        self.assertIn("строк с NaN sse: 4", out)
        self.assertIn("naive_last: сверено 17 строк", out)
        self.assertIn("seasonal_naive: сверено 17 строк", out)
        self.assertIn("максимальное расхождение", out)

    def test_panel_and_folds_come_from_config(self):
        self.config["data"].update({"category": "Продовольствие", "max_gap": 5})
        self.config["split"]["n_folds"] = 2
        with self.assertRaises(ValueError) as caught:
            self.run_backfill()
        self.build_matrix.assert_called_once_with(None, "Продовольствие", max_gap=5)
        # Три фолда в файле против двух в конфиге: фолда 2 в плане нет.
        self.assertIn("фолды", str(caught.exception))

    def test_summary_gets_r2_and_keeps_old_columns(self):
        old = pd.read_csv(self.results / "summary.csv", index_col=0)
        self.run_backfill()
        new = pd.read_csv(self.results / "summary.csv", index_col=0)
        self.assertEqual(list(new.columns), SUMMARY)
        pd.testing.assert_frame_equal(new.loc[old.index, old.columns], old)
        # Пул naive_last напрямую: все его точки, кроме ровного теста (там sse не восстановить).
        y, p = [], []
        for (model, col, fold_index), y_pred in self.predicted.items():
            if model != "naive_last" or (col, fold_index) == ("мо_ровный", 0):
                continue
            fold = FOLDS[fold_index]
            y.append(self.wide[col].to_numpy()[fold.test_start : fold.test_end])
            p.append(y_pred)
        y, p = np.concatenate(y), np.concatenate(p)
        pool = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        self.assertAlmostEqual(new.loc["naive_last", "R² пул"], pool, places=9)
        self.assertTrue(np.isfinite(new["R² медиана"]).all())

    def test_horizons_summaries_are_resummarised_with_nan_r2(self):
        old_summary = pd.read_csv(self.results / "horizons_summary.csv")
        old_folds = pd.read_csv(self.results / "horizons_folds.csv")
        out = self.run_backfill()
        summary = pd.read_csv(self.results / "horizons_summary.csv")
        folds = pd.read_csv(self.results / "horizons_folds.csv")
        self.assertEqual(list(summary.columns), ["horizon", "model", *SUMMARY, "фолдов", "фолдов зачтено"])
        self.assertEqual(list(folds.columns), ["horizon", "model", "fold", "train_end", *SUMMARY])
        for new, old, keys in ((summary, old_summary, ["horizon", "model"]),
                               (folds, old_folds, ["horizon", "model", "fold", "train_end"])):
            new, old = new.set_index(keys), old.set_index(keys)
            pd.testing.assert_frame_equal(new.loc[old.index, old.columns], old)
            self.assertTrue(new[["R² пул", "R² медиана"]].isna().all().all())
            self.assertTrue(new["к Prophet, %"].notna().all())
        self.assertIn("до перепрогона scripts/horizons.py", out)

    def test_backup_holds_originals_and_second_run_is_byte_identical(self):
        originals = self.digests()
        self.run_backfill()
        first = self.digests()
        self.assertNotEqual(first, originals)
        self.run_backfill()
        self.assertEqual(self.digests(), first)
        # Копия сделана до первой записи, и повторный запуск её не перезаписал.
        backup = self.results / "backup_2026-09-25"
        self.assertEqual({name: digest(backup / name) for name in FILES}, originals)

    def test_dry_run_writes_nothing(self):
        before = self.digests()
        out = self.run_backfill("--dry-run")
        self.assertEqual(self.digests(), before)
        self.assertFalse((self.results / "backup_2026-09-25").exists())
        self.assertIn("R² пул", out)
        self.assertIn("naive_last", out)
        self.assertIn("--dry-run", out)

    def test_distorted_r2_fails_the_reconstruction_and_writes_nothing(self):
        path = self.results / "per_series.csv"
        frame = read_results(path)  # только задуманное искажение, без шума разбора
        row = frame.index[(frame["model"] == "naive_last") & (frame["mo"] == "мо_3") & (frame["fold"] == 2)]
        frame.loc[row, "r2"] = frame.loc[row, "r2"] - 1e-3
        frame.to_csv(path, index=False)
        before = self.digests()
        with self.assertRaises(ValueError) as caught:
            self.run_backfill()
        self.assertIn("naive_last", str(caught.exception))
        self.assertIn("мо_3", str(caught.exception))
        self.assertEqual(self.digests(), before)
        self.assertFalse((self.results / "backup_2026-09-25").exists())

    def test_known_forecast_without_a_number_fails_instead_of_passing(self):
        # NaN в панели там, где прогон видел числа, но только во входе известного прогноза:
        # окна теста целы, выведенный sse конечен, а прямой SSE — NaN. Такая строка
        # не сверена, и скрипт должен упасть, а не счесть её совпавшей.
        self.wide.iloc[14, 0] = np.nan  # мо_0: вход naive_last на фолде 0 (конец обучения)
        self.wide.iloc[3, 1] = np.nan  # мо_1: вход seasonal_naive на фолде 0 (год назад)
        before = self.digests()
        with self.assertRaises(ValueError) as caught:
            self.run_backfill()
        message = str(caught.exception)
        self.assertIn("naive_last: 1", message)
        self.assertIn("seasonal_naive: 1", message)
        self.assertEqual(self.digests(), before)
        self.assertFalse((self.results / "backup_2026-09-25").exists())

    def test_changed_old_summary_stops_before_writing(self):
        path = self.results / "summary.csv"
        summary = read_results(path, index_col=0)
        summary.loc["drift", "MAE"] += 1.0
        summary.to_csv(path)
        before = self.digests()
        with self.assertRaises(ValueError) as caught:
            self.run_backfill()
        self.assertIn("MAE", str(caught.exception))
        self.assertEqual(self.digests(), before)

    def test_series_outside_the_panel_is_rejected(self):
        self.wide = self.wide.drop(columns=["мо_4"])
        before = self.digests()
        with self.assertRaises(ValueError) as caught:
            self.run_backfill()
        self.assertIn("мо_4", str(caught.exception))
        self.assertEqual(self.digests(), before)


class HorizonsImportTest(unittest.TestCase):
    """`scripts/horizons.py` грузится по пути, а не через пакет `scripts`."""

    def test_foreign_scripts_package_does_not_shadow_horizons(self):
        # Обычный пакет `scripts` где угодно на sys.path перекрывает неявный пакет
        # пространства имён: `from scripts import horizons` искал бы модуль в чужом пакете.
        code = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('backfill_r2', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "print(module.horizons.__file__)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "scripts").mkdir()
            (Path(tmp) / "scripts" / "__init__.py").write_text("", encoding="utf-8")
            done = subprocess.run(
                [sys.executable, "-c", code, str(ROOT / "scripts" / "backfill_r2.py")],
                capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": tmp},
            )
        self.assertEqual(done.returncode, 0, done.stderr[-1500:])
        loaded = Path(done.stdout.strip()).resolve()
        self.assertEqual(loaded, (ROOT / "scripts" / "horizons.py").resolve())


class InvariantsTest(unittest.TestCase):
    """Сверка до записи: бэкфилл только дописывает колонки."""

    def setUp(self):
        wide = matrix()
        self.before = per_series(wide, forecasts(wide))
        self.after = backfill.fill_parts(self.before, wide, FOLDS)

    def test_filled_frame_passes(self):
        backfill.check_invariants(self.before, self.after)

    def test_changed_metric_is_caught(self):
        for column in ("mae", "mase", "smape", "r2"):
            with self.subTest(column):
                broken = self.after.copy()
                broken.loc[5, column] += 1.0
                with self.assertRaises(ValueError) as caught:
                    backfill.check_invariants(self.before, broken)
                self.assertIn(column, str(caught.exception))

    def test_reordered_rows_are_caught(self):
        with self.assertRaises(ValueError):
            backfill.check_invariants(self.before, self.after.iloc[::-1].reset_index(drop=True))

    def test_dropped_or_extra_column_is_caught(self):
        with self.assertRaises(ValueError):
            backfill.check_invariants(self.before, self.after.drop(columns=["smape"]))
        with self.assertRaises(ValueError):
            backfill.check_invariants(self.before, self.after.assign(extra=1.0))

    def test_dropped_row_is_caught(self):
        with self.assertRaises(ValueError):
            backfill.check_invariants(self.before, self.after.iloc[1:])


class KnownForecastTest(unittest.TestCase):
    """Прогноз проверки повторяет `src/models/naive.py`: сверено с классами, а не предположено."""

    def test_formula_matches_naive_models(self):
        series = matrix()["мо_2"].to_numpy()
        for train_end, horizon in ((15, 3), (21, 3), (12, 12), (15, 1), (18, 6), (12, 13), (8, 3)):
            fold = Fold(index=0, train_end=train_end, test_start=train_end, test_end=train_end + horizon)
            y_train = series[:train_end]
            with self.subTest(train_end=train_end, horizon=horizon):
                np.testing.assert_array_equal(
                    backfill.known_forecast("naive_last", series, fold, 12),
                    NaiveLast().fit(y_train).predict(horizon))
                np.testing.assert_array_equal(
                    backfill.known_forecast("seasonal_naive", series, fold, 12),
                    SeasonalNaive(season=12).fit(y_train).predict(horizon))

    def test_season_is_that_of_the_registered_model(self):
        self.assertEqual(backfill.SEASON, run.REGISTRY["seasonal_naive"]().season)


if __name__ == "__main__":
    unittest.main()
