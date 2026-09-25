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
# Метрики строки в порядке файла: четыре прежние и слагаемые R² пула.
METRICS = ["mae", "r2", "smape", "mase", "sse", "sst", "n", "y_sum", "y_sq"]
SUMMARY = ["MAE", "MASE", "sMAPE", "R² пул", "R² медиана", "к Prophet, %", "к наивной, %",
           "серий", "отказов"]


def frame(models: list[str], series: list[str], mae: float = 1.0) -> pd.DataFrame:
    """Строки ряд × фолд в полной раскладке `per_series.csv` — на ней работает `main()`."""
    return pd.DataFrame([
        {"model": m, "fold": f, "mo": s, "mae": mae, "r2": 0.0, "smape": 1.0, "mase": 1.0,
         "error": None}
        for m in models for s in series for f in (0, 1, 2)
    ])


def matrix(series: list[str]) -> pd.DataFrame:
    """24 месяца × ряды с трендом и колебанием: у наивной модели конечные метрики."""
    t = np.arange(24, dtype=float)
    return pd.DataFrame({s: 100 + (i + 1) * t + 5 * np.sin(t) for i, s in enumerate(series)})


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


class _NoHistory:
    """Панельная модель, которой не хватило истории на признаки: прогноз — NaN."""

    def fit(self, full, train_end, horizon):
        return self

    def predict(self, subset, train_end, horizon):
        return np.full((subset.shape[1], horizon), np.nan)


class FailureTextTest(unittest.TestCase):
    """Пустой текст отказа уходит в CSV пустым полем и читается обратно как успех."""

    def test_series_model_failure_names_exception_type(self):
        folds = rolling_origin(24, 3, 1)
        task = ("silent", "мо_0", np.arange(24, dtype=float), pd.RangeIndex(24), folds, 3)
        with mock.patch.object(run, "_build_model", return_value=_Silent()):
            rows = run._score_series(task)
        self.assertEqual([row["error"] for row in rows], ["AssertionError: "])

    def test_failure_rows_carry_empty_metrics(self):
        # Без колонок метрик партия из одних отказов остаётся без `mae`, и сводка
        # падает с KeyError ещё до слияния.
        with self.subTest("модель по рядам"):
            folds = rolling_origin(24, 3, 1)
            task = ("silent", "мо_0", np.arange(24, dtype=float), pd.RangeIndex(24), folds, 3)
            with mock.patch.object(run, "_build_model", return_value=_Silent()):
                row = run._score_series(task)[0]
            self.assertTrue(all(np.isnan(row[metric]) for metric in METRICS))
        with self.subTest("панельная модель без истории для признаков"):
            wide = matrix(SERIES[:2])
            with mock.patch.dict(run.GLOBAL_MODELS, {"no_history": _NoHistory}):
                part = run.evaluate_global(wide, wide, "no_history", 3, 1)
            self.assertEqual(set(part["error"]), {"недостаточно истории для признаков"})
            self.assertTrue(part[METRICS].isna().all().all())


class _LastValue:
    """Панельная модель с известным прогнозом: последнее значение обучения каждого ряда."""

    def fit(self, full, train_end, horizon):
        return self

    def predict(self, subset, train_end, horizon):
        last = subset.to_numpy(dtype=float)[train_end - 1]
        return np.repeat(last[:, None], horizon, axis=1)


def parts(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Слагаемые пула, посчитанные в тесте напрямую."""
    return {
        "sse": float(np.sum((y_test - y_pred) ** 2)),
        "sst": float(np.sum((y_test - y_test.mean()) ** 2)),
        "n": len(y_test), "y_sum": float(np.sum(y_test)), "y_sq": float(np.sum(y_test ** 2)),
    }


class RowPartsTest(unittest.TestCase):
    """Строки успеха несут слагаемые R² пула, у моделей по рядам и панельных одинаково."""

    def assert_row(self, row, y_test: np.ndarray, y_pred: np.ndarray) -> None:
        self.assertEqual(list(row.keys()), ["model", "fold", "mo", *METRICS, "error"])
        for key, value in parts(y_test, y_pred).items():
            self.assertAlmostEqual(row[key], value, places=6, msg=key)
        self.assertAlmostEqual(row["r2"], 1 - row["sse"] / row["sst"], places=12)

    def test_series_model_rows(self):
        series = matrix(["мо_0"])["мо_0"].to_numpy()
        folds = rolling_origin(24, 3, 3)
        rows = run._score_series(("naive_last", "мо_0", series, pd.RangeIndex(24), folds, 3))
        self.assertEqual(len(rows), 3)
        for row, fold in zip(rows, folds):
            self.assert_row(row, series[fold.test_start : fold.test_end],
                            np.full(3, series[fold.train_end - 1]))

    def test_panel_model_rows(self):
        wide = matrix(SERIES[:2])
        with mock.patch.dict(run.GLOBAL_MODELS, {"last": _LastValue}):
            part = run.evaluate_global(wide, wide, "last", 3, 3)
        self.assertEqual(len(part), 2 * 3)
        for row in part.to_dict("records"):
            fold = rolling_origin(24, 3, 3)[row["fold"]]
            series = wide[row["mo"]].to_numpy()
            self.assert_row(row, series[fold.test_start : fold.test_end],
                            np.full(3, series[fold.train_end - 1]))


def scored(model: str, mo: str, fold: int, y: np.ndarray, p: np.ndarray) -> dict:
    """Строка успеха с метриками, посчитанными в тесте напрямую."""
    row = parts(y, p)
    return {"model": model, "fold": fold, "mo": mo, "mae": float(np.mean(np.abs(y - p))),
            "r2": 1 - row["sse"] / row["sst"], "smape": 1.0, "mase": 1.0, **row, "error": None}


class SummariseTest(unittest.TestCase):
    def test_columns_in_order(self):
        # Разные MAE: одинаковые модели сводка отметила бы предупреждением в выводе.
        per_series = pd.concat([frame(["naive_last"], SERIES), frame(["drift"], SERIES, mae=2.0)],
                               ignore_index=True)
        summary = run.summarise(per_series)
        self.assertEqual(list(summary.columns), SUMMARY)
        self.assertEqual(summary.index.name, "model")

    def test_pool_median_and_gains(self):
        rng = np.random.default_rng(3)
        noise = {"prophet": 0.05, "naive_last": 0.08, "best": 0.02}
        actual, forecast, rows = {m: [] for m in noise}, {m: [] for m in noise}, []
        for s, level in enumerate([100.0, 1_000.0, 10_000.0]):
            for fold in (0, 1, 2):
                y = level * (1 + 0.1 * rng.standard_normal(3))
                for model, scale in noise.items():
                    p = y + level * scale * rng.standard_normal(3)
                    actual[model].append(y)
                    forecast[model].append(p)
                    rows.append(scored(model, f"мо_{s}", fold, y, p))
        # Отказ не входит ни в пул, ни в медиану, но считается в отказах.
        rows.append({"model": "best", "fold": 0, "mo": "мо_9", **dict.fromkeys(METRICS, np.nan),
                     "error": "ValueError: мало точек"})
        per_series = pd.DataFrame(rows)
        summary = run.summarise(per_series)

        ok = per_series[per_series["error"].isna()]
        for model in noise:
            with self.subTest(model):
                y, p = np.concatenate(actual[model]), np.concatenate(forecast[model])
                pool = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
                self.assertAlmostEqual(summary.loc[model, "R² пул"], pool, places=12)
                self.assertEqual(summary.loc[model, "R² медиана"],
                                 ok.loc[ok["model"] == model, "r2"].median())
        mae = summary["MAE"]
        self.assertAlmostEqual(summary.loc["best", "к Prophet, %"],
                               100 * (mae["prophet"] - mae["best"]) / mae["prophet"], places=12)
        self.assertAlmostEqual(summary.loc["best", "к наивной, %"],
                               100 * (mae["naive_last"] - mae["best"]) / mae["naive_last"], places=12)
        self.assertEqual(summary.loc["prophet", "к Prophet, %"], 0.0)
        self.assertEqual(summary.loc["best", "отказов"], 1)

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


def run_main(root: Path, models: list[str], wide: pd.DataFrame, **patches) -> str:
    """`run.main()` во временном каталоге `root`: загрузка панели подменена, модели —
    настоящие, если их не подменили в `patches`. Возвращает напечатанное."""
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data": {"path": "panel.parquet", "category": "Все категории", "max_gap": 2},
        "split": {"horizon": 3, "n_folds": 3},
        "sample": {"n_series": None, "seed": 1},
        "models": models,
        "output": {"dir": "results"},
        "compute": {"workers": 1},
    }, allow_unicode=True), encoding="utf-8")
    replaced = {
        "ROOT": root,
        "load_panel": mock.Mock(return_value=None),
        "build_matrix": mock.Mock(return_value=(wide, _Report())),
        "build_context": mock.Mock(return_value=None),
        **patches,
    }
    out = io.StringIO()
    with contextlib.ExitStack() as stack:
        for name, value in replaced.items():
            stack.enter_context(mock.patch.object(run, name, value))
        stack.enter_context(mock.patch.object(sys, "argv", ["run.py", "--config", str(config)]))
        stack.enter_context(contextlib.redirect_stdout(out))
        run.main()
    return out.getvalue()


class MainTest(unittest.TestCase):
    """`main()` целиком на синтетике: от плана до записанных файлов."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.per_path = self.root / "results" / "per_series.csv"
        self.per_path.parent.mkdir()

    def test_plan_on_other_series_stops_before_any_model(self):
        frame(["naive_last", "drift"], SERIES).to_csv(self.per_path, index=False)
        before = digest(self.per_path)
        evaluate, panel = mock.Mock(side_effect=_model_started), mock.Mock(side_effect=_model_started)
        # Матрица на 3 рядах из 5, что лежат в файле: пилот против полной панели.
        with self.assertRaises(ValueError) as caught:
            run_main(self.root, ["naive_last"], matrix(SERIES[:3]),
                     evaluate=evaluate, evaluate_global=panel)
        self.assertIn("модели не запускались", str(caught.exception))
        evaluate.assert_not_called()
        panel.assert_not_called()
        self.assertEqual(digest(self.per_path), before)

    def test_batch_replaces_its_models_keeps_the_rest_and_writes_summary(self):
        frame(["naive_last"], SERIES[:3], mae=999.0).to_csv(self.per_path, index=False)
        frame(["drift"], SERIES[:3], mae=5.0).to_csv(self.per_path, index=False, mode="a", header=False)
        real_build = run._build_model
        # theta отказывает на каждом ряду и фолде — через настоящий _score_series.
        build = mock.Mock(side_effect=lambda name: _Silent() if name == "theta" else real_build(name))
        run_main(self.root, ["naive_last", "theta"], matrix(SERIES[:3]), _build_model=build)
        build.assert_called()

        on_disk = pd.read_csv(self.per_path)
        naive = on_disk[on_disk["model"] == "naive_last"]
        self.assertEqual(len(naive), 9)
        self.assertTrue(naive["error"].isna().all())
        self.assertFalse((naive["mae"] == 999.0).any())
        self.assertEqual(on_disk.loc[on_disk["model"] == "drift", "mae"].tolist(), [5.0] * 9)
        theta = on_disk[on_disk["model"] == "theta"]
        self.assertEqual(len(theta), 9)
        self.assertTrue(theta["mae"].isna().all())
        self.assertTrue(theta["error"].str.startswith("AssertionError").all())

        self.assertTrue(naive[METRICS].notna().all().all())
        self.assertTrue(theta[METRICS].isna().all().all())

        summary = pd.read_csv(self.root / "results" / "summary.csv", index_col=0)
        self.assertEqual(list(summary.columns), SUMMARY)
        self.assertEqual(summary.loc["theta", "отказов"], 9)
        self.assertEqual(summary.loc["theta", "серий"], 0)
        self.assertEqual(summary["серий"].dtype.kind, "i")  # целые, а не дробные из-за NaN
        self.assertTrue(np.isnan(summary.loc["theta", "MAE"]))
        self.assertEqual(summary.loc["drift", "MAE"], 5.0)
        # Строки drift — из файла старой раскладки, слагаемых пула у них нет.
        self.assertTrue(np.isfinite(summary.loc["naive_last", "R² пул"]))
        self.assertTrue(np.isnan(summary.loc["drift", "R² пул"]))

    def test_identical_models_warning_is_printed_once(self):
        # Раньше сводка считалась и по одной партии до слияния (результат не читался),
        # и после: предупреждение о совпадающих прогнозах печаталось дважды.
        frame(["drift"], SERIES[:3], mae=5.0).to_csv(self.per_path, index=False)
        real_build = run._build_model
        twin = mock.Mock(side_effect=lambda name: real_build("naive_last" if name == "theta" else name))
        out = run_main(self.root, ["naive_last", "theta"], matrix(SERIES[:3]), _build_model=twin)
        self.assertEqual(out.count("naive_last и theta дали идентичные прогнозы"), 1)

    def test_batch_of_only_refusals_is_written_and_summarised(self):
        # Первый прогон в пустой каталог, и модель отказала везде: раньше KeyError: 'mae'.
        build = mock.Mock(return_value=_Silent())
        run_main(self.root, ["theta"], matrix(SERIES[:3]), _build_model=build)
        build.assert_called()
        on_disk = pd.read_csv(self.per_path)
        self.assertEqual(len(on_disk), 9)
        self.assertTrue(on_disk["mae"].isna().all())
        summary = pd.read_csv(self.root / "results" / "summary.csv", index_col=0)
        self.assertEqual(summary.loc["theta", "отказов"], 9)
        self.assertEqual(summary.loc["theta", "серий"], 0)
        self.assertEqual(summary["серий"].dtype.kind, "i")


if __name__ == "__main__":
    unittest.main()
