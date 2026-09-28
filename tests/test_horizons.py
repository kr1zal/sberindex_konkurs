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

from src.results_guard import read_results  # noqa: E402
from src.split import rolling_origin  # noqa: E402

# scripts/ — не пакет: скрипт грузится по пути, без правки sys.path под все тесты.
_spec = importlib.util.spec_from_file_location("horizons", ROOT / "scripts" / "horizons.py")
horizons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(horizons)

SERIES = ["мо_0", "мо_1", "мо_2"]
PAIR = {"model": "m", "horizon": 3}
STAMP = "20260925T200000"
# Метрики строки в порядке файла: четыре прежние и слагаемые R² пула.
METRICS = ["mae", "r2", "smape", "mase", "sse", "sst", "n", "y_sum", "y_sq"]
SUMMARY = ["MAE", "MASE", "sMAPE", "R² пул", "R² медиана", "к Prophet, %", "к наивной, %",
           "серий", "отказов"]


def _train_end(horizon: int, folds: list[int], fold: int) -> int:
    """Длина обучения фолда — как у `rolling_origin` на 24 точках."""
    return 24 - (len(folds) - fold) * horizon


def per_series(model: str, horizon: int, folds: list[int], series=SERIES, mae=1.0) -> pd.DataFrame:
    """Строки ряд × фолд в раскладке `horizons_per_series.csv`."""
    return pd.DataFrame([
        {"model": model, "horizon": horizon, "fold": f, "train_end": _train_end(horizon, folds, f), "mo": s,
         "mae": mae, "smape": 1.0, "mase": 1.0, "error": np.nan}
        for s in series for f in folds
    ])


def awkward(n: int, seed: int) -> np.ndarray:
    """Числа, неудобные для разбора CSV: два из сводки и случайные. Разбор pandas
    по умолчанию читает примерно каждое восьмое из них не тем, что записано."""
    rng = np.random.default_rng(seed)
    return np.concatenate([[1217.032436222938, 0.6207988257547953], rng.uniform(0, 5000, n - 2)])


def steps(model: str, horizon: int, folds: list[int], mae=1.0) -> pd.DataFrame:
    """Строки фолд × шаг в раскладке `horizons_steps.csv`: колонки `mo` в нём нет."""
    return pd.DataFrame([
        {"model": model, "horizon": horizon, "fold": f, "train_end": _train_end(horizon, folds, f),
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

    def test_other_pairs_keep_their_numbers_to_the_last_digit(self):
        # Слияние читает файл и пишет его заново: строки чужих пар должны вернуться
        # теми же числами, иначе каждая партия сдвигала бы их в последнем разряде.
        path = self.dir / "horizons_per_series.csv"
        many = [f"мо_{i}" for i in range(10)]
        other = per_series("other", 3, [0, 1, 2], series=many)
        for i, column in enumerate(["mae", "smape", "mase"]):
            other[column] = awkward(len(other), seed=10 + i)
        pd.concat([per_series("m", 3, [0, 1, 2], series=many), other], ignore_index=True).to_csv(path, index=False)
        horizons.merge_into(path, per_series("m", 3, [0, 1, 2], series=many, mae=2.0), PAIR, same_panel=True)
        back = pd.read_csv(path, float_precision="round_trip")
        kept = back[back["model"] == "other"]
        for column in ["mae", "smape", "mase"]:
            with self.subTest(column):
                self.assertTrue(np.array_equal(kept[column].to_numpy(), other[column].to_numpy()))

    def test_batch_on_other_series_is_rejected_and_file_untouched(self):
        path = self.dir / "horizons_per_series.csv"
        per_series("m", 3, [0, 1, 2]).to_csv(path, index=False)
        before = path.read_bytes()
        fresh = per_series("m", 3, [0, 1, 2], series=SERIES[:2], mae=2.0)
        with self.assertRaises(ValueError):
            horizons.merge_into(path, fresh, PAIR, same_panel=True, stamp=STAMP)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.dir / "realizations").exists())  # сверка идёт раньше снимка

    def test_replaced_pair_is_snapshotted_exactly(self):
        path = self.dir / "horizons_per_series.csv"
        old = per_series("m", 3, [0, 1, 2])
        old["mae"] = awkward(len(old), seed=30)
        pd.concat([old, per_series("m", 1, [0, 1]), per_series("other", 3, [0, 1, 2])],
                  ignore_index=True).to_csv(path, index=False)
        before = read_results(path)
        with contextlib.redirect_stdout(io.StringIO()):
            horizons.merge_into(path, per_series("m", 3, [0, 1, 2], mae=2.0), PAIR,
                                same_panel=True, stamp=STAMP)
        snapshots = self.dir / "realizations"
        self.assertEqual(sorted(p.name for p in snapshots.iterdir()), [f"m__h3__{STAMP}.csv"])
        # Ровно старые строки пары, все колонки файла, числа до последнего разряда.
        expected = before.loc[(before["model"] == "m") & (before["horizon"] == 3)].reset_index(drop=True)
        pd.testing.assert_frame_equal(read_results(snapshots / f"m__h3__{STAMP}.csv"), expected)

    def test_new_pair_leaves_no_snapshot(self):
        path = self.dir / "horizons_per_series.csv"
        per_series("other", 3, [0, 1, 2]).to_csv(path, index=False)
        horizons.merge_into(path, per_series("m", 3, [0, 1, 2]), PAIR, same_panel=True, stamp=STAMP)
        self.assertFalse((self.dir / "realizations").exists())


class MergeIntoFromSnapshotTest(unittest.TestCase):
    """I1: на чистом клоне обычного `horizons_per_series.csv` нет, есть только его
    `.csv.gz` (`scripts/export_results.py`). `merge_into` должен слить партию со
    снимком, а не лечь в файл в одиночестве — иначе пересчёт сводок горизонтов молча
    теряет все пары модель × горизонт, которых не было в партии."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "horizons_per_series.csv"
        self.gz = self.dir / "horizons_per_series.csv.gz"

    def test_batch_of_one_pair_keeps_the_rest_of_the_snapshot(self):
        # Ровно инцидент I1: партия из одной пары на клоне с одним .csv.gz.
        pd.concat([per_series("m", 3, [0, 1, 2]), per_series("other", 3, [0, 1, 2], mae=5.0)],
                  ignore_index=True).to_csv(self.gz, index=False, compression="gzip")
        horizons.merge_into(self.path, per_series("m", 3, [0, 1, 2], mae=2.0), PAIR, same_panel=True)
        self.assertTrue(self.path.exists())  # обычный CSV появился и сразу полный
        after = pd.read_csv(self.path)
        self.assertEqual(pairs(after), [("m", 3), ("other", 3)])
        self.assertEqual(after.loc[after["model"] == "m", "mae"].unique().tolist(), [2.0])
        self.assertEqual(after.loc[after["model"] == "other", "mae"].unique().tolist(), [5.0])

    def test_new_pair_is_added_next_to_the_whole_snapshot(self):
        per_series("other", 3, [0, 1, 2]).to_csv(self.gz, index=False, compression="gzip")
        horizons.merge_into(self.path, per_series("m", 3, [0, 1, 2]), PAIR, same_panel=True)
        after = pd.read_csv(self.path)
        self.assertEqual(pairs(after), [("m", 3), ("other", 3)])

    def test_batch_on_other_series_is_rejected_against_the_snapshot(self):
        # Сверка (check_same_panel) срабатывает против снимка, не только против
        # обычного файла: партия на подмножестве рядов отклоняется так же.
        per_series("m", 3, [0, 1, 2]).to_csv(self.gz, index=False, compression="gzip")
        fresh = per_series("m", 3, [0, 1, 2], series=SERIES[:2], mae=2.0)
        with self.assertRaises(ValueError):
            horizons.merge_into(self.path, fresh, PAIR, same_panel=True, stamp=STAMP)
        self.assertFalse(self.path.exists())  # партия отклонена — обычный файл не создан
        self.assertFalse((self.dir / "realizations").exists())


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


def naive_rows(wide: pd.DataFrame, horizon: int, n_folds: int) -> tuple[pd.DataFrame, dict]:
    """Строки naive_last по всем рядам через настоящий `_score_series` и его известный
    прогноз: {фолд: (факт, прогноз)} — точки всех рядов фолда подряд."""
    folds = rolling_origin(wide.shape[0], horizon, n_folds)
    rows, points = [], {f.index: ([], []) for f in folds}
    for col in wide.columns:
        series = wide[col].to_numpy(dtype=float)
        part, _ = horizons._score_series(("naive_last", col, series, wide.index, folds, horizon))
        rows.extend(part)
        for fold in folds:
            points[fold.index][0].append(series[fold.test_start : fold.test_end])
            points[fold.index][1].append(np.full(horizon, series[fold.train_end - 1]))
    return pd.DataFrame(rows), points


def pooled(points: list[tuple[list, list]]) -> float:
    y = np.concatenate([a for actual, _ in points for a in actual])
    p = np.concatenate([b for _, forecast in points for b in forecast])
    return 1 - float(np.sum((y - p) ** 2)) / float(np.sum((y - y.mean()) ** 2))


class RowPartsTest(unittest.TestCase):
    def test_series_model_rows_carry_r2_and_pool_parts(self):
        series = matrix(["мо_0"])["мо_0"].to_numpy()
        folds = rolling_origin(24, 3, 3)
        rows, _ = horizons._score_series(("naive_last", "мо_0", series, pd.RangeIndex(24), folds, 3))
        for row, fold in zip(rows, folds):
            self.assertEqual(list(row), ["model", "horizon", "fold", "train_end", "mo", *METRICS, "error"])
            y = series[fold.test_start : fold.test_end]
            p = np.full(3, series[fold.train_end - 1])
            self.assertAlmostEqual(row["sse"], float(np.sum((y - p) ** 2)), places=6)
            self.assertAlmostEqual(row["sst"], float(np.sum((y - y.mean()) ** 2)), places=6)
            self.assertEqual(row["n"], 3)
            self.assertAlmostEqual(row["y_sum"], float(y.sum()), places=6)
            self.assertAlmostEqual(row["y_sq"], float(np.sum(y ** 2)), places=3)
            self.assertAlmostEqual(row["r2"], 1 - row["sse"] / row["sst"], places=12)

    def test_failure_rows_carry_empty_metrics(self):
        folds = rolling_origin(24, 3, 1)
        with self.subTest("модель по рядам"):
            task = ("silent", "мо_0", np.arange(24, dtype=float), pd.RangeIndex(24), folds, 3)
            with mock.patch.object(horizons, "_build_model", return_value=_Silent()):
                rows, _ = horizons._score_series(task)
            self.assertTrue(all(np.isnan(rows[0][metric]) for metric in METRICS))
        with self.subTest("панельная модель"):
            wide = pd.DataFrame(np.ones((24, 2)), columns=["мо_0", "мо_1"])
            with mock.patch.dict(horizons.EXTRA_GLOBAL, {"silent": _Silent}), \
                    contextlib.redirect_stdout(io.StringIO()):
                part, _ = horizons.evaluate_panel_model(wide, wide, "silent", 3, folds, context=None)
            self.assertTrue(part[METRICS].isna().all().all())


class SummariseTest(unittest.TestCase):
    def test_columns_in_order(self):
        by_fold, by_horizon = horizons.summarise(per_series("m", 3, [0, 1, 2]))
        self.assertEqual(list(by_fold.columns), ["horizon", "model", "fold", "train_end", *SUMMARY])
        self.assertEqual(list(by_horizon.columns), ["horizon", "model", *SUMMARY, "фолдов", "фолдов зачтено"])

    def test_counted_folds_leave_out_a_fold_of_only_refusals(self):
        # chronos_ft на h=6: первый фолд с обучением 12 — отказ на всех рядах. «фолдов»
        # остаётся плановым числом, «фолдов зачтено» — из скольких фолдов его MAE.
        frame = pd.concat([per_series("naive_last", 6, [0, 1]), per_series("chronos_ft", 6, [0, 1]),
                           per_series("global_gbm", 6, [0, 1])], ignore_index=True)
        frame["error"] = frame["error"].astype(object)
        refusals = ((frame["model"] == "chronos_ft") & (frame["fold"] == 0)) | (frame["model"] == "global_gbm")
        frame.loc[refusals, ["mae", "smape", "mase"]] = np.nan
        frame.loc[refusals, "error"] = "ValueError: нет обучающих окон"
        with contextlib.redirect_stdout(io.StringIO()):  # пары у моделей разные — будет ВНИМАНИЕ
            _, by_horizon = horizons.summarise(frame)
        table = by_horizon.set_index("model")
        self.assertEqual(table.loc["chronos_ft", "фолдов"], 2)
        self.assertEqual(table.loc["chronos_ft", "фолдов зачтено"], 1)
        self.assertEqual(table.loc["naive_last", "фолдов зачтено"], 2)
        self.assertEqual(table.loc["global_gbm", "фолдов"], 2)
        self.assertEqual(table.loc["global_gbm", "фолдов зачтено"], 0)
        self.assertEqual(by_horizon["фолдов зачтено"].dtype.kind, "i")

    def test_models_on_different_pairs_within_a_horizon_are_flagged(self):
        frame = pd.concat([per_series("naive_last", 3, [0, 1, 2]), per_series("m", 3, [0, 1, 2]),
                           per_series("naive_last", 1, [0, 1]), per_series("m", 1, [0, 1])],
                          ignore_index=True)
        lost = (frame["model"] == "m") & (frame["horizon"] == 3) & (frame["mo"] == "мо_0") & (frame["fold"] == 1)
        frame.loc[lost, "mae"] = np.nan
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            horizons.summarise(frame)
        self.assertTrue(out.getvalue().startswith("ВНИМАНИЕ: горизонт 3:"))
        self.assertIn("8 — m", out.getvalue())
        self.assertNotIn("горизонт 1", out.getvalue())

    def test_r2_pool_per_horizon_and_per_fold(self):
        wide = matrix(SERIES)
        h1, points1 = naive_rows(wide, 1, 9)
        h3, points3 = naive_rows(wide, 3, 3)
        by_fold, by_horizon = horizons.summarise(pd.concat([h1, h3], ignore_index=True))
        summary = by_horizon.set_index(["horizon", "model"])
        folds = by_fold.set_index(["horizon", "model", "fold"])
        for horizon, points in ((1, points1), (3, points3)):
            with self.subTest(horizon=horizon):
                self.assertAlmostEqual(summary.loc[(horizon, "naive_last"), "R² пул"],
                                       pooled(list(points.values())), places=12)
                for fold, fold_points in points.items():
                    self.assertAlmostEqual(folds.loc[(horizon, "naive_last", fold), "R² пул"],
                                           pooled([fold_points]), places=12)
        self.assertEqual(summary.loc[(3, "naive_last"), "R² медиана"], h3["r2"].median())
        # На горизонте 1 в тесте пары ряд × фолд одна точка: её SST ноль, R² строки не определён,
        # и медиана — NaN по построению. Пул определён: его SST — вокруг среднего всех точек.
        self.assertTrue(h1["r2"].isna().all())
        self.assertTrue(np.isnan(summary.loc[(1, "naive_last"), "R² медиана"]))

    def test_old_file_without_r2_gives_nan_r2_and_gains_within_horizon(self):
        # Файл горизонтов, посчитанный до слагаемых пула: колонок r2 и sse в нём нет.
        frame = pd.concat([
            per_series("prophet", 1, [0, 1], mae=50.0), per_series("m", 1, [0, 1], mae=40.0),
            per_series("naive_last", 3, [0, 1, 2], mae=100.0), per_series("m", 3, [0, 1, 2], mae=90.0),
        ], ignore_index=True)
        by_fold, by_horizon = horizons.summarise(frame)
        for table in (by_fold, by_horizon):
            self.assertTrue(table[["R² пул", "R² медиана"]].isna().all().all())
        summary = by_horizon.set_index(["horizon", "model"])
        self.assertAlmostEqual(summary.loc[(1, "m"), "к Prophet, %"], 20.0)
        self.assertTrue(np.isnan(summary.loc[(3, "m"), "к Prophet, %"]))
        self.assertAlmostEqual(summary.loc[(3, "m"), "к наивной, %"], 10.0)
        self.assertTrue(np.isnan(summary.loc[(1, "m"), "к наивной, %"]))

    def test_fold_summary_takes_reference_of_the_same_fold(self):
        frame = pd.concat([per_series("prophet", 3, [0, 1, 2]), per_series("m", 3, [0, 1, 2])],
                          ignore_index=True)
        frame["mae"] = np.where(frame["model"] == "prophet", 50.0 * (frame["fold"] + 1), 40.0)
        by_fold, _ = horizons.summarise(frame)
        gains = by_fold[by_fold["model"] == "m"].set_index("fold")["к Prophet, %"]
        self.assertAlmostEqual(gains[0], 20.0)
        self.assertAlmostEqual(gains[1], 60.0)
        self.assertAlmostEqual(gains[2], 100 * (150.0 - 40.0) / 150.0)

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


# Настоящий прогон моделей по рядам: подмены ниже его оборачивают, а не заменяют.
_evaluate_series_models = horizons.evaluate_series_models


def matrix(series: list[str]) -> pd.DataFrame:
    """24 месяца × ряды с трендом и колебанием: у наивной модели конечные метрики."""
    t = np.arange(24, dtype=float)
    return pd.DataFrame({s: 100 + (i + 1) * t + 5 * np.sin(t) for i, s in enumerate(series)})


def run_main(root: Path, plan: dict, wide: pd.DataFrame, **patches) -> str:
    """`horizons.main()` во временном каталоге `root`: загрузка панели подменена, модели —
    настоящие, если их не подменили в `patches`. `plan` — горизонты и модели конфига.
    Возвращает напечатанное."""
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data": {"path": "panel.parquet", "category": "Все категории", "max_gap": 2},
        "sample": {"n_series": None, "seed": 1},
        "output": {"dir": "results", "prefix": "horizons"},
        "compute": {"workers": 1},
        **plan,
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
        # main() глушит предупреждения и меняет формат печати pandas глобально.
        stack.enter_context(warnings.catch_warnings())
        stack.enter_context(pd.option_context("display.float_format", None))
        for name, value in replaced.items():
            stack.enter_context(mock.patch.object(horizons, name, value))
        stack.enter_context(mock.patch.object(sys, "argv", ["horizons.py", "--config", str(config)]))
        stack.enter_context(contextlib.redirect_stdout(out))
        horizons.main()
    return out.getvalue()


H3 = {"horizons": [{"horizon": 3, "n_folds": 3}], "models": ["naive_last"]}


class MainTest(unittest.TestCase):
    """`main()` целиком на синтетике: от плана до записанных файлов."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.per_path = self.root / "results" / "horizons_per_series.csv"
        self.steps_path = self.root / "results" / "horizons_steps.csv"
        self.per_path.parent.mkdir()

    def write_files(self, series: list[str]) -> None:
        """В файлах — naive_last и drift на горизонте 3, как после прошлого прогона."""
        pd.concat([per_series("naive_last", 3, [0, 1, 2], series=series, mae=999.0),
                   per_series("drift", 3, [0, 1, 2], series=series, mae=5.0)],
                  ignore_index=True).to_csv(self.per_path, index=False)
        pd.concat([steps("naive_last", 3, [0, 1, 2], mae=999.0), steps("drift", 3, [0, 1, 2], mae=5.0)],
                  ignore_index=True).to_csv(self.steps_path, index=False)

    def test_replaced_pairs_are_snapshotted_under_one_stamp_and_steps_are_not(self):
        # В файле naive_last на горизонтах 3 и 1: обе пары заменяются одним вызовом main,
        # и оба снимка — под одним штампом, хотя слияний два.
        pd.concat([per_series("naive_last", 3, [0, 1, 2], mae=999.0),
                   per_series("naive_last", 1, list(range(9)), mae=999.0)],
                  ignore_index=True).to_csv(self.per_path, index=False)
        pd.concat([steps("naive_last", 3, [0, 1, 2], mae=999.0), steps("naive_last", 1, list(range(9)), mae=999.0)],
                  ignore_index=True).to_csv(self.steps_path, index=False)
        stamp = mock.Mock(side_effect=[STAMP, "20260925T200001"])
        plan = {"horizons": [{"horizon": 3, "n_folds": 3}, {"horizon": 1, "n_folds": 9}],
                "models": ["naive_last"]}
        run_main(self.root, plan, matrix(SERIES), realization_stamp=stamp)
        snapshots = self.root / "results" / "realizations"
        self.assertEqual(sorted(p.name for p in snapshots.iterdir()),
                         [f"naive_last__h1__{STAMP}.csv", f"naive_last__h3__{STAMP}.csv"])
        self.assertEqual(stamp.call_count, 1)
        for name, n_rows in ((f"naive_last__h1__{STAMP}.csv", 9 * 3), (f"naive_last__h3__{STAMP}.csv", 3 * 3)):
            with self.subTest(name):
                snapshot = pd.read_csv(snapshots / name)
                self.assertEqual(len(snapshot), n_rows)
                self.assertIn("mo", snapshot.columns)  # снимок файла по рядам, не шагов
                self.assertTrue((snapshot["mae"] == 999.0).all())

    def test_plan_on_other_series_stops_before_any_model(self):
        five = [f"мо_{i}" for i in range(5)]
        self.write_files(five)
        before = self.per_path.read_bytes()
        series, panel = mock.Mock(side_effect=_model_started), mock.Mock(side_effect=_model_started)
        with self.assertRaises(ValueError) as caught:
            run_main(self.root, H3, matrix(five[:3]),
                     evaluate_series_models=series, evaluate_panel_model=panel)
        self.assertIn("модели не запускались", str(caught.exception))
        series.assert_not_called()
        panel.assert_not_called()
        self.assertEqual(self.per_path.read_bytes(), before)

    def test_plan_check_works_against_gz_snapshot_when_plain_csv_is_missing(self):
        # I1: на чистом клоне обычного файла нет, есть только horizons_per_series.csv.gz —
        # сверка плана должна сработать против него, а не молча пропуститься.
        five = [f"мо_{i}" for i in range(5)]
        gz_path = self.per_path.with_name(self.per_path.name + ".gz")
        pd.concat([per_series("naive_last", 3, [0, 1, 2], series=five, mae=999.0),
                   per_series("drift", 3, [0, 1, 2], series=five, mae=5.0)],
                  ignore_index=True).to_csv(gz_path, index=False, compression="gzip")
        series, panel = mock.Mock(side_effect=_model_started), mock.Mock(side_effect=_model_started)
        with self.assertRaises(ValueError) as caught:
            run_main(self.root, H3, matrix(five[:3]),
                     evaluate_series_models=series, evaluate_panel_model=panel)
        self.assertIn("модели не запускались", str(caught.exception))
        series.assert_not_called()
        panel.assert_not_called()
        self.assertFalse(self.per_path.exists())  # план отклонён — обычный файл не появился

    def test_batch_merges_with_gz_snapshot_when_plain_csv_is_missing(self):
        # Тот же инцидент целиком через main(): пересчёт сводок горизонтов не должен
        # потерять drift, которого не было в партии, только потому что обычного файла
        # ещё нет — есть только снимок.
        gz_path = self.per_path.with_name(self.per_path.name + ".gz")
        pd.concat([per_series("naive_last", 3, [0, 1, 2], mae=999.0),
                   per_series("drift", 3, [0, 1, 2], mae=5.0)],
                  ignore_index=True).to_csv(gz_path, index=False, compression="gzip")
        run_main(self.root, H3, matrix(SERIES))
        self.assertTrue(self.per_path.exists())
        on_disk = pd.read_csv(self.per_path)
        self.assertEqual(pairs(on_disk), [("drift", 3), ("naive_last", 3)])
        self.assertFalse((on_disk.loc[on_disk["model"] == "naive_last", "mae"] == 999.0).any())
        self.assertEqual(on_disk.loc[on_disk["model"] == "drift", "mae"].unique().tolist(), [5.0])

    def test_merges_existing_and_new_horizon_and_writes_steps(self):
        self.write_files(SERIES)
        runner = mock.Mock(wraps=_evaluate_series_models)
        plan = {"horizons": [{"horizon": 3, "n_folds": 3}, {"horizon": 1, "n_folds": 9}],
                "models": ["naive_last"]}
        run_main(self.root, plan, matrix(SERIES), evaluate_series_models=runner)
        self.assertEqual(runner.call_count, 2)  # горизонт 3 из файла и новый горизонт 1

        on_disk = pd.read_csv(self.per_path)
        self.assertEqual(pairs(on_disk), [("drift", 3), ("naive_last", 1), ("naive_last", 3)])
        naive3 = on_disk[(on_disk["model"] == "naive_last") & (on_disk["horizon"] == 3)]
        self.assertEqual(len(naive3), len(SERIES) * 3)
        self.assertFalse((naive3["mae"] == 999.0).any())
        naive1 = on_disk[(on_disk["model"] == "naive_last") & (on_disk["horizon"] == 1)]
        self.assertEqual(len(naive1), len(SERIES) * 9)
        self.assertEqual(on_disk.loc[on_disk["model"] == "drift", "mae"].unique().tolist(), [5.0])

        steps_on_disk = pd.read_csv(self.steps_path)
        self.assertEqual(pairs(steps_on_disk), [("drift", 3), ("naive_last", 1), ("naive_last", 3)])
        new_steps = steps_on_disk[steps_on_disk["model"] == "naive_last"]
        self.assertFalse((new_steps["mae"] == 999.0).any())
        self.assertEqual(len(new_steps), 3 * 3 + 9 * 1)  # фолды × шаги горизонта
        self.assertEqual(steps_on_disk.loc[steps_on_disk["model"] == "drift", "mae"].unique().tolist(), [5.0])

        summary = pd.read_csv(self.root / "results" / "horizons_summary.csv")
        self.assertEqual(list(summary.columns), ["horizon", "model", *SUMMARY, "фолдов", "фолдов зачтено"])
        self.assertEqual(sorted(zip(summary["model"], summary["horizon"])),
                         [("drift", 3), ("naive_last", 1), ("naive_last", 3)])
        self.assertTrue((self.root / "results" / "horizons_folds.csv").exists())

    def test_batch_on_other_series_is_stopped_before_writing(self):
        # План сходится с файлом, а партия — нет: прогон потерял ряд. Ранняя сверка
        # такого не видит, поздняя в merge_into ловит до записи обоих файлов.
        self.write_files(SERIES)
        before = self.per_path.read_bytes(), self.steps_path.read_bytes()

        def lossy(wide, name, horizon, folds, workers):
            return _evaluate_series_models(wide.iloc[:, :2], name, horizon, folds, workers)

        runner = mock.Mock(side_effect=lossy)
        with self.assertRaises(ValueError) as caught:
            run_main(self.root, H3, matrix(SERIES), evaluate_series_models=runner)
        runner.assert_called()
        self.assertIn("в файл ничего не записано", str(caught.exception))
        self.assertEqual((self.per_path.read_bytes(), self.steps_path.read_bytes()), before)

    def test_progress_line_counts_refusal_without_text(self):
        self.write_files(SERIES)

        def hidden_refusal(wide, name, horizon, folds, workers):
            part, per_fold_steps = _evaluate_series_models(wide, name, horizon, folds, workers)
            part.loc[0, ["mae", "smape", "mase"]] = np.nan  # отказ без текста: error пуст
            return part, per_fold_steps

        out = run_main(self.root, H3, matrix(SERIES),
                       evaluate_series_models=mock.Mock(side_effect=hidden_refusal))
        self.assertIn("naive_last h=3:", out)
        self.assertIn("| отказов 1 |", out)


if __name__ == "__main__":
    unittest.main()
