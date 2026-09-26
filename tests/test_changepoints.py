"""Детекторы в штрафном режиме, стенд разладок и потоковый сигнал на панели:
`src/changepoints.py`, `src/cp_bench.py`, `scripts/changepoints.py`.

Ряды синтетические, файлы пишутся только во временный каталог: живые
`results/cp_*.csv` тесты не трогают.
"""
from __future__ import annotations

import contextlib
import importlib.util
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

from src import cp_bench  # noqa: E402
from src.changepoints import DETECTORS, detect_pelt  # noqa: E402
from src.cp_bench import (  # noqa: E402
    BENCH_COLUMNS, MIN_HISTORY, RECENT_WINDOW, SUMMARY_COLUMNS, inject, panel_realtime,
    preprocess, realtime_signals, run_bench, streaming_signal, summarise,
)

# scripts/ — не пакет: скрипт грузится по пути, без правки sys.path под все тесты.
# Имя модуля своё: `changepoints` уже занято детекторами в src.
_spec = importlib.util.spec_from_file_location("changepoints_script", ROOT / "scripts" / "changepoints.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

PENALIZED = ["pelt", "binseg", "window", "bottomup", "kernel_rbf"]
# Штраф, при котором чистая синусоида целиком не окупает ни одного излома, а сдвиг в 4σ
# окупает. У методов l2 штраф в единицах дисперсии ряда; у ядра стоимость в пространстве
# ядра, и масштаб другой — поэтому на стенде его штраф калибруется, а не берётся как есть.
# Значения — середина диапазона, где проходят оба теста: у l2 от 9,5 (bottomup) до 19,
# у ядра от 2,75 до 7; на краю тест зависел бы от мелочей реализации ruptures.
TEST_PENALTY = {"pelt": 14.0, "binseg": 14.0, "window": 14.0, "bottomup": 14.0, "kernel_rbf": 5.0}
MONTHS = pd.date_range("2023-01-01", periods=24, freq="MS")
KEY = ["detector", "mode", "penalty"]


def sine(n: int = 24) -> np.ndarray:
    """Чистая сезонность без шума и без изломов: период 12, как у месячных расходов."""
    t = np.arange(n, dtype=float)
    return 100.0 + 10.0 * np.sin(2 * np.pi * t / 12)


def noisy_panel(n_series: int, n_months: int = 24, seed: int = 0) -> pd.DataFrame:
    """Ряды расходов без структурных изменений: свой уровень, сезонность и шум."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months, dtype=float)
    levels = rng.uniform(1_000.0, 50_000.0, n_series)
    season = 1 + 0.05 * np.sin(2 * np.pi * t / 12)
    values = levels * season[:, None] * (1 + 0.03 * rng.standard_normal((n_months, n_series)))
    return pd.DataFrame(values, index=MONTHS[:n_months], columns=[f"мо_{i}" for i in range(n_series)])


def shifted_panel(n_series: int = 20, shift_at: int = 14, jump: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """Панель с общим сдвигом уровня: с месяца `shift_at` все ряды выше на долю `jump`."""
    rng = np.random.default_rng(seed)
    levels = rng.uniform(1_000.0, 50_000.0, n_series)
    values = levels * (1 + 0.01 * rng.standard_normal((24, n_series)))
    values[shift_at:] *= 1 + jump
    return pd.DataFrame(values, index=MONTHS, columns=[f"мо_{i}" for i in range(n_series)])


def month(i: int) -> str:
    return MONTHS[i].strftime("%Y-%m")


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


class StreamingSignalTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = 100 + rng.standard_normal(20)
        self.y[12:] += 3.0

    def test_penalty_reaches_every_penalized_detector(self):
        # 20.09 перебор штрафа ничего не менял: потоковая функция звала детектор с его
        # штрафом по умолчанию. Здесь крайние штрафы обязаны дать разные ответы.
        for name in PENALIZED:
            with self.subTest(name):
                self.assertIsNotNone(streaming_signal(self.y, name, "raw", penalty=0.01))
                self.assertIsNone(streaming_signal(self.y, name, "raw", penalty=1e6))

    def test_cusum_keeps_its_own_threshold(self):
        alone = streaming_signal(self.y, "cusum", "raw")
        self.assertIsNotNone(alone)
        self.assertEqual(streaming_signal(self.y, "cusum", "raw", penalty=1e6), alone)


class InjectTest(unittest.TestCase):
    def test_variance_noise_differs_between_series_and_repeats_for_the_same_seed(self):
        # Прежде генератор сеялся позицией врезки: у всех рядов стенда был один и тот же шум.
        y = np.linspace(100.0, 120.0, 24)

        def spoil(i: int, position: int) -> np.ndarray:
            return inject(y, "variance", position, 2.0, rng=np.random.default_rng([7, i, position]))

        self.assertTrue(np.array_equal(spoil(0, 14), spoil(0, 14)))
        self.assertFalse(np.array_equal(spoil(0, 14), spoil(1, 14)))
        self.assertFalse(np.array_equal(spoil(0, 14)[17:], spoil(0, 17)[17:]))

    def test_variance_without_generator_is_rejected(self):
        with self.assertRaises(ValueError):
            inject(np.arange(24.0), "variance", 14, 1.0)

    def test_level_and_trend_need_no_generator(self):
        y = np.arange(24.0)
        self.assertEqual(inject(y, "level", 14, 1.0)[13], y[13])
        self.assertGreater(inject(y, "trend", 14, 1.0)[23], y[23])


BENCH = {
    "positions": (14,), "magnitudes": (4.0,), "kinds": ("level", "variance"), "modes": ("ratio",),
    "detectors": ("pelt", "kernel_rbf", "cusum"), "penalties": (1.0, 3.0), "seed": 7,
    "kernel_penalties": {1.0: 0.5, 3.0: 2.0},
}


class RunBenchTest(unittest.TestCase):
    def setUp(self):
        self.panel = noisy_panel(5)

    def test_columns_rows_and_repeatability(self):
        first = run_bench(self.panel, **BENCH)
        second = run_bench(self.panel, **BENCH)
        self.assertEqual(list(first.columns), BENCH_COLUMNS)
        pd.testing.assert_frame_equal(first, second)
        # 5 рядов × (pelt и ядро при двух штрафах, CUSUM один раз) × 1 режим × (чистый + 2 возмущения)
        self.assertEqual(len(first), 5 * 5 * 1 * 3)
        self.assertEqual(set(first["series_id"]), set(self.panel.columns))

    def test_penalty_columns_per_detector(self):
        bench = run_bench(self.panel, **BENCH)
        cusum = bench[bench["detector"] == "cusum"]
        self.assertTrue(cusum["penalty"].isna().all() and cusum["penalty_effective"].isna().all())
        kernel = bench[bench["detector"] == "kernel_rbf"]
        self.assertEqual(kernel["penalty_effective"].tolist(), kernel["penalty"].map(BENCH["kernel_penalties"]).tolist())
        pelt = bench[bench["detector"] == "pelt"]
        self.assertTrue((pelt["penalty_effective"] == pelt["penalty"]).all())

    def test_rows_agree_with_the_signal(self):
        bench = run_bench(self.panel, **BENCH)
        clean = bench[bench["kind"] == "none"]
        self.assertTrue(clean["position"].isna().all())
        self.assertTrue((clean["false_alarm"] == clean["signal"].notna()).all())
        spoiled = bench[bench["kind"] != "none"]
        self.assertTrue((spoiled["position"] == 14).all())
        self.assertTrue((spoiled["detected"] == (spoiled["signal"] >= 14)).all())
        self.assertTrue((spoiled["false_alarm"] == (spoiled["signal"] < 14)).all())
        hit = spoiled[spoiled["detected"]]
        self.assertTrue((hit["delay"] == hit["signal"] - 14).all())

    def test_kernel_without_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            run_bench(self.panel, **{**BENCH, "kernel_penalties": None})

    def test_variance_is_seeded_by_config_seed_series_number_and_position(self):
        with mock.patch.object(cp_bench, "inject", wraps=inject) as spy:
            run_bench(self.panel, **{**BENCH, "kinds": ("variance",), "positions": (11, 14)})
        seeds = [(c.args[2], c.kwargs["rng"].bit_generator.seed_seq.entropy) for c in spy.call_args_list]
        expected = [(position, [BENCH["seed"], i, position]) for i in range(5) for position in (11, 14)]
        self.assertEqual(seeds, expected)


def bench_rows(detector, penalty, spoiled, clean, magnitude=1.0) -> list[dict]:
    """Строки стенда: `spoiled` — пары (обнаружено, задержка), `clean` — ложные тревоги."""
    rows = [{"detector": detector, "mode": "ratio", "penalty": penalty, "kind": "level",
             "magnitude": magnitude, "detected": hit, "delay": delay, "false_alarm": False}
            for hit, delay in spoiled]
    rows += [{"detector": detector, "mode": "ratio", "penalty": penalty, "kind": "none",
              "magnitude": 0.0, "detected": alarm, "delay": np.nan, "false_alarm": alarm}
             for alarm in clean]
    return rows


class SummariseTest(unittest.TestCase):
    def setUp(self):
        self.bench = pd.DataFrame(
            bench_rows("pelt", 1.0, [(True, 0), (True, 2), (True, 4), (False, np.nan)],
                       [True, False, False, False, False])
            + bench_rows("cusum", np.nan, [(True, 1), (False, np.nan)], [True, True])
        )

    def test_columns_and_youden(self):
        summary = summarise(self.bench)
        self.assertEqual(list(summary.columns), KEY + SUMMARY_COLUMNS)
        pelt = summary.iloc[0]  # сортировка по J Юдена: pelt +55 выше CUSUM −50
        self.assertEqual(pelt["detector"], "pelt")
        self.assertAlmostEqual(pelt["обнаружено, %"], 75.0)
        self.assertAlmostEqual(pelt["ложных на чистых, %"], 20.0)
        self.assertAlmostEqual(pelt["J Юдена"], pelt["обнаружено, %"] - pelt["ложных на чистых, %"])
        self.assertEqual((pelt["n испорченных"], pelt["n чистых"]), (4, 5))

    def test_delay_is_counted_among_detected_only(self):
        pelt = summarise(self.bench).iloc[0]
        self.assertEqual(pelt["задержка, медиана"], 2.0)
        self.assertEqual(pelt["задержка, среднее"], 2.0)
        self.assertAlmostEqual(pelt["доля с задержкой 0, %"], 100 / 3)

    def test_cusum_without_penalty_stays_in_the_summary(self):
        cusum = summarise(self.bench).iloc[1]
        self.assertEqual(cusum["detector"], "cusum")
        self.assertTrue(np.isnan(cusum["penalty"]))
        self.assertAlmostEqual(cusum["J Юдена"], 50.0 - 100.0)

    def test_by_magnitude_shares_the_clean_rate(self):
        bench = pd.DataFrame(
            bench_rows("pelt", 1.0, [(True, 3), (False, np.nan)], [True, False, False, False], magnitude=1.0)
            + bench_rows("pelt", 1.0, [(True, 1), (True, 2)], [], magnitude=4.0)
        )
        by = summarise(bench, by=("magnitude",))
        self.assertEqual(list(by.columns), KEY + ["magnitude"] + SUMMARY_COLUMNS)
        self.assertEqual(by["magnitude"].tolist(), [1.0, 4.0])
        self.assertEqual(by["обнаружено, %"].tolist(), [50.0, 100.0])
        self.assertEqual(by["ложных на чистых, %"].tolist(), [25.0, 25.0])
        self.assertEqual(by["n чистых"].tolist(), [4, 4])


class CalibrationTest(unittest.TestCase):
    def test_nearest_false_alarm_rate_and_smaller_pen_on_tie(self):
        clean = noisy_panel(6, n_months=16, seed=3)
        # Штраф 1e6 не пропускает у PELT ни одного излома: доля ложных 0, и на хвосте сетки,
        # где ядро тоже молчит, ближайших значений несколько — берётся меньшее.
        grid = np.array([0.05, 0.5, 5.0, 25.0, 30.0])
        table = cp_bench.calibrate_kernel_penalty(clean, (1.0, 1e6), "ratio", grid=grid)
        self.assertEqual(list(table.columns), ["penalty", "kernel_pen", "fa_pelt", "fa_kernel"])
        self.assertEqual(table["penalty"].tolist(), [1.0, 1e6])

        def alarms(detector, penalty) -> int:
            return sum(streaming_signal(clean[c].to_numpy(float), detector, "ratio", penalty=penalty) is not None
                       for c in clean.columns)

        n = clean.shape[1]
        kernel = np.array([alarms("kernel_rbf", pen) for pen in grid])
        for row in table.itertuples(index=False):
            with self.subTest(penalty=row.penalty):
                pelt = alarms("pelt", row.penalty)
                self.assertAlmostEqual(row.fa_pelt, 100 * pelt / n)
                distance = np.abs(kernel - pelt)
                self.assertEqual(row.kernel_pen, grid[distance == distance.min()].min())
                self.assertAlmostEqual(row.fa_kernel, 100 * kernel[grid == row.kernel_pen][0] / n)
        # ничья на самом деле разыграна: PELT молчит, и ядро молчит больше чем при одном штрафе
        self.assertEqual(alarms("pelt", 1e6), 0)
        self.assertGreater((kernel == 0).sum(), 1)


class PanelRealtimeTest(unittest.TestCase):
    SHIFT = 14

    def setUp(self):
        self.panel = shifted_panel(shift_at=self.SHIFT)

    def realtime(self, **kwargs):
        args = {"detector": "pelt", "mode": "ratio", "penalty": 1.0, "threshold_share": 50,
                "events": [month(self.SHIFT)], **kwargs}
        return panel_realtime(self.panel, **args)

    def test_common_shift_crosses_threshold_within_recent_window(self):
        monthly, events = self.realtime()
        self.assertEqual(list(monthly.columns), ["month", "share", "n_signals"])
        self.assertEqual(monthly["month"].tolist(), [month(i) for i in range(24)])
        self.assertTrue((monthly["share"].iloc[: self.SHIFT] < 50).all())
        self.assertEqual(list(events.columns), ["event", "crossed_month", "delay", "max_share"])
        row = events.iloc[0]
        crossed = monthly["month"].tolist().index(row["crossed_month"])
        self.assertTrue(self.SHIFT <= crossed <= self.SHIFT + RECENT_WINDOW, row["crossed_month"])
        self.assertEqual(row["delay"], crossed - self.SHIFT)
        self.assertEqual(row["max_share"], monthly["share"].iloc[self.SHIFT:].max())

    def test_one_break_is_counted_once_per_series(self):
        # Излом остаётся в последних точках окна несколько шагов подряд; без паузы
        # после сигнала ряд попал бы в долю в каждом из них.
        y = self.panel.iloc[:, 0].to_numpy(float)
        late = self.SHIFT + RECENT_WINDOW
        seen_late = detect_pelt(preprocess(y[: late + 1], "ratio"), penalty=1.0).breakpoints
        self.assertTrue(any(b + 1 >= late - RECENT_WINDOW for b in seen_late))
        signals = realtime_signals(y, "pelt", "ratio", 1.0)
        near = [t for t in signals if self.SHIFT <= t <= late]
        self.assertEqual(len(near), 1)
        self.assertTrue(all(b - a > RECENT_WINDOW for a, b in zip(signals, signals[1:])))

    def test_first_signal_is_the_streaming_signal(self):
        for col in self.panel.columns:
            y = self.panel[col].to_numpy(float)
            signals = realtime_signals(y, "pelt", "ratio", 1.0)
            with self.subTest(col):
                self.assertEqual(signals[0] if signals else None, streaming_signal(y, "pelt", "ratio", penalty=1.0))
                self.assertTrue(all(t >= MIN_HISTORY for t in signals))

    def test_event_without_crossing_has_empty_month_and_delay(self):
        monthly, events = self.realtime(threshold_share=101)
        row = events.iloc[0]
        self.assertTrue(pd.isna(row["crossed_month"]) and pd.isna(row["delay"]))
        self.assertEqual(row["max_share"], monthly["share"].iloc[self.SHIFT:].max())

    def test_max_share_stops_before_the_next_event_month(self):
        first, second = self.SHIFT - 5, self.SHIFT
        monthly, events = self.realtime(events=[month(first), month(second)])
        share = monthly["share"]
        # иначе проверка не отличила бы окно с месяцем следующего события от окна без него
        self.assertGreater(share.iloc[second], share.iloc[first:second].max())
        self.assertEqual(events["max_share"].iloc[0], share.iloc[first:second].max())
        self.assertEqual(events["max_share"].iloc[1], share.iloc[second:].max())

    def test_unknown_event_month_is_rejected(self):
        with self.assertRaises(ValueError):
            self.realtime(events=["2030-01"])


def summary_rows(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame([{"detector": d, "mode": m, "penalty": p, "J Юдена": j} for d, m, p, j in rows])


class SelectPenaltyTest(unittest.TestCase):
    def test_highest_youden_of_the_detector_and_mode(self):
        summary = summary_rows([("pelt", "ratio", 1.0, 30.0), ("pelt", "ratio", 3.0, 40.0),
                                ("pelt", "raw", 1.0, 90.0), ("cusum", "ratio", np.nan, 95.0)])
        self.assertEqual(script.select_penalty(summary, "pelt", "ratio"), 3.0)

    def test_tie_goes_to_the_smaller_penalty(self):
        # Разность долей с разными знаменателями: равные по смыслу J расходятся в последнем разряде.
        # Больший из двух достаётся большему штрафу: без округления выбор ушёл бы к нему.
        a = 100 * 30 / 540 - 100 * 2 / 60
        b = 100 * 39 / 540 - 100 * 3 / 60
        self.assertNotEqual(a, b)
        summary = summary_rows([("pelt", "ratio", 3.0, max(a, b)), ("pelt", "ratio", 1.0, min(a, b))])
        self.assertEqual(script.select_penalty(summary, "pelt", "ratio"), 1.0)

    def test_detector_without_penalty_rows_is_rejected(self):
        summary = summary_rows([("cusum", "ratio", np.nan, 10.0)])
        with self.assertRaises(ValueError):
            script.select_penalty(summary, "cusum", "ratio")


class OfflineSharesTest(unittest.TestCase):
    def test_break_in_growth_rates_is_dated_by_the_new_level(self):
        # Излом b в ряду темпов роста — месяц b + 1: с него уровень ряда другой.
        panel = shifted_panel(shift_at=14)
        table = script.offline_shares(panel, "pelt", "ratio", 1.0)
        self.assertEqual(list(table.columns), ["month", "share", "n_breaks"])
        self.assertEqual(table["share"].idxmax(), 14)
        self.assertEqual(table.loc[14, "month"], month(14))
        self.assertAlmostEqual(table.loc[14, "share"], table.loc[14, "n_breaks"] / panel.shape[1] * 100)


class _Report:
    def as_text(self) -> str:
        return ""


CONFIG = {
    "data": {"path": "panel.parquet", "category": "Все категории", "max_gap": 2},
    "bench": {"n_series": 3, "seed": 5, "positions": [14], "magnitudes": [4.0], "kinds": ["level"],
              "modes": ["ratio"], "detectors": ["pelt", "kernel_rbf", "cusum"], "penalties": [1.0, 3.0]},
    "selection": {"rule": "J Юдена на режиме ratio, среди штрафов из сетки; при равенстве — меньший штраф"},
    "realtime": {"detector": "pelt", "mode": "ratio", "threshold_share": 50, "events": ["2024-03"]},
    "output": {"dir": "results"},
}
FILES = {
    "cp_bench.csv": BENCH_COLUMNS,
    "cp_summary.csv": KEY + SUMMARY_COLUMNS,
    "cp_summary_by_magnitude.csv": KEY + ["magnitude"] + SUMMARY_COLUMNS,
    "cp_calibration.csv": ["penalty", "kernel_pen", "fa_pelt", "fa_kernel"],
    "cp_realtime.csv": ["penalty", "month", "share", "n_signals"],
    "cp_realtime_events.csv": ["penalty", "selected", "event", "crossed_month", "delay", "max_share"],
    "cp_offline.csv": ["penalty", "month", "share", "n_breaks"],
}


class MainTest(unittest.TestCase):
    """`main()` целиком на синтетике: от калибровки до записанных файлов."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8")
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for name, value in {"ROOT": self.root, "load_panel": mock.Mock(return_value=None),
                                "build_matrix": mock.Mock(return_value=(shifted_panel(8), _Report()))}.items():
                stack.enter_context(mock.patch.object(script, name, value))
            stack.enter_context(mock.patch.object(sys, "argv", ["changepoints.py", "--config", str(config)]))
            stack.enter_context(contextlib.redirect_stdout(out))
            script.main()
        self.log = out.getvalue()
        self.results = self.root / "results"

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.results / name)

    def test_every_file_is_written_with_its_columns(self):
        for name, columns in FILES.items():
            with self.subTest(name):
                self.assertEqual(list(self.read(name).columns), columns)

    def test_penalty_is_chosen_by_the_rule_before_the_real_data(self):
        summary = self.read("cp_summary.csv")
        chosen = script.select_penalty(summary, "pelt", "ratio")
        events = self.read("cp_realtime_events.csv")
        self.assertEqual(sorted(events["penalty"]), [1.0, 3.0])
        self.assertEqual(events.loc[events["selected"], "penalty"].tolist(), [chosen])
        self.assertLess(self.log.index(CONFIG["selection"]["rule"]), self.log.index("РЕАЛЬНЫЕ ДАННЫЕ"))

    def test_bench_uses_the_calibrated_kernel_penalty(self):
        calibration = self.read("cp_calibration.csv")
        bench = self.read("cp_bench.csv")
        kernel = bench[bench["detector"] == "kernel_rbf"]
        expected = kernel["penalty"].map(dict(zip(calibration["penalty"], calibration["kernel_pen"])))
        self.assertTrue(np.allclose(kernel["penalty_effective"], expected))

    def test_realtime_and_offline_cover_every_penalty_and_month(self):
        for name in ["cp_realtime.csv", "cp_offline.csv"]:
            with self.subTest(name):
                table = self.read(name)
                self.assertEqual(len(table), 2 * 24)
                self.assertEqual(sorted(table["penalty"].unique()), [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
