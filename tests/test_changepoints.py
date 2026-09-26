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
from src.changepoints import DETECTORS, Detection, detect_kernel, detect_pelt  # noqa: E402
from src.cp_bench import (  # noqa: E402
    BENCH_COLUMNS, CALIBRATION_COLUMNS, MIN_HISTORY, RECENT_WINDOW, SUMMARY_COLUMNS, inject,
    panel_realtime, preprocess, realtime_signals, run_bench, streaming_signal, summarise,
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


def flagged(y: np.ndarray, t: int, penalty: float = 1.0) -> bool:
    """Флаг потокового правила в месяц t, заново по детектору: PELT на темпах роста y[:t+1]
    нашёл излом в последних RECENT_WINDOW точках."""
    found = detect_pelt(preprocess(y[: t + 1], "ratio"), penalty=penalty).breakpoints
    return any(b + 1 >= t - RECENT_WINDOW for b in found)


def hysteresis_starts(flags: dict[int, bool]) -> list[int]:
    """Начала эпизодов по определению: флаг в t и ни одного в t − 1, …, t − RECENT_WINDOW."""
    return [t for t, on in flags.items()
            if on and not any(flags.get(t - k, False) for k in range(1, RECENT_WINDOW + 1))]


def pattern_detector(pattern: str):
    """Заглушка с заданными флагами: излом у конца окна ровно в месяцы с «X», первый
    символ — месяц MIN_HISTORY. Значения ряда не важны — только длина окна (режим raw)."""
    def detect(window, **kwargs):
        i = len(window) - 1 - MIN_HISTORY
        near_end = 0 <= i < len(pattern) and pattern[i] == "X"
        return Detection("pattern", [len(window) - 1] if near_end else [])
    return detect


def exploding_detector(window, **kwargs):
    """Заглушка сломанного детектора: падает на каждом шаге."""
    raise RuntimeError("детектор сломан")


def flaky_pelt(window, penalty=3.0):
    """PELT, который падает на окнах короче полного ряда темпов роста (23 точки на 24
    месяцах): в потоке сбой на всех шагах, кроме последнего, по полному ряду — как обычно."""
    if len(window) < 23:
        raise RuntimeError("детектор сломан на коротком окне")
    return detect_pelt(window, penalty=penalty)


def threshold_detector(window, penalty=3.0):
    """Заглушка для калибровки: тревога у конца окна, пока штраф ниже уровня ряда.
    Число рядов с тревогой при штрафе — число рядов с уровнем выше него."""
    return Detection("threshold", [len(window) - 1] if penalty < window[0] else [])


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

    def test_clean_rows_are_never_detected_and_the_summary_does_not_read_them(self):
        bench = run_bench(self.panel, **BENCH)
        clean = bench["kind"] == "none"
        self.assertFalse(bench.loc[clean, "detected"].any())
        # прежняя запись: у чистой строки «обнаружено» = была тревога; числа сводки те же
        legacy = bench.copy()
        legacy.loc[clean, "detected"] = legacy.loc[clean, "false_alarm"]
        self.assertTrue(legacy.loc[clean, "detected"].any())
        pd.testing.assert_frame_equal(summarise(bench), summarise(legacy))

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
             "magnitude": magnitude, "detected": hit, "delay": delay, "false_alarm": False, "n_failed": 0}
            for hit, delay in spoiled]
    rows += [{"detector": detector, "mode": "ratio", "penalty": penalty, "kind": "none",
              "magnitude": 0.0, "detected": False, "delay": np.nan, "false_alarm": alarm, "n_failed": 0}
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
    def test_nearest_false_alarm_rate_and_smallest_pen_on_tie(self):
        clean = noisy_panel(6, n_months=16, seed=3)
        # Штраф 1e6 не пропускает у PELT ни одного излома: доля ложных 0, и на хвосте сетки,
        # где ядро тоже молчит, ближайших значений несколько — берётся меньший штраф.
        grid = np.array([0.05, 0.5, 5.0, 25.0, 30.0])
        table = cp_bench.calibrate_kernel_penalty(clean, (1.0, 1e6), "ratio", grid=grid)
        self.assertEqual(list(table.columns), CALIBRATION_COLUMNS)
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

    def calibrate(self, levels, penalty, grid):
        """Калибровка на заглушке (`threshold_detector`) вместо PELT и ядра: число рядов
        с тревогой при штрафе p — число уровней выше p, плато задаются уровнями."""
        sample = pd.DataFrame({f"мо_{i}": np.full(12, level) for i, level in enumerate(levels)})
        with mock.patch.dict(cp_bench.DETECTORS, {"pelt": threshold_detector, "kernel_rbf": threshold_detector}):
            return cp_bench.calibrate_kernel_penalty(sample, (penalty,), "raw", grid=np.asarray(grid, float))

    def test_plateau_of_equal_rates_gives_its_smallest_pen(self):
        # Уровни 0,5; 1,5 и трижды 6,5: при штрафе 2 у PELT тревога у трёх рядов, у ядра —
        # тоже три при штрафах 2…6. Берётся 2 — самое чувствительное ядро при той же доле
        # ложных тревог; середина плато (4) зависела бы от того, где кончается сетка.
        table = self.calibrate([0.5, 1.5, 6.5, 6.5, 6.5], 2.0, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(table.loc[0, "kernel_pen"], 2.0)
        self.assertEqual((table.loc[0, "fa_pelt"], table.loc[0, "fa_kernel"]), (60.0, 60.0))

    def test_plateau_reaching_the_upper_edge_gives_its_inner_end(self):
        # Как при штрафе 3 на крошечном прогоне: у PELT ни одной тревоги, ядро молчит от 3
        # до конца сетки. Выбор — 3, внутри сетки; предупреждения о крае нет.
        grid = [1, 2, 3, 4, 5, 6, 7]
        table = self.calibrate([0.5, 1.5, 2.5, 2.5, 2.5], 3.0, grid)
        self.assertEqual(table.loc[0, "kernel_pen"], 3.0)
        self.assertEqual(script.edge_warnings(table, np.asarray(grid, float)), [])

    def test_plateau_from_the_lower_edge_gives_the_edge_and_a_warning(self):
        # Обе доли 100% на всей сетке, как у штрафа 1,0 на крошечном прогоне: меньший штраф —
        # нижний край, и лог это отмечает.
        grid = [1, 2, 3, 4, 5, 6, 7]
        table = self.calibrate([9.0] * 5, 2.0, grid)
        self.assertEqual(table.loc[0, "kernel_pen"], 1.0)
        lines = script.edge_warnings(table, np.asarray(grid, float))
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("ВНИМАНИЕ: калибровка на краю сетки"))


class PanelRealtimeTest(unittest.TestCase):
    SHIFT = 14

    def setUp(self):
        self.panel = shifted_panel(shift_at=self.SHIFT)

    def realtime(self, panel=None, **kwargs):
        args = {"detector": "pelt", "mode": "ratio", "penalty": 1.0, "threshold_share": 50,
                "events": [month(self.SHIFT)], **kwargs}
        return panel_realtime(self.panel if panel is None else panel, **args)

    def signals(self, col: str, penalty: float = 1.0) -> list[int]:
        return realtime_signals(self.panel[col].to_numpy(float), "pelt", "ratio", penalty)

    def test_signals_follow_the_hysteresis_rule(self):
        # Определение по флагам, посчитанным здесь заново. При штрафе 1 на панели есть эпизоды
        # шума до сдвига и слитые с эпизодом сдвига, при штрафе 3 — перерывы во флагах после него.
        for penalty in (1.0, 3.0):
            for col in self.panel.columns:
                y = self.panel[col].to_numpy(float)
                flags = {t: flagged(y, t, penalty) for t in range(MIN_HISTORY, len(y))}
                with self.subTest(col, penalty=penalty):
                    self.assertEqual(self.signals(col, penalty), hysteresis_starts(flags))

    def test_short_gaps_after_the_shift_no_longer_start_a_second_episode(self):
        # При штрафе 3 серия флагов после разового сдвига прерывается на месяц-два перед
        # вторым изломом, и по одному переходу «нет флага → флаг» у части рядов в течение
        # шести месяцев после сдвига начинался второй эпизод. Гистерезис его убирает. Второй
        # эпизод остаётся, только если флагов не было RECENT_WINDOW месяцев подряд, — по
        # определению это два эпизода (здесь у одного ряда перерыв ровно в три месяца).
        near = range(self.SHIFT, self.SHIFT + 7)
        rising_echo = echo = 0
        for col in self.panel.columns:
            y = self.panel[col].to_numpy(float)
            flags = {t: flagged(y, t, 3.0) for t in range(MIN_HISTORY, len(y))}
            rising = [t for t, on in flags.items() if on and not flags.get(t - 1, False)]
            rising_echo += sum(t in near for t in rising) > 1
            second = [t for t in self.signals(col, 3.0) if t in near][1:]
            echo += bool(second)
            for t in second:
                with self.subTest(col, t=t):
                    self.assertFalse(any(flags.get(t - k, False) for k in range(1, RECENT_WINDOW + 1)))
        self.assertGreaterEqual(rising_echo, 5)
        self.assertLess(echo, rising_echo)

    def test_single_shift_gives_one_signal_per_series(self):
        # Сдвиг в первый проверяемый месяц: до него сигналить нечему, после него дисперсию
        # темпов роста задаёт выброс, и шум тревог не поднимает. Флаг у каждого ряда держится
        # и в m + RECENT_WINDOW + 1 — пауза такой длины дала бы там второй сигнал. Серия флагов
        # здесь у каждого ряда непрерывна; прерванная дала бы второй эпизод — это ограничение
        # правила (`realtime_signals`), а не этой проверки.
        m = MIN_HISTORY
        panel = shifted_panel(shift_at=m)
        for col in panel.columns:
            y = panel[col].to_numpy(float)
            with self.subTest(col):
                self.assertTrue(flagged(y, m + RECENT_WINDOW + 1))
                self.assertEqual(realtime_signals(y, "pelt", "ratio", 1.0), [m])
        monthly, events = self.realtime(panel, events=[month(m)])
        self.assertEqual(monthly["n_signals"].sum(), panel.shape[1])
        self.assertEqual(events.loc[0, "crossed_month"], month(m))
        self.assertEqual(events.loc[0, "share_in_window"], 100.0)

    def test_no_second_signal_after_the_shift(self):
        # Трасса со сдвигом в m = 14: у большинства рядов флаг держится и в m + 4, где прежняя
        # пауза в три месяца давала второй сигнал. Эпизоды шума до сдвига — свои эпизоды,
        # но после сдвига у ряда не больше одного сигнала.
        echo = self.SHIFT + RECENT_WINDOW + 1
        still = sum(flagged(self.panel[col].to_numpy(float), echo) for col in self.panel.columns)
        self.assertGreater(still, self.panel.shape[1] / 2)
        monthly, _ = self.realtime()
        self.assertEqual(monthly["n_signals"].iloc[echo], 0)
        for col in self.panel.columns:
            with self.subTest(col):
                self.assertLessEqual(sum(t >= self.SHIFT for t in self.signals(col)), 1)

    def test_common_shift_crosses_threshold_within_recent_window(self):
        # Штраф 3: при штрафе 1 шум даёт флаги и в месяцы перед сдвигом, и эпизод сдвига
        # у половины рядов сливается с эпизодом шума — ограничение правила, не этой проверки.
        monthly, events = self.realtime(penalty=3.0)
        self.assertEqual(list(monthly.columns), ["month", "share", "n_signals", "n_failed"])
        self.assertEqual(monthly["month"].tolist(), [month(i) for i in range(24)])
        self.assertTrue((monthly["share"].iloc[: self.SHIFT] < 50).all())
        self.assertEqual(list(events.columns), ["event", "crossed_month", "delay", "max_share", "share_in_window"])
        row = events.iloc[0]
        crossed = monthly["month"].tolist().index(row["crossed_month"])
        self.assertTrue(self.SHIFT <= crossed <= self.SHIFT + RECENT_WINDOW, row["crossed_month"])
        self.assertEqual(row["delay"], crossed - self.SHIFT)
        self.assertEqual(row["max_share"], monthly["share"].iloc[self.SHIFT:].max())

    def test_first_signal_is_the_streaming_signal(self):
        for col in self.panel.columns:
            y = self.panel[col].to_numpy(float)
            signals = self.signals(col)
            with self.subTest(col):
                self.assertEqual(signals[0] if signals else None, streaming_signal(y, "pelt", "ratio", penalty=1.0))
                self.assertTrue(all(t >= MIN_HISTORY for t in signals))

    def test_event_without_crossing_has_empty_month_and_delay(self):
        monthly, events = self.realtime(threshold_share=101)
        row = events.iloc[0]
        self.assertTrue(pd.isna(row["crossed_month"]) and pd.isna(row["delay"]))
        self.assertEqual(row["max_share"], monthly["share"].iloc[self.SHIFT:].max())

    def test_crossing_is_searched_inside_the_event_window(self):
        # Первое событие своё окно порогом не проходит; переход в окне второго — чужой
        # и первому не приписывается. Штраф 3 — как в проверке перехода выше.
        first, second = self.SHIFT - 5, self.SHIFT
        monthly, events = self.realtime(penalty=3.0, events=[month(first), month(second)])
        self.assertTrue((monthly["share"].iloc[first:second] < 50).all())
        self.assertTrue(pd.notna(events.loc[1, "crossed_month"]))
        self.assertTrue(pd.isna(events.loc[0, "crossed_month"]) and pd.isna(events.loc[0, "delay"]))

    def test_max_share_stops_before_the_next_event_month(self):
        first, second = self.SHIFT - 5, self.SHIFT
        monthly, events = self.realtime(events=[month(first), month(second)])
        share = monthly["share"]
        # иначе проверка не отличила бы окно с месяцем следующего события от окна без него
        self.assertGreater(share.iloc[second], share.iloc[first:second].max())
        self.assertEqual(events["max_share"].iloc[0], share.iloc[first:second].max())
        self.assertEqual(events["max_share"].iloc[1], share.iloc[second:].max())

    def test_share_in_window_counts_series_with_an_episode_start_inside(self):
        first, second = self.SHIFT - 5, self.SHIFT
        _, events = self.realtime(events=[month(first), month(second)])
        per_series = [self.signals(col) for col in self.panel.columns]
        for row, (start, end) in zip(events.itertuples(index=False), [(first, second), (second, 24)]):
            with self.subTest(row.event):
                inside = sum(any(start <= t < end for t in signals) for signals in per_series)
                self.assertAlmostEqual(row.share_in_window, 100 * inside / len(per_series))
                self.assertGreaterEqual(row.share_in_window, row.max_share)
        # накопленная за окно доля больше пика: эпизоды шума до сдвига разбросаны по месяцам
        self.assertGreater(events.loc[0, "share_in_window"], events.loc[0, "max_share"])

    def test_unknown_event_month_is_rejected(self):
        with self.assertRaises(ValueError):
            self.realtime(events=["2030-01"])


class HysteresisTest(unittest.TestCase):
    """Правило эпизода на заданных флагах (`pattern_detector`)."""

    def signals(self, pattern: str) -> list[int]:
        with mock.patch.dict(cp_bench.DETECTORS, {"pattern": pattern_detector(pattern)}):
            return realtime_signals(np.ones(24), "pattern", "raw", None)

    def test_gap_shorter_than_recent_window_keeps_one_episode(self):
        # флаги ряда мо_2 при штрафе 3: перерыв в два месяца перед вторым изломом
        self.assertEqual(self.signals("XXXX..X..."), [MIN_HISTORY])
        self.assertEqual(self.signals("X" + "." * (RECENT_WINDOW - 1) + "X"), [MIN_HISTORY])

    def test_gap_of_recent_window_quiet_months_starts_a_new_episode(self):
        self.assertEqual(self.signals("X" + "." * RECENT_WINDOW + "X"),
                         [MIN_HISTORY, MIN_HISTORY + RECENT_WINDOW + 1])

    def test_first_evaluated_month_may_signal(self):
        self.assertEqual(self.signals("X"), [MIN_HISTORY])
        self.assertEqual(self.signals("..XX"), [MIN_HISTORY + 2])


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
        breaks = script.offline_breaks(panel, "pelt", "ratio", 1.0)
        self.assertEqual(list(breaks.columns), ["series_id", "month"])
        table = script.offline_shares(breaks, panel)
        self.assertEqual(list(table.columns), ["month", "share", "n_breaks"])
        self.assertEqual(table["share"].idxmax(), 14)
        self.assertEqual(table.loc[14, "month"], month(14))
        self.assertAlmostEqual(table.loc[14, "share"], table.loc[14, "n_breaks"] / panel.shape[1] * 100)

    def test_rows_are_the_breaks_of_each_series(self):
        panel = shifted_panel(shift_at=14)
        breaks = script.offline_breaks(panel, "pelt", "ratio", 1.0)
        for col in panel.columns:
            found = detect_pelt(preprocess(panel[col].to_numpy(float), "ratio"), penalty=1.0).breakpoints
            with self.subTest(col):
                self.assertEqual(breaks.loc[breaks["series_id"] == col, "month"].tolist(),
                                 [month(b + 1) for b in found])


class EdgeWarningsTest(unittest.TestCase):
    def test_warns_only_for_calibration_at_either_edge_of_the_grid(self):
        grid = np.geomspace(0.001, 100, 40)
        table = pd.DataFrame({"penalty": [1.0, 3.0, 10.0], "kernel_pen": [grid[0], grid[17], grid[-1]],
                              "fa_pelt": [100.0, 5.0, 0.0], "fa_kernel": [100.0, 5.0, 0.0]})
        lines = script.edge_warnings(table, grid)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith("ВНИМАНИЕ: калибровка на краю сетки") for line in lines))
        self.assertIn("штраф 1:", lines[0])
        self.assertIn("штраф 10:", lines[1])


class _Report:
    def as_text(self) -> str:
        return ""


CONFIG = {
    "data": {"path": "panel.parquet", "category": "Все категории", "max_gap": 2},
    "bench": {"n_series": 3, "seed": 5, "positions": [14], "magnitudes": [4.0], "kinds": ["level"],
              "modes": ["ratio"], "detectors": ["pelt", "kernel_rbf", "cusum"], "penalties": [1.0, 3.0],
              # своя сетка, не протокольная: калиброванный штраф обязан взяться именно из неё
              "kernel_pen_grid": {"min": 0.002, "max": 50, "points": 25}},
    "selection": {"rule": "J Юдена у {detector} на режиме {mode}, среди штрафов из сетки; при равенстве — меньший штраф"},
    "realtime": {"detector": "pelt", "mode": "ratio", "threshold_share": 50, "events": ["2024-03"]},
    "output": {"dir": "results"},
}
FILES = {
    "cp_bench.csv": BENCH_COLUMNS,
    "cp_summary.csv": KEY + SUMMARY_COLUMNS,
    "cp_summary_by_magnitude.csv": KEY + ["magnitude"] + SUMMARY_COLUMNS,
    "cp_calibration.csv": CALIBRATION_COLUMNS,
    "cp_realtime.csv": ["penalty", "month", "share", "n_signals", "n_failed"],
    "cp_realtime_events.csv": ["penalty", "selected", "event", "crossed_month", "delay", "max_share",
                               "share_in_window"],
    "cp_offline.csv": ["penalty", "month", "share", "n_breaks"],
    "cp_offline_series.csv": ["penalty", "series_id", "month"],
}


def run_main(root: Path, config: dict, panel: pd.DataFrame, detectors: dict | None = None) -> str:
    """`main()` во временном каталоге `root`: загрузка панели подменена, детекторы —
    настоящие, плюс `detectors`, если заданы. Возвращает напечатанное."""
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    out = io.StringIO()
    with contextlib.ExitStack() as stack:
        for name, value in {"ROOT": root, "load_panel": mock.Mock(return_value=None),
                            "build_matrix": mock.Mock(return_value=(panel, _Report()))}.items():
            stack.enter_context(mock.patch.object(script, name, value))
        stack.enter_context(mock.patch.dict(cp_bench.DETECTORS, detectors or {}))
        stack.enter_context(mock.patch.object(sys, "argv", ["changepoints.py", "--config", str(path)]))
        stack.enter_context(contextlib.redirect_stdout(out))
        script.main()
    return out.getvalue()


class MainTest(unittest.TestCase):
    """`main()` целиком на синтетике: от калибровки до записанных файлов."""

    @classmethod
    def setUpClass(cls):
        # Проверки только читают файлы и лог, поэтому main() гоняется один раз на класс.
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.log = run_main(cls.root, CONFIG, shifted_panel(8))
        cls.results = cls.root / "results"

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
        rule = CONFIG["selection"]["rule"].format(detector="pelt", mode="ratio")
        self.assertLess(self.log.index(rule), self.log.index("РЕАЛЬНЫЕ ДАННЫЕ"))

    def test_bench_uses_the_calibrated_kernel_penalty(self):
        calibration = self.read("cp_calibration.csv")
        bench = self.read("cp_bench.csv")
        kernel = bench[bench["detector"] == "kernel_rbf"]
        expected = kernel["penalty"].map(dict(zip(calibration["penalty"], calibration["kernel_pen"])))
        self.assertTrue(np.allclose(kernel["penalty_effective"], expected))

    def test_kernel_penalty_comes_from_the_configured_grid(self):
        spec = CONFIG["bench"]["kernel_pen_grid"]
        grid = np.geomspace(spec["min"], spec["max"], spec["points"])
        for pen in self.read("cp_calibration.csv")["kernel_pen"]:
            self.assertTrue(np.isclose(grid, pen, rtol=1e-12, atol=0).any(), pen)

    def test_offline_series_add_up_to_offline_counts(self):
        counts = self.read("cp_offline_series.csv").groupby(["penalty", "month"]).size()
        offline = self.read("cp_offline.csv").set_index(["penalty", "month"])["n_breaks"]
        self.assertGreater(offline.sum(), 0)
        self.assertTrue(counts.reindex(offline.index, fill_value=0).equals(offline))

    def test_realtime_and_offline_cover_every_penalty_and_month(self):
        for name in ["cp_realtime.csv", "cp_offline.csv"]:
            with self.subTest(name):
                table = self.read(name)
                self.assertEqual(len(table), 2 * 24)
                self.assertEqual(sorted(table["penalty"].unique()), [1.0, 3.0])


class DetectorFailureTest(unittest.TestCase):
    """Шаг, на котором детектор упал, — шаг без флага, но не молчание: сбой считается."""

    def test_failed_steps_are_counted_not_swallowed(self):
        y = noisy_panel(1).iloc[:, 0].to_numpy(float)
        failed: list[int] = []
        with mock.patch.dict(cp_bench.DETECTORS, {"exploding": exploding_detector}):
            self.assertIsNone(streaming_signal(y, "exploding", "ratio", failed=failed))
            self.assertEqual(failed, list(range(MIN_HISTORY, len(y))))
            failed = []
            self.assertEqual(realtime_signals(y, "exploding", "ratio", None, failed), [])
            self.assertEqual(len(failed), len(y) - MIN_HISTORY)

    def test_failures_reach_the_summary_and_the_panel_table(self):
        panel = noisy_panel(3)
        bench_args = {**BENCH, "detectors": ("pelt", "exploding", "cusum"), "kernel_penalties": None}
        with mock.patch.dict(cp_bench.DETECTORS, {"exploding": exploding_detector}):
            summary = summarise(run_bench(panel, **bench_args))
            monthly, _ = panel_realtime(panel, "exploding", "ratio", 1.0, 50, [month(14)])
        steps = 24 - MIN_HISTORY
        broken = summary[summary["detector"] == "exploding"]
        # на каждом ряду 1 чистый и 2 испорченных просмотра, каждый падает на всех шагах
        self.assertTrue((broken["n_failed"] == 3 * 3 * steps).all())
        self.assertTrue((summary.loc[summary["detector"] != "exploding", "n_failed"] == 0).all())
        self.assertEqual(monthly["n_failed"].tolist(), [0] * MIN_HISTORY + [3] * steps)
        self.assertEqual(monthly["n_signals"].sum(), 0)

    def test_real_detectors_do_not_fail_on_synthetic_series(self):
        bench = run_bench(noisy_panel(3), **{
            **BENCH, "detectors": tuple(DETECTORS), "modes": ("raw", "ratio", "deseason"),
            "kinds": ("level", "trend", "variance"),
        })
        self.assertEqual(int(bench["n_failed"].sum()), 0)

    def test_warning_line_names_the_place_and_the_count(self):
        self.assertEqual(script.failure_warning("стенд, pelt / ratio / штраф 1", 0), [])
        lines = script.failure_warning("стенд, pelt / ratio / штраф 1", 16)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("ВНИМАНИЕ: стенд, pelt / ratio / штраф 1:"))
        self.assertIn("16 шагах", lines[0])

    def test_main_prints_a_warning_for_a_failing_detector(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        config = {**CONFIG, "bench": {**CONFIG["bench"], "detectors": ["pelt", "exploding", "cusum"]}}
        # CUSUM тоже сломан — у него штрафа нет, и в строке вместо штрафа прочерк
        broken = {"exploding": exploding_detector, "cusum": exploding_detector}
        log = run_main(Path(tmp.name), config, shifted_panel(8), broken)
        summary = pd.read_csv(Path(tmp.name) / "results" / "cp_summary.csv")
        self.assertIn("ВНИМАНИЕ: стенд, exploding / ratio / штраф 1:", log)
        self.assertIn("ВНИМАНИЕ: стенд, exploding / ratio / штраф 3:", log)
        self.assertIn("ВНИМАНИЕ: стенд, cusum / ratio / штраф —:", log)
        self.assertTrue((summary.loc[summary["detector"] != "pelt", "n_failed"] > 0).all())
        self.assertTrue((summary.loc[summary["detector"] == "pelt", "n_failed"] == 0).all())
        self.assertNotIn("ВНИМАНИЕ: стенд, pelt", log)


class CalibrationFailureTest(unittest.TestCase):
    def test_failing_grid_point_is_counted_even_when_it_wins(self):
        # PELT падает на каждом шаге и тревог не даёт; ядро падает только при штрафе 2 и там
        # тоже «молчит». До PELT одинаково близки 2 и 3, берётся меньший — сломанная точка.
        sample = pd.DataFrame({f"мо_{i}": np.full(12, level) for i, level in enumerate([0.5, 1.5, 2.5, 2.5, 2.5])})

        def kernel(window, penalty=3.0):
            if penalty == 2.0:
                raise RuntimeError("ядро сломано при штрафе 2")
            return threshold_detector(window, penalty)

        with mock.patch.dict(cp_bench.DETECTORS, {"pelt": exploding_detector, "kernel_rbf": kernel}):
            table = cp_bench.calibrate_kernel_penalty(sample, (1.0,), "raw", grid=[1.0, 2.0, 3.0])
        failed_steps = 5 * (12 - MIN_HISTORY)  # все шаги всех пяти рядов
        row = table.iloc[0]
        self.assertEqual(row["kernel_pen"], 2.0)
        self.assertEqual((row["n_failed_pelt"], row["n_failed_kernel"], row["n_failed_grid"]),
                         (failed_steps, failed_steps, failed_steps))

    def test_main_prints_calibration_and_real_data_warnings(self):
        # PELT падает на коротких окнах — в калибровке, на стенде и в потоке по панели;
        # ядро падает в одной точке сетки, которую калибровка может и не выбрать.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        edge = CONFIG["bench"]["kernel_pen_grid"]["min"]

        def kernel(window, penalty=3.0):
            if np.isclose(penalty, edge):
                raise RuntimeError("ядро сломано в одной точке сетки")
            return detect_kernel(window, penalty=penalty)

        log = run_main(Path(tmp.name), CONFIG, shifted_panel(8), {"pelt": flaky_pelt, "kernel_rbf": kernel})
        calibration = pd.read_csv(Path(tmp.name) / "results" / "cp_calibration.csv")
        self.assertTrue((calibration["n_failed_pelt"] > 0).all())
        self.assertTrue((calibration["n_failed_grid"] > 0).all())
        for line in ["ВНИМАНИЕ: калибровка, PELT при штрафе 1:", "ВНИМАНИЕ: калибровка, PELT при штрафе 3:",
                     "ВНИМАНИЕ: калибровка, ядро на сетке штрафа:", "ВНИМАНИЕ: стенд, pelt / ratio / штраф 1:",
                     "ВНИМАНИЕ: реальные данные, pelt / ratio / штраф 1:",
                     "ВНИМАНИЕ: реальные данные, pelt / ratio / штраф 3:"]:
            with self.subTest(line):
                self.assertIn(line, log)
        realtime = pd.read_csv(Path(tmp.name) / "results" / "cp_realtime.csv")
        self.assertEqual(realtime.groupby("penalty")["n_failed"].sum().tolist(), [8 * (23 - MIN_HISTORY)] * 2)


class EmptyInputTest(unittest.TestCase):
    def test_bench_sample_needs_at_least_one_series(self):
        with self.assertRaises(ValueError):
            cp_bench.bench_sample(noisy_panel(3), 0, seed=1)

    def test_panel_without_series_is_rejected(self):
        empty = noisy_panel(3).iloc[:, :0]
        for events in ([month(14)], []):
            with self.subTest(events=events), self.assertRaises(ValueError):
                panel_realtime(empty, "pelt", "ratio", 1.0, 50, events)

    def test_calibration_needs_series_and_a_grid(self):
        with self.assertRaises(ValueError):
            cp_bench.calibrate_kernel_penalty(noisy_panel(3).iloc[:, :0], (1.0,), "ratio", grid=[1.0])
        with self.assertRaises(ValueError):
            cp_bench.calibrate_kernel_penalty(noisy_panel(3), (1.0,), "ratio", grid=[])


if __name__ == "__main__":
    unittest.main()
