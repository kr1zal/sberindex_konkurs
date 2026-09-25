"""Дообучение Chronos и сид Moirai: `src/models/foundation.py` без весов моделей.

Веса не грузятся. Разбиение на обучение и проверку считается на синтетических
origin, конвейер Chronos подменён пустым, сэмплирование Moirai — шумом глобального
генератора torch: проверяется только, откуда берётся случайность и какие месяцы
видит ранняя остановка.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import foundation  # noqa: E402
from src.models.foundation import ChronosPanel, Moirai  # noqa: E402


def matrix(n_series: int, n_months: int = 24) -> pd.DataFrame:
    """Месяцы × ряды с трендом и колебанием, без пропусков."""
    t = np.arange(n_months, dtype=float)
    return pd.DataFrame({f"мо_{i}": 100 + (i + 1) * t + 5 * np.sin(t) for i in range(n_series)})


def origins_like_samples(n_series: int, train_end: int, horizon: int) -> np.ndarray:
    """Origin окон дообучения в раскладке `ChronosPanel._samples`: у каждого ряда
    t = horizon … train_end − horizon − 1, ряды друг за другом."""
    return np.tile(np.arange(horizon, train_end - horizon), n_series)


def target_months(origins: np.ndarray, horizon: int) -> set[int]:
    """Месяцы цели окон: у окна с origin t — t+1 … t+horizon."""
    return {int(t) + step for t in origins for step in range(1, horizon + 1)}


class ValidationSplitTest(unittest.TestCase):
    """Ранняя остановка должна смотреть на месяцы, которых обучение не видело:
    иначе валидация по времени перестаёт быть валидацией по времени."""

    def test_training_and_validation_targets_share_no_month(self):
        for horizon in (1, 3, 6):
            for train_end in (21, 24):
                with self.subTest(horizon=horizon, train_end=train_end):
                    origins = origins_like_samples(3, train_end, horizon)
                    train_idx, valid_idx = ChronosPanel._validation_split(origins, horizon)
                    self.assertGreater(len(train_idx), 0)
                    self.assertGreater(len(valid_idx), 0)
                    train = target_months(origins[train_idx], horizon)
                    valid = target_months(origins[valid_idx], horizon)
                    self.assertEqual(train & valid, set())
                    self.assertLess(max(train | valid), train_end)  # всё внутри обучения фолда

    def test_validation_is_the_last_window_of_each_series_and_training_is_not_cut(self):
        origins = origins_like_samples(3, 21, 3)
        train_idx, valid_idx = ChronosPanel._validation_split(origins, 3)
        self.assertEqual(origins[valid_idx].tolist(), [origins.max()] * 3)
        # Обучающие окна те же, что при прежнем правиле: цели кончаются на последнем
        # месяце перед первым проверочным.
        self.assertEqual(origins[train_idx].max(), origins.max() - 3)


class _Model:
    def eval(self):
        return self


class _Pipeline:
    """Конвейер без весов: `fit` копирует его и переводит модель в режим оценки."""

    def __init__(self):
        self.model = _Model()


class FitSplitTest(unittest.TestCase):
    def test_fit_passes_disjoint_months_to_training(self):
        wide = matrix(4)
        seen = []

        def train_one(model, context, target, train_idx, valid_idx, learning_rate):
            seen.append((train_idx, valid_idx))
            return None, 0.5, 0

        with mock.patch.object(foundation, "_pipeline", return_value=_Pipeline()), \
                mock.patch.object(ChronosPanel, "_train_one", train_one):
            model = ChronosPanel(finetune=True).fit(wide, 21, 3)
        origins = model._samples(wide, 21, 3)[2]
        self.assertEqual(len(seen), len(model.learning_rates))
        for train_idx, valid_idx in seen:
            self.assertEqual(target_months(origins[train_idx], 3) & target_months(origins[valid_idx], 3),
                             set())
            self.assertEqual(len(valid_idx), wide.shape[1])  # одно окно на ряд


class SamplesTest(unittest.TestCase):
    def test_training_of_exactly_two_horizons_is_refused_with_a_true_reason(self):
        # 12 = 2 · 6: прежний текст «короче двух горизонтов» при равенстве был неправдой.
        with self.assertRaises(ValueError) as caught:
            ChronosPanel()._samples(matrix(3), 12, 6)
        self.assertEqual(
            str(caught.exception),
            "обучение 12 мес не длиннее двух горизонтов по 6: "
            "окно контекста и окно цели не умещаются вместе",
        )

    def test_one_month_more_gives_one_window_per_series(self):
        context, target, origins = ChronosPanel()._samples(matrix(3), 13, 6)
        self.assertEqual(origins.tolist(), [6, 6, 6])
        self.assertEqual(tuple(context.shape), (3, 7))
        self.assertEqual(tuple(target.shape), (3, 6))


class _NoisyForecast:
    """`MoiraiForecast` без весов: выборка — последний уровень контекста плюс шум
    глобального генератора torch, как у настоящего сэмплирования."""

    def __init__(self, *, prediction_length, num_samples, **_):
        self.horizon, self.num_samples = prediction_length, num_samples

    def __call__(self, past_target, past_observed_target, past_is_pad):
        level = past_target[:, -1:, :]  # (ряды, 1, 1)
        return level + torch.randn(past_target.shape[0], self.num_samples, self.horizon)


class MoiraiSeedTest(unittest.TestCase):
    NAME = "uni2ts.model.moirai"

    def setUp(self):
        # Подменяется один модуль, а не весь sys.modules: patch.dict на sys.modules
        # при откате выкинул бы и то, что torch догрузил во время теста.
        fake = types.ModuleType(self.NAME)
        fake.MoiraiForecast = _NoisyForecast
        fake.MoiraiModule = types.SimpleNamespace(from_pretrained=lambda repo: object())
        previous = sys.modules.get(self.NAME)
        sys.modules[self.NAME] = fake

        def restore():
            if previous is None:
                sys.modules.pop(self.NAME, None)
            else:
                sys.modules[self.NAME] = previous

        self.addCleanup(restore)
        patcher = mock.patch.object(Moirai, "_MODULE", None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.values = matrix(8, 15).to_numpy().T

    def forecast(self, **kwargs) -> np.ndarray:
        # Пачка меньше числа рядов: сид должен покрывать весь цикл, а не первую пачку.
        return Moirai(batch_size=3, **kwargs)._forecast(self.values, 12, 3, 8)

    def test_repeated_forecast_is_identical_to_the_bit(self):
        self.assertEqual(self.forecast().tobytes(), self.forecast().tobytes())

    def test_other_seed_gives_other_forecast(self):
        self.assertNotEqual(self.forecast(seed=1).tobytes(), self.forecast(seed=2).tobytes())

    def test_global_generator_is_left_as_it_was(self):
        # Сид Moirai не должен сдвигать случайность того, что идёт после неё в процессе.
        before = torch.get_rng_state()
        self.forecast()
        self.assertTrue(torch.equal(torch.get_rng_state(), before))


if __name__ == "__main__":
    unittest.main()
