"""Наивные ориентиры. Модель, которая их не бьёт, не заслуживает строчки в отчёте."""
from __future__ import annotations

import numpy as np


class NaiveLast:
    name = "naive_last"

    def fit(self, y: np.ndarray) -> "NaiveLast":
        self._last = float(y[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._last, dtype=float)


class SeasonalNaive:
    """Повтор значения год назад. На месячных данных с двумя циклами — сильный ориентир."""

    def __init__(self, season: int = 12) -> None:
        self.season = season
        self.name = f"seasonal_naive_{season}"

    def fit(self, y: np.ndarray) -> "SeasonalNaive":
        self._y = np.asarray(y, dtype=float)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        m, y = self.season, self._y
        if len(y) < m:
            return np.full(horizon, float(y[-1]))
        return np.array([y[-m + (i % m)] for i in range(horizon)], dtype=float)


class Drift:
    """Продолжение прямой, проведённой через первое и последнее наблюдение."""

    name = "drift"

    def fit(self, y: np.ndarray) -> "Drift":
        y = np.asarray(y, dtype=float)
        self._last = float(y[-1])
        self._slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.0
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self._last + self._slope * np.arange(1, horizon + 1, dtype=float)


class SeasonalDrift:
    """Сезонно-наивный прогноз с поправкой на линейный тренд последнего года."""

    def __init__(self, season: int = 12) -> None:
        self.season = season
        self.name = f"seasonal_drift_{season}"

    def fit(self, y: np.ndarray) -> "SeasonalDrift":
        y = np.asarray(y, dtype=float)
        self._y = y
        m = self.season

        # Год-к-году по перекрывающемуся окну: берём столько последних месяцев,
        # сколько реально можно сравнить с теми же месяцами годом раньше.
        # На 15 точках это 3 месяца, на 21 — девять. Требование len >= 2*m
        # обнуляло бы поправку на всех фолдах и превращало модель в seasonal_naive.
        k = min(m, len(y) - m)
        self._step = (
            float(np.mean(y[-k:]) - np.mean(y[-(m + k) : -m])) / m if k >= 1 else 0.0
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        m, y = self.season, self._y
        if len(y) < m:
            return np.full(horizon, float(y[-1]))
        base = np.array([y[-m + (i % m)] for i in range(horizon)], dtype=float)
        return base + self._step * np.arange(1, horizon + 1, dtype=float)
