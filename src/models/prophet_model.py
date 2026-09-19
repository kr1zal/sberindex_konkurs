"""Prophet — базовая модель, которую по условиям конкурса требуется превзойти.

Импорт ленивый: остальной каркас должен считаться и там, где prophet
со своим cmdstanpy не собрался.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

for _noisy in ("prophet", "cmdstanpy", "cmdstanpy.utils"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


class ProphetModel:
    name = "prophet"

    def __init__(self, freq: str = "MS", **kwargs) -> None:
        self.freq = freq
        self.kwargs = {
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            **kwargs,
        }
        self._index: pd.DatetimeIndex | None = None
        self._model = None

    def set_index(self, index: pd.DatetimeIndex) -> "ProphetModel":
        """Prophet опирается на календарь, поэтому получает реальные даты, а не позиции."""
        self._index = pd.DatetimeIndex(index)
        return self

    def fit(self, y: np.ndarray) -> "ProphetModel":
        from prophet import Prophet

        y = np.asarray(y, dtype=float)
        if self._index is None or len(self._index) < len(y):
            ds = pd.date_range("2023-01-01", periods=len(y), freq=self.freq)
        else:
            ds = self._index[: len(y)]

        self._model = Prophet(**self.kwargs)
        self._model.fit(pd.DataFrame({"ds": ds, "y": y}))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict вызван до fit")
        future = self._model.make_future_dataframe(periods=horizon, freq=self.freq)
        forecast = self._model.predict(future)
        return forecast["yhat"].to_numpy()[-horizon:]
