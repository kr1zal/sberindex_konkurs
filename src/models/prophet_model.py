"""Prophet — базовая модель, которую по условиям конкурса требуется превзойти.

**Настройки по умолчанию здесь — настоящие настройки Prophet.** Раньше обёртка
прописывала `yearly_seasonality=True` своим дефолтом, и мы принимали за
каноническое поведение то, которое задали сами. На самом деле дефолт Prophet —
`'auto'`, и эта логика **сама выключает годовую сезонность**, когда истории
меньше 730 дней. На наших окнах обучения (425, 517 и 609 дней) она выключается
на всех трёх фолдах, и голый `Prophet()` даёт ровно тот же прогноз, что явный
`yearly_seasonality=False`: проверено, максимум расхождения по двадцати рядам
0,000000. Более того, на пятнадцати месяцах Prophet не включает вообще ни одной
сезонности.

Поэтому вариант `prophet` — это Prophet как есть, а катастрофа с отрицательным
потреблением живёт в `prophet_forced_yearly`, где сезонность включена насильно.

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
        # Ничего не навязываем: 'auto' — собственный дефолт Prophet, и он же
        # единственный честный эталон. uncertainty_samples=0 не трогает yhat,
        # а только не считает доверительный интервал, который нам не нужен:
        # прогноз с ним и без него совпадает до последнего знака.
        self.kwargs = {
            "yearly_seasonality": "auto",
            "weekly_seasonality": "auto",
            "daily_seasonality": "auto",
            "uncertainty_samples": 0,
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
