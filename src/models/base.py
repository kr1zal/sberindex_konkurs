"""Единый интерфейс модели. Всё, что появится дальше, подключается через него."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Forecaster(Protocol):
    name: str

    def fit(self, y: np.ndarray) -> "Forecaster": ...

    def predict(self, horizon: int) -> np.ndarray: ...
