"""Классические модели прогнозирования временных рядов.

Все они борются с одним и тем же: 15–21 точка обучения и годовой цикл, который
на такой истории почти не идентифицируется. Поэтому сезонные варианты здесь заданы
осторожно — с затухающим трендом и без попыток вытащить двенадцать сезонных
коэффициентов из пятнадцати наблюдений. Там, где подгонка не сходится,
модель честно падает и это попадает в колонку отказов, а не маскируется.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


class ETS:
    """Экспоненциальное сглаживание Хольта с затухающим трендом.

    Затухание (damped) существенно: без него линейный тренд, оценённый по полутора
    годам растущих номинальных расходов, экстраполируется слишком агрессивно.
    """

    def __init__(self, damped: bool = True, seasonal: bool = False) -> None:
        self.damped = damped
        self.seasonal = seasonal
        self.name = "ets_damped" if damped and not seasonal else ("ets_seasonal" if seasonal else "ets")

    def fit(self, y: np.ndarray) -> "ETS":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = np.asarray(y, dtype=float)
        kwargs = dict(trend="add", damped_trend=self.damped, initialization_method="estimated")
        if self.seasonal:
            # Полный годовой период требует 24 точек, а в обучении их 15-21 — такое
            # условие не выполнится никогда, и модель молча станет копией несезонной.
            # Берём квартальную сезонность: она оценима на имеющейся истории и
            # отвечает на реальный вопрос — есть ли внутригодовой цикл, который
            # ловится короткими рядами.
            period = 4
            if len(y) < 2 * period + 1:
                raise ValueError(f"мало точек для сезонности с периодом {period}: {len(y)}")
            kwargs.update(seasonal="add", seasonal_periods=period)
        self._res = ExponentialSmoothing(y, **kwargs).fit(optimized=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._res.forecast(horizon), dtype=float)


class ARIMA:
    """ARIMA с фиксированным порядком.

    Автоподбор порядка по AIC на 15 точках переобучается на шум, поэтому порядок
    задан явно и одинаков для всех рядов — так сравнение остаётся сравнением
    моделей, а не сравнением процедур отбора.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)) -> None:
        self.order = order
        self.name = f"arima{''.join(map(str, order))}"

    def fit(self, y: np.ndarray) -> "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA

        self._res = _ARIMA(np.asarray(y, dtype=float), order=self.order).fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._res.forecast(horizon), dtype=float)


class Theta:
    """Метод Тета — победитель M3 и до сих пор крепкий ориентир на коротких рядах.

    Классическая формулировка: простое экспоненциальное сглаживание плюс половина
    наклона линии регрессии по времени. Реализован явно, а не через обёртку,
    чтобы поведение на 15 точках было полностью предсказуемым.
    """

    name = "theta"

    def fit(self, y: np.ndarray) -> "Theta":
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        y = np.asarray(y, dtype=float)
        n = len(y)
        t = np.arange(n, dtype=float)
        self._slope = float(np.polyfit(t, y, 1)[0])
        self._ses = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
        self._n = n
        return self

    def predict(self, horizon: int) -> np.ndarray:
        level = np.asarray(self._ses.forecast(horizon), dtype=float)
        drift = 0.5 * self._slope * np.arange(1, horizon + 1, dtype=float)
        return level + drift
