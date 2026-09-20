"""Глобальная модель: одна обученная модель на всю панель вместо 2190 отдельных.

Классические модели упёрлись в потолок около 1 700–2 000 MAE, и это не недостаток
конкретных алгоритмов: пятнадцать точек истории просто не содержат больше информации.
Единственный способ добавить информации — взять её у соседних рядов. Две тысячи
муниципалитетов переживают одни и те же общероссийские шоки и один и тот же
сезонный ритм, так что форма динамики у них общая, а различается масштаб.

Отсюда две конструктивные идеи:

1. **Целевая переменная — отношение, а не уровень.** Модель предсказывает
   y[t+h] / y[t], то есть во сколько раз изменятся расходы. Это ставит Магадан
   с 74 тысячами и сельский район с десятью в одинаковые условия: учимся на форме,
   а не на масштабе. Заодно снимается вопрос о завышенности абсолютного уровня
   показателя — в отношении постоянное смещение сокращается.

2. **Прямой прогноз на каждый горизонт.** Для h = 1, 2, 3 обучаются отдельные модели,
   а не одна с рекурсивной подстановкой собственных прогнозов. На горизонте в три шага
   прямой подход и точнее, и не накапливает ошибку.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Лаг 12 намеренно отсутствует. Он требует одиннадцати точек истории, и при обучении
# в 15 точек на горизонт 3 не остаётся ни одного обучающего примера. Годовой цикл на
# этих рядах всё равно не идентифицируется — это уже показано на Prophet.
LAGS = (1, 2, 3, 4, 5, 6)


def _news_row(news: dict | None, region: str | None, month_index: int) -> dict:
    """Новостные признаки региона за месяц. Отсутствие новостей — не ноль, а NaN.

    Ноль означал бы «ничего не писали», а у нас нет издания по этому региону вовсе.
    Гистограммный бустинг sklearn работает с пропусками штатно и учится обходиться
    без признака там, где его нет, — подменять их нулями значило бы врать модели.
    """
    if not news or region is None:
        return {}
    return news.get((region, month_index), {})


def _features(series: np.ndarray, t: int) -> dict | None:
    """Признаки, доступные в момент t. Всё считается только по прошлому."""
    if t < max(LAGS) - 1:
        return None
    last = series[t]
    if last <= 0:
        return None

    row = {f"ratio_lag_{lag}": series[t - lag + 1] / last for lag in LAGS if lag > 1}
    row["level_log"] = float(np.log(last))
    for window in (3, 6):
        if t + 1 >= window:
            window_slice = series[t - window + 1 : t + 1]
            row[f"mean_ratio_{window}"] = float(window_slice.mean() / last)
            row[f"std_ratio_{window}"] = float(window_slice.std() / last)
    row["month"] = (t % 12) + 1
    return row


def _build_training(
    wide: pd.DataFrame, train_end: int, horizon_step: int,
    news: dict | None = None, regions: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Все пары (ряд, момент) из обучающей части, где известен и признак, и ответ."""
    rows, targets = [], []
    values = wide.to_numpy(dtype=float).T  # ряды по строкам
    for col, series in zip(wide.columns, values):
        region = (regions or {}).get(col)
        for t in range(max(LAGS) - 1, train_end - horizon_step):
            row = _features(series, t)
            if row is None:
                continue
            row |= _news_row(news, region, t)
            target = series[t + horizon_step]
            if not np.isfinite(target) or series[t] <= 0:
                continue
            rows.append(row)
            targets.append(target / series[t])
    return pd.DataFrame(rows), np.asarray(targets, dtype=float)


class GlobalGBM:
    """Гистограммный градиентный бустинг на признаках отношений, обученный на всей панели.

    Взят sklearn, а не LightGBM, сознательно: LightGBM тянет системную библиотеку
    OpenMP, которой на чистой macOS нет, и запуск у проверяющего упирается в
    `brew install libomp`. Для воспроизводимости зависимость, требующая системного
    пакета, того не стоит. Алгоритм тот же —
    гистограммный бустинг на деревьях.
    """

    name = "global_gbm"

    def __init__(
        self, max_iter: int = 300, learning_rate: float = 0.05, max_leaf_nodes: int = 31,
        news: dict | None = None, regions: dict | None = None,
    ) -> None:
        self.news = news
        self.regions = regions
        self.params = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=20260920,
        )
        self._models: dict[int, object] = {}
        self._columns: list[str] = []

    def fit(self, wide: pd.DataFrame, train_end: int, horizon: int) -> "GlobalGBM":
        from sklearn.ensemble import HistGradientBoostingRegressor

        for step in range(1, horizon + 1):
            features, target = _build_training(wide, train_end, step, self.news, self.regions)
            if features.empty:
                raise ValueError(f"нет обучающих примеров для шага {step}")
            self._columns = list(features.columns)
            model = HistGradientBoostingRegressor(**self.params)
            model.fit(features.to_numpy(dtype=float), target)
            self._models[step] = model
        return self

    def predict(self, wide: pd.DataFrame, train_end: int, horizon: int) -> np.ndarray:
        """Прогноз для всех рядов панели. Возвращает матрицу (рядов × горизонт)."""
        values = wide.to_numpy(dtype=float).T
        t = train_end - 1
        out = np.full((values.shape[0], horizon), np.nan)

        rows, index = [], []
        for i, (col, series) in enumerate(zip(wide.columns, values)):
            row = _features(series, t)
            if row is not None:
                row |= _news_row(self.news, (self.regions or {}).get(col), t)
                rows.append(row)
                index.append(i)
        if not rows:
            return out

        features = pd.DataFrame(rows).reindex(columns=self._columns)
        base = values[index, t]
        for step in range(1, horizon + 1):
            ratio = self._models[step].predict(features.to_numpy(dtype=float))
            out[index, step - 1] = base * ratio
        return out
