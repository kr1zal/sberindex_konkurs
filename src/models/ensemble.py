"""Ансамбль глобальной модели и фундаментальной.

По отдельности Chronos проигрывает даже наивной модели: он инвариантен к масштабу
и не воспроизводит инфляционный рост в 15-17% годовых. Но корреляция его ошибок
с ошибками бустинга всего 0,47 — модели промахиваются в разных местах, и смесь
оказывается точнее любой из них.

**Вес подбирается только на прошлом.** Оптимальный вес, выбранный по тестовой выборке,
дал бы 1 349 против 1 492 — но это подгонка под ответ, и на новых данных так не будет.
Поэтому вес оценивается на ранних фолдах и применяется к поздним, ни разу не заглядывая
в то, на чём считается итоговая метрика.
"""
from __future__ import annotations

import numpy as np

WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def fit_weight(actual: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """Вес, минимизирующий MAE смеси на предъявленных данных."""
    errors = [np.abs(actual - (w * first + (1 - w) * second)).mean() for w in WEIGHT_GRID]
    return float(WEIGHT_GRID[int(np.argmin(errors))])


def blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    return weight * first + (1.0 - weight) * second
