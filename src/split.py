"""Разбиение на обучение и тест. Одно на все модели — иначе сравнение ничего не значит."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: int   # число наблюдений в обучении
    test_start: int
    test_end: int

    @property
    def horizon(self) -> int:
        return self.test_end - self.test_start


def rolling_origin(n_obs: int, horizon: int, n_folds: int) -> list[Fold]:
    """Расширяющееся окно: origin сдвигается вперёд на horizon, обучение растёт.

    На 24 месячных точках единственный holdout даёт 3 тестовых значения на ряд —
    это шум, а не оценка. Скользящий origin даёт n_folds × horizon точек и является
    стандартной практикой backtesting'а временных рядов.
    """
    if horizon < 1:
        raise ValueError("horizon должен быть >= 1")
    if n_folds < 1:
        raise ValueError("n_folds должен быть >= 1")

    min_train = n_obs - n_folds * horizon
    if min_train < horizon + 1:
        raise ValueError(
            f"не хватает истории: {n_obs} точек на {n_folds} фолдов по {horizon} "
            f"оставляют {min_train} точек обучения в первом фолде"
        )

    folds = []
    for i in range(n_folds):
        train_end = min_train + i * horizon
        folds.append(
            Fold(index=i, train_end=train_end, test_start=train_end, test_end=train_end + horizon)
        )
    return folds
