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


# Короче одного полного года обучение не бывает: сезонный цикл должен встретиться
# хотя бы раз. Раньше нижняя граница была привязана к горизонту (horizon + 1),
# и на горизонте 12 она отсекала единственный возможный фолд — обучение 12,
# тест 13..24. Требование «обучение длиннее горизонта» само по себе ничего
# не гарантирует: моделям с лагами нужны примеры, а не месяцы, и у каждой свои.
MIN_TRAIN = 12


def rolling_origin(
    n_obs: int, horizon: int, n_folds: int, min_train: int = MIN_TRAIN
) -> list[Fold]:
    """Расширяющееся окно: origin сдвигается вперёд на horizon, обучение растёт.

    На 24 месячных точках единственный holdout даёт 3 тестовых значения на ряд —
    это шум, а не оценка. Скользящий origin даёт n_folds × horizon точек и является
    стандартной практикой backtesting'а временных рядов.

    ``min_train`` — нижняя граница длины обучения в первом фолде. Она не зависит
    от горизонта: на горизонте 12 из 24 точек получается ровно один фолд
    с обучением 12, и это законная постановка, а не ошибка конфигурации.
    """
    if horizon < 1:
        raise ValueError("horizon должен быть >= 1")
    if n_folds < 1:
        raise ValueError("n_folds должен быть >= 1")

    first_train = n_obs - n_folds * horizon
    if first_train < min_train:
        raise ValueError(
            f"не хватает истории: {n_obs} точек на {n_folds} фолдов по {horizon} "
            f"оставляют {first_train} точек обучения в первом фолде, нужно не меньше {min_train}"
        )

    folds = []
    for i in range(n_folds):
        train_end = first_train + i * horizon
        folds.append(
            Fold(index=i, train_end=train_end, test_start=train_end, test_end=train_end + horizon)
        )
    return folds
