"""Метрики качества прогноза.

MAE обязательна по условиям конкурса.
R² идёт рядом как опциональная. MASE добавлена как масштабно-независимая —
уровни расходов по МО различаются на порядок, и средний по панели MAE
без неё перекошен в сторону крупных муниципалитетов.
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 1) -> float:
    """MAE, нормированная на ошибку сезонно-наивного прогноза внутри обучающей выборки."""
    if len(y_train) <= season:
        return float("nan")
    scale = float(np.mean(np.abs(y_train[season:] - y_train[:-season])))
    if scale == 0.0:
        return float("nan")
    return mae(y_true, y_pred) / scale
