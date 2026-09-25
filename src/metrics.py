"""Метрики качества прогноза.

MAE обязательна по условиям конкурса.
R² идёт рядом как опциональная. MASE добавлена как масштабно-независимая —
уровни расходов по МО различаются на порядок, и средний по панели MAE
без неё перекошен в сторону крупных муниципалитетов.

R² отдельного ряда считается по трём тестовым точкам вокруг их же среднего:
ряд с почти ровным тестом даёт R² в минус тысячи, и среднее по рядам
переставляет модели относительно MAE. Поэтому сводки дают R² по объединённому
пулу тестовых точек и медиану R² по рядам. Пул собирается из сумм, которые
каждая строка ряд × фолд хранит рядом с метриками: сводке не нужны сами ряды.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Метрики строки ряд × фолд в порядке файла результатов. sse, n, y_sum, y_sq —
# слагаемые R² пула; sst — SST ряда вокруг среднего его теста, из него r2 строки.
ROW_METRICS = ("mae", "r2", "smape", "mase", "sse", "sst", "n", "y_sum", "y_sq")
POOL_PARTS = ("sse", "n", "y_sum", "y_sq")

R2_POOL = "R² пул"
R2_MEDIAN = "R² медиана"
# Выигрыш по MAE в процентах от MAE эталона: к эталону конкурса и к наивной модели.
GAINS = {"к Prophet, %": "prophet", "к наивной, %": "naive_last"}
# Колонки сводок и их порядок — одни на src/run.py и scripts/horizons.py.
SUMMARY_COLUMNS = ["MAE", "MASE", "sMAPE", R2_POOL, R2_MEDIAN, *GAINS, "серий", "отказов"]


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


def target_sums(y_true: np.ndarray) -> dict[str, float]:
    """Суммы по фактическим значениям теста: SST вокруг среднего окна (как в `r2`),
    число точек, Σy и Σy². От прогноза они не зависят, поэтому для строк, посчитанных
    до их появления, восстанавливаются по панели той же функцией (scripts/backfill_r2.py)."""
    y = np.asarray(y_true, dtype=float)
    return {
        "sst": float(np.sum((y - np.mean(y)) ** 2)),
        "n": int(y.size),
        "y_sum": float(np.sum(y)),
        "y_sq": float(np.sum(y ** 2)),
    }


def row_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    """Все метрики строки ряд × фолд в порядке `ROW_METRICS`. Одна функция на модели
    по рядам и панельные в src/run.py и на scripts/horizons.py, чтобы набор колонок
    у строк не расходился."""
    return {
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train, season=1),
        "sse": float(np.sum((y_true - y_pred) ** 2)),
        **target_sums(y_true),
    }


def r2_and_gains(ok: pd.DataFrame, summary_mae: pd.Series) -> pd.DataFrame:
    """Колонки сводки `R² пул`, `R² медиана`, `к Prophet, %`, `к наивной, %` — одна функция
    на src/run.py и scripts/horizons.py.

    `ok` — успешные строки ряд × фолд, `summary_mae` — колонка MAE сводки: её индекс — ключи
    группировки, один из уровней — `model`. Возвращает кадр с тем же индексом.

    R² пул группы — 1 − Σsse / (Σy_sq − (Σy_sum)² / Σn): один R² по объединённым тестовым
    точкам всех рядов и фолдов группы, SST вокруг среднего пула. Строка без `sse` не входит
    ни в одну сумму — её точек в пуле нет. R² медиана — медиана `r2` строк, NaN не в счёт.
    В кадре без колонок-слагаемых (файл посчитан до них) обе R²-колонки — NaN.

    Выигрыш — 100 · (MAE эталона − MAE) / MAE эталона, плюс — лучше эталона. Эталон берётся
    из строки сводки с теми же ключами, кроме модели: в сводке run.py — из всей сводки,
    в сводке горизонтов — из того же горизонта, в пофолдовой — из того же горизонта и фолда.
    Эталона в группе нет — NaN.
    """
    keys = list(summary_mae.index.names)
    out = pd.DataFrame(np.nan, index=summary_mae.index, columns=[R2_POOL, R2_MEDIAN])
    if set(POOL_PARTS) <= set(ok.columns):
        pooled = ok.loc[ok["sse"].notna()]
        sums = pooled.groupby(keys)[list(POOL_PARTS)].sum()
        sst = sums["y_sq"] - sums["y_sum"] ** 2 / sums["n"]
        # Все точки пула равны — R² не определён, как у ряда с ровным тестом в `r2`.
        out[R2_POOL] = (1.0 - sums["sse"] / sst).where(sst > 0).reindex(summary_mae.index)
        if "r2" in ok.columns:
            out[R2_MEDIAN] = ok.groupby(keys)["r2"].median().reindex(summary_mae.index)

    table = summary_mae.rename("MAE").reset_index()
    within = [key for key in keys if key != "model"]
    for column, reference in GAINS.items():
        own = table["MAE"].where(table["model"] == reference)
        # MAE эталона в каждой строке его группы; в группе без эталона — NaN.
        if within:
            base = own.groupby([table[key] for key in within]).transform("max")
        else:
            base = pd.Series(own.max(), index=table.index)
        out[column] = (100.0 * (base - table["MAE"]) / base).to_numpy()
    return out
