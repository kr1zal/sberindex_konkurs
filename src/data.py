"""Загрузка муниципальной панели СберИндекса и приведение её к матрице рядов."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = {"period", "value", "mo", "category_15", "freq", "unit_measure"}


@dataclass
class PanelReport:
    """Что именно мы выбросили и почему. Нужен для раздела воспроизводимости в отчёте."""

    n_rows: int
    n_mo: int
    n_categories: int
    n_periods: int
    n_series_total: int
    n_series_complete: int
    n_series_filled: int
    n_series_dropped: int
    n_cells_missing: int

    def as_text(self) -> str:
        return (
            f"строк: {self.n_rows}\n"
            f"МО: {self.n_mo} | категорий: {self.n_categories} | периодов: {self.n_periods}\n"
            f"рядов всего: {self.n_series_total}\n"
            f"  без пропусков: {self.n_series_complete}\n"
            f"  восстановлено интерполяцией: {self.n_series_filled}\n"
            f"  отброшено: {self.n_series_dropped}\n"
            f"пропущенных ячеек в исходнике: {self.n_cells_missing}"
        )


def load_panel(path: str | Path) -> pd.DataFrame:
    """Читает parquet и возвращает длинную панель с восстановленными сущностями МО.

    Колонка ``mo`` в выгрузке — это НАЗВАНИЕ, а не идентификатор, и 49 названий
    встречаются в нескольких регионах сразу ("Михайловский муниципальный район"
    и т.п., до 8 разных МО на одно имя). Наивный pivot по имени склеивает разные
    муниципалитеты в один ряд и молча теряет остальные.

    Выгрузка при этом хранит каждое МО непрерывным блоком строк по возрастанию
    периода, поэтому сущности восстанавливаются точно: новая начинается там, где
    период перестал расти. Проверка сходится — 2190 сущностей на категорию, ровно
    столько муниципалитетов в фильтре дашборда.
    """
    df = pd.read_parquet(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"в файле нет ожидаемых колонок: {sorted(missing)}")

    df = df.loc[df["freq"] == "Месяц"].copy()
    df["period"] = pd.to_datetime(df["period"]).dt.to_period("M").dt.to_timestamp()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    prev = df.groupby(["mo", "category_15"], sort=False)["period"].shift(1)
    starts = prev.isna() | (df["period"] <= prev)
    df["entity_no"] = starts.groupby([df["mo"], df["category_15"]]).cumsum().astype(int)

    occurrences = df.groupby(["mo", "category_15"])["entity_no"].transform("max")
    df["series_id"] = np.where(
        occurrences > 1,
        df["mo"] + " #" + df["entity_no"].astype(str),
        df["mo"],
    )
    return df.sort_values(["series_id", "category_15", "period"], ignore_index=True)


def build_matrix(
    df: pd.DataFrame,
    category: str,
    max_gap: int = 2,
) -> tuple[pd.DataFrame, PanelReport]:
    """Разворачивает панель в матрицу (период × МО) для одной категории трат.

    Правило пропусков задано явно и одинаково для всех моделей:
      * внутренний разрыв длиной <= max_gap — линейная интерполяция;
      * разрыв длиннее, либо дыра на краю ряда — ряд целиком исключается из оценки.
    Молчаливого dropna здесь нет намеренно: доля и судьба каждого ряда попадают в отчёт.
    """
    sub = df.loc[df["category_15"] == category]
    if sub.empty:
        raise ValueError(f"категория не найдена: {category!r}")

    wide = sub.pivot(index="period", columns="series_id", values="value")
    wide = wide.sort_index()

    n_missing = int(wide.isna().sum().sum())
    n_total = wide.shape[1]

    complete = wide.notna().all()
    interior = wide.ffill().notna() & wide.bfill().notna()
    gap_ok = _max_run_of_nan(wide.where(interior)) <= max_gap
    edges_ok = wide.iloc[0].notna() & wide.iloc[-1].notna()

    keep = complete | (gap_ok & edges_ok)
    filled = keep & ~complete

    out = wide.loc[:, keep].interpolate(method="linear", axis=0, limit_area="inside")

    report = PanelReport(
        n_rows=len(df),
        n_mo=df["mo"].nunique(),
        n_categories=df["category_15"].nunique(),
        n_periods=wide.shape[0],
        n_series_total=n_total,
        n_series_complete=int(complete.sum()),
        n_series_filled=int(filled.sum()),
        n_series_dropped=int((~keep).sum()),
        n_cells_missing=n_missing,
    )
    return out, report


def _max_run_of_nan(wide: pd.DataFrame) -> pd.Series:
    """Длина самой длинной серии подряд идущих NaN в каждом столбце."""
    isna = wide.isna()
    block = (~isna).cumsum()
    out: dict[str, int] = {}
    for col in wide.columns:
        flags = isna[col]
        out[col] = int(flags.groupby(block[col]).sum().max()) if flags.any() else 0
    return pd.Series(out)


def sample_series(
    wide: pd.DataFrame,
    n: int | None,
    seed: int = 20260919,
) -> pd.DataFrame:
    """Отбирает подвыборку рядов, стратифицированную по среднему уровню расходов.

    Перебор всех 2118 МО × нескольких моделей × фолдов на этапе выбора модели стоит
    часы счёта и ничего не добавляет к выводу. Поэтому отбор моделей идёт на выборке,
    а финальные числа считаются на полной панели (n=None).
    """
    if n is None or n >= wide.shape[1]:
        return wide

    level = wide.mean(axis=0)
    strata = pd.qcut(level.rank(method="first"), q=10, labels=False)
    rng = np.random.default_rng(seed)

    picked: list[str] = []
    per_stratum = max(1, n // 10)
    for s in range(10):
        members = level.index[strata == s].to_numpy()
        take = min(per_stratum, len(members))
        picked.extend(rng.choice(members, size=take, replace=False).tolist())

    rest = [c for c in wide.columns if c not in set(picked)]
    if len(picked) < n and rest:
        extra = rng.choice(rest, size=min(n - len(picked), len(rest)), replace=False)
        picked.extend(extra.tolist())

    return wide.loc[:, sorted(picked)]
