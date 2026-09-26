"""Отличается ли новостной фон в месяцы массовых структурных изменений.

Первая конструкция новостных признаков — месячная интенсивность и доли тем
по региону — дала эффект на прогноз +4,7 ₽ при интервале [−3,9; +13,2], то есть
ноль. Но искали её вслепую: подавали признаки в модель и смотрели на метрику.

Здесь проверка прямая и на уже найденных событиях. Детектор разладок нашёл три
месяца массового согласия — октябрь 2023, январь 2024, октябрь 2024, — когда
структурное изменение обнаруживается у половины и более муниципалитетов сразу.
Вопрос: отличается ли новостной фон в эти месяцы от прочих.

## Две проверки, потому что они отвечают на разные вопросы

**Общероссийская.** Три месяца против двадцати одного, по среднему по всем
регионам. Проверяется точным перестановочным тестом: троек из 24 месяцев ровно
2024 штуки, их можно перебрать все и получить точное p-значение, а не приближение
из теории, которая на трёх наблюдениях не работает.

**Региональная, она же главная.** Общероссийская проверка не отличает «новости
предсказывают разладку» от «в этот месяц и новостей много, и разладок много,
потому что это декабрь». Поэтому вторая проверка идёт по панели регион × месяц
с поправкой на месяц: сравниваются регионы **внутри одного месяца**. Вопрос
становится таким: если в регионе новостной фон необычен для этого месяца, чаще
ли там ломаются ряды, чем у соседей в тот же месяц. Общероссийское движение
такая постановка убирает целиком.

## Множественность

Проверяется семь признаков на трёх сдвигах — двадцать одна гипотеза. При таком
числе проверок «значимый на уровне 0,05» результат появляется почти наверняка
и сам по себе ничего не значит: мы уже один раз получили такой и он не пережил
поправку. Поэтому порог Бонферрони 0,05/21 заявлен заранее, до расчёта.

    .venv/bin/python -u scripts/news_event_study.py
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.changepoints import detect_pelt  # noqa: E402
from src.cp_bench import preprocess  # noqa: E402
from src.data import build_matrix, load_panel  # noqa: E402
from src.regions import attach_regions, load_dictionary  # noqa: E402

PENALTY = 1.0  # штраф офлайновой картины (три месяца массового согласия по полному ряду); стенд выбирает 3,0
LAGS = (0, 1, 2)  # новости месяца t против разладок в t, t+1, t+2
FEATURES = ["intensity", "t_ceny", "t_dohody", "t_zanjatost", "t_proizvodstvo", "t_kredit", "t_torgovlja"]
ALPHA = 0.05


def break_flags(wide: pd.DataFrame) -> np.ndarray:
    """Матрица (ряды × месяцы): единица там, где детектор нашёл структурное изменение."""
    flags = np.zeros((wide.shape[1], wide.shape[0]), dtype=float)
    for i, col in enumerate(wide.columns):
        series = preprocess(wide[col].to_numpy(dtype=float), "ratio")
        for point in detect_pelt(series, penalty=PENALTY).breakpoints:
            # preprocess отрезает первую точку, поэтому индекс сдвигается на единицу
            if 0 <= point + 1 < wide.shape[0]:
                flags[i, point + 1] = 1.0
    return flags


def exact_permutation_p(values: np.ndarray, event_positions: tuple[int, ...]) -> float:
    """Точное p: доля троек месяцев, где разрыв средних не меньше наблюдённого.

    Перебираются все C(24,3) = 2024 тройки, поэтому приближать нечем — это
    и есть всё распределение статистики при нулевой гипотезе.
    """
    n = len(values)
    observed = abs(values[list(event_positions)].mean() - np.delete(values, list(event_positions)).mean())
    extreme = 0
    total = 0
    for triple in combinations(range(n), len(event_positions)):
        rest = np.delete(values, list(triple))
        statistic = abs(values[list(triple)].mean() - rest.mean())
        extreme += statistic >= observed - 1e-12
        total += 1
    return extreme / total


def within_month_effect(
    table: pd.DataFrame, feature: str, lag: int, n_permutations: int = 2000, seed: int = 20260920
) -> tuple[float, float]:
    """Связь новостей с разладками внутри месяца. Возвращает наклон и p.

    И признак, и отклик центрируются по месяцу, то есть из обоих вычитается
    среднее по всем регионам этого месяца. После этого общероссийская динамика
    в данных отсутствует, и остаётся только то, чем регионы различаются между
    собой в один и тот же месяц.

    Значимость — перестановочная: внутри каждого месяца регионы перемешиваются,
    поэтому сохраняется и структура месяцев, и распределение обеих величин.
    """
    frame = table.dropna(subset=[feature, f"share_lag{lag}"]).copy()
    frame["x"] = frame[feature] - frame.groupby("month")[feature].transform("mean")
    frame["y"] = frame[f"share_lag{lag}"] - frame.groupby("month")[f"share_lag{lag}"].transform("mean")
    x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
    denominator = float((x**2).sum())
    if denominator <= 0:
        return np.nan, np.nan
    observed = float((x * y).sum() / denominator)

    rng = np.random.default_rng(seed)
    months = frame["month"].to_numpy()
    groups = [np.flatnonzero(months == m) for m in np.unique(months)]
    extreme = 0
    for _ in range(n_permutations):
        shuffled = y.copy()
        for members in groups:
            shuffled[members] = rng.permutation(y[members])
        extreme += abs(float((x * shuffled).sum() / denominator)) >= abs(observed)
    return observed, (extreme + 1) / (n_permutations + 1)


def main() -> int:
    warnings.filterwarnings("ignore")
    panel = load_panel(ROOT / "data/raw/mo_spending_2023-2024.parquet")
    wide, _ = build_matrix(panel, "Все категории", max_gap=2)
    months = pd.to_datetime(wide.index).to_period("M")

    print("поиск структурных изменений по всей панели...")
    flags = break_flags(wide)
    national = flags.mean(axis=0)
    events = np.argsort(national)[-3:]
    events = tuple(sorted(int(e) for e in events))
    print(f"рядов {wide.shape[1]}, штраф {PENALTY}")
    print("месяцы массового согласия: " + ", ".join(
        f"{months[e]} ({national[e] * 100:.1f}%)" for e in events
    ))
    print(f"в остальные месяцы не выше {np.delete(national, list(events)).max() * 100:.1f}%\n")

    features = pd.read_parquet(ROOT / "data/news/monthly.parquet")
    features["month"] = pd.to_datetime(features["month"]).dt.to_period("M")
    print(f"новости: {features['region_name'].nunique()} регионов, {features['month'].nunique()} месяцев")

    n_tests = len(FEATURES) * len(LAGS)
    threshold = ALPHA / n_tests
    print(f"проверок {n_tests}, порог Бонферрони {threshold:.4f} (объявлен до расчёта)\n")

    # --- проверка 1: общероссийская, три месяца против двадцати одного ---
    print("ПРОВЕРКА 1. Общероссийский фон в месяцы согласия против прочих")
    national_features = features.groupby("month")[FEATURES].mean().reindex(months)
    print(f"{'признак':16s} {'сдвиг':>6s} {'в события':>11s} {'в прочие':>10s} {'разрыв':>9s} {'точное p':>10s} {'прошёл':>7s}")
    rows = []
    for feature in FEATURES:
        series = national_features[feature].to_numpy(dtype=float)
        for lag in LAGS:
            shifted = tuple(e - lag for e in events)
            if min(shifted) < 0:
                continue
            p = exact_permutation_p(series, shifted)
            inside = series[list(shifted)].mean()
            outside = np.delete(series, list(shifted)).mean()
            rows.append((feature, lag, p))
            print(f"{feature:16s} {lag:>6d} {inside:11.4f} {outside:10.4f} "
                  f"{inside - outside:+9.4f} {p:10.4f} {'да' if p < threshold else 'нет':>7s}")

    survived = [r for r in rows if r[2] < threshold]
    print(f"\nпрошло поправку: {len(survived)} из {len(rows)}")
    if not survived:
        best = min(rows, key=lambda r: r[2])
        print(f"лучшее p = {best[2]:.4f} у {best[0]} со сдвигом {best[1]} — "
              f"порог {threshold:.4f} не пройден")

    # --- проверка 2: внутри месяца, между регионами ---
    print("\n\nПРОВЕРКА 2. Между регионами внутри одного месяца (главная)")
    entities = pd.DataFrame(
        {"series_id": wide.columns, "mo": [str(c).split(" #")[0] for c in wide.columns]}
    )
    joined, report = attach_regions(entities, load_dictionary())
    region_of = {
        row.series_id: row.region_name
        for row in joined.itertuples()
        if isinstance(row.region_name, str) and row.region_name
    }
    positions: dict[str, list[int]] = {}
    for i, col in enumerate(wide.columns):
        region = region_of.get(col)
        if region:
            positions.setdefault(region, []).append(i)

    share_rows = []
    for region, members in positions.items():
        region_share = flags[members].mean(axis=0)
        for j, month in enumerate(months):
            share_rows.append({"region_name": region, "month": month, "share": region_share[j]})
    shares = pd.DataFrame(share_rows)

    table = features.merge(shares, on=["region_name", "month"], how="inner")
    for lag in LAGS:
        future = shares.copy()
        future["month"] = future["month"] - lag  # новости месяца t против разладок t+lag
        table = table.merge(
            future.rename(columns={"share": f"share_lag{lag}"}),
            on=["region_name", "month"], how="left",
        )
    matched = table["region_name"].nunique()
    print(f"регионов с новостями и панелью: {matched}, строк регион-месяц: {len(table)}")
    print(f"привязка МО к регионам: {report.n_resolved} из {report.n_series}\n")

    print(f"{'признак':16s} {'сдвиг':>6s} {'наклон':>11s} {'p (перестановки)':>18s} {'прошёл':>7s}")
    survived2 = []
    for feature in FEATURES:
        for lag in LAGS:
            slope, p = within_month_effect(table, feature, lag)
            if not np.isfinite(slope):
                continue
            passed = p < threshold
            survived2.append((feature, lag, p, slope, passed))
            print(f"{feature:16s} {lag:>6d} {slope:+11.4f} {p:18.4f} {'да' if passed else 'нет':>7s}")

    kept = [r for r in survived2 if r[4]]
    print(f"\nпрошло поправку: {len(kept)} из {len(survived2)}")
    if kept:
        for feature, lag, p, slope, _ in kept:
            print(f"  {feature}, сдвиг {lag}: наклон {slope:+.4f}, p = {p:.4f}")
    else:
        best = min(survived2, key=lambda r: r[2])
        print(f"лучшее p = {best[2]:.4f} у {best[0]} со сдвигом {best[1]}; "
              f"без поправки это выглядело бы находкой, с поправкой — нет")

    out = ROOT / "results" / "news_event_study.csv"
    pd.DataFrame(survived2, columns=["признак", "сдвиг", "p", "наклон", "прошёл"]).to_csv(out, index=False)
    print(f"\nсохранено: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
