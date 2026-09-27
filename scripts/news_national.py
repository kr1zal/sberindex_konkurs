"""Новости на национальном уровне: проверка на том разрешении, где живёт сигнал.

Две предыдущие конструкции проверяли **региональные** новости против
**муниципальных** рядов и дали согласованный ноль. Но разложение ошибки
показало, что межрядовая составляющая не двигается ничем — 695–849 рублей
по всем восемнадцати сочетаниям модели и фолда, — а весь выигрыш сидит
в попадании в общероссийское движение. То есть мы проверяли не на том
разрешении: искали региональный сигнал там, где региональной изменчивости
почти нет.

Здесь корпус свёрнут в один национальный ряд и проверяется против того самого
федерального агрегата, который прогнозирует первый этап двухэтапной модели.

## Две проверки

**Первая, прогнозная.** Сезонная модель агрегата оставляет остаток. Объясняют
ли его новости? Остаток честный: прогноз в каждый момент t делается только
по истории до t. Значимость — перестановочная, признак перемешивается
по месяцам.

**Вторая, событийная.** Отличается ли доля публикаций про ставку, инфляцию
и ЦБ в три месяца найденных изломов — октябрь 2023, январь 2024, октябрь 2024 —
от остальных. Троек из 24 месяцев ровно 2024, их можно перебрать все и получить
точное p-значение.

## Порог объявлен до расчёта

Восемь признаков — семь тем плюс интенсивность — на трёх горизонтах в первой
проверке и трёх сдвигах во второй. Итого **48 гипотез**, порог Бонферрони
**0,05 / 48 = 0,00104**. Точный перебор троек даёт минимально достижимое
p = 1/2024 = 0,00049, перестановочная проверка с 5 000 повторов — 0,0002,
так что порог достижим обеими.

## Известное заранее ограничение

Агрегат идёт с декабря 2018-го, корпус новостей — с января 2023-го. Сезонный
прогноз опирается на 64–70 месяцев, регрессия остатков — максимум на 24.
Это тот же закон о длине истории, только теперь он ограничивает саму проверку.

    .venv/bin/python -u scripts/news_national.py
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

from src.external import CANDIDATES, choose_model, load_aggregate  # noqa: E402

HORIZONS = (1, 2, 3)
LAGS = (0, 1, 2)
EVENTS = ("2023-10", "2024-01", "2024-10")
N_PERMUTATIONS = 5000
ALPHA = 0.05
SEED = 20260921


def permutation_slope_p(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    """Наклон регрессии y на x и перестановочное p. Перемешивается признак."""
    centred = x - x.mean()
    denominator = float((centred**2).sum())
    if denominator <= 0:
        return np.nan, np.nan
    observed = float((centred * (y - y.mean())).sum() / denominator)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(x)
        c = shuffled - shuffled.mean()
        slope = float((c * (y - y.mean())).sum() / float((c**2).sum()))
        extreme += abs(slope) >= abs(observed)
    return observed, (extreme + 1) / (N_PERMUTATIONS + 1)


def exact_event_p(values: np.ndarray, positions: tuple[int, ...]) -> float:
    """Точное p по всем тройкам месяцев: доля троек с разрывом не меньше наблюдённого."""
    observed = abs(
        values[list(positions)].mean() - np.delete(values, list(positions)).mean()
    )
    extreme = total = 0
    for triple in combinations(range(len(values)), len(positions)):
        rest = np.delete(values, list(triple))
        extreme += abs(values[list(triple)].mean() - rest.mean()) >= observed - 1e-12
        total += 1
    return extreme / total


def main() -> int:
    warnings.filterwarnings("ignore")
    news = pd.read_parquet(ROOT / "data/news/national.parquet")
    news["month"] = pd.PeriodIndex(news["month"], freq="M")
    news = news.set_index("month").sort_index()
    features = [c for c in news.columns if c.startswith("t_")] + ["intensity"]

    aggregate = load_aggregate(ROOT / "data/reference/sberindex")
    log_series = np.log(aggregate)

    n_tests = len(features) * (len(HORIZONS) + len(LAGS))
    threshold = ALPHA / n_tests
    print(f"признаков {len(features)}: {', '.join(features)}")
    print(f"корпус {len(news)} мес ({news.index.min()}..{news.index.max()}), "
          f"агрегат {len(aggregate)} мес ({aggregate.index.min()}..{aggregate.index.max()})")
    print(f"гипотез {n_tests}, порог Бонферрони {threshold:.5f} — объявлен до расчёта\n")

    # --- проверка 1: объясняют ли новости остаток сезонного прогноза агрегата ---
    print("ПРОВЕРКА 1. Остаток прогноза федерального агрегата против новостей")
    periods = list(log_series.index)
    values = log_series.to_numpy(dtype=float)
    model_name, _ = choose_model(values, max(HORIZONS))
    print(f"модель агрегата на всей истории: {model_name}")

    residuals: dict[int, dict] = {h: {} for h in HORIZONS}
    for position, period in enumerate(periods):
        if period not in news.index or position < 25:
            continue
        predicted = CANDIDATES[model_name](values[: position + 1], max(HORIZONS)) - values[position]
        for h in HORIZONS:
            if position + h < len(periods):
                residuals[h][period] = float(values[position + h] - values[position] - predicted[h - 1])

    print(f"пар с новостями: " + ", ".join(f"h={h}: {len(residuals[h])}" for h in HORIZONS))
    print(f"\n{'признак':14s} {'h':>3s} {'точек':>6s} {'наклон':>10s} {'|t|':>6s} {'p':>9s} {'прошёл':>7s}")
    rows = []
    for feature in features:
        for h in HORIZONS:
            months = sorted(residuals[h])
            y = np.array([residuals[h][m] for m in months])
            x = news.loc[months, feature].to_numpy(dtype=float)
            slope, p = permutation_slope_p(x, y)
            if not np.isfinite(slope):
                continue
            centred = x - x.mean()
            fitted = y.mean() + slope * centred
            se = np.sqrt(((y - fitted) ** 2).sum() / max(len(y) - 2, 1) / (centred**2).sum())
            rows.append({"проверка": "остаток агрегата", "признак": feature, "сдвиг": h,
                         "наклон": slope, "p": p, "прошёл": p < threshold})
            print(f"{feature:14s} {h:>3d} {len(y):>6d} {slope:10.4f} {abs(slope / se):6.2f} "
                  f"{p:9.4f} {'да' if p < threshold else 'нет':>7s}")

    # --- проверка 2: событийная ---
    print("\n\nПРОВЕРКА 2. Новостной фон в месяцы изломов против остальных")
    months = list(news.index)
    events = tuple(months.index(pd.Period(e, freq="M")) for e in EVENTS)
    print("месяцы изломов: " + ", ".join(EVENTS))
    print(f"\n{'признак':14s} {'сдвиг':>6s} {'в события':>11s} {'в прочие':>10s} {'разрыв':>10s} {'точное p':>10s} {'прошёл':>7s}")
    for feature in features:
        series = news[feature].to_numpy(dtype=float)
        for lag in LAGS:
            shifted = tuple(e - lag for e in events)
            if min(shifted) < 0:
                continue
            p = exact_event_p(series, shifted)
            inside = series[list(shifted)].mean()
            outside = np.delete(series, list(shifted)).mean()
            rows.append({"проверка": "события", "признак": feature, "сдвиг": lag,
                         "наклон": inside - outside, "p": p, "прошёл": p < threshold})
            print(f"{feature:14s} {lag:>6d} {inside:11.4f} {outside:10.4f} "
                  f"{inside - outside:+10.4f} {p:10.4f} {'да' if p < threshold else 'нет':>7s}")

    table = pd.DataFrame(rows)
    out = ROOT / "results" / "news_national.csv"
    table.to_csv(out, index=False)

    passed = table[table["прошёл"]]
    print(f"\n\nИТОГО. Гипотез {len(table)}, порог {threshold:.5f}, прошло {len(passed)}")
    if len(passed):
        print(passed.to_string(index=False))
    best = table.loc[table["p"].idxmin()]
    print(f"лучшее p = {best['p']:.4f} у {best['признак']} ({best['проверка']}, сдвиг {best['сдвиг']})")
    print(f"значимых даже без поправки, на уровне 0,05: {(table['p'] < 0.05).sum()} из {len(table)}")
    print(f"\nсохранено: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
