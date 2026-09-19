"""Прогон всех моделей по единому протоколу. Точка входа каркаса."""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_matrix, load_panel, sample_series  # noqa: E402
from src.metrics import mae, mase, r2, smape  # noqa: E402
from src.models.classical import ARIMA, ETS, Theta  # noqa: E402
from src.models.naive import Drift, NaiveLast, SeasonalDrift, SeasonalNaive  # noqa: E402
from src.split import rolling_origin  # noqa: E402

REGISTRY = {
    "naive_last": lambda: NaiveLast(),
    "seasonal_naive": lambda: SeasonalNaive(season=12),
    "drift": lambda: Drift(),
    "seasonal_drift": lambda: SeasonalDrift(season=12),
    "ets_damped": lambda: ETS(damped=True),
    "ets_seasonal": lambda: ETS(damped=True, seasonal=True),
    "theta": lambda: Theta(),
    "arima111": lambda: ARIMA(order=(1, 1, 1)),
    "arima011": lambda: ARIMA(order=(0, 1, 1)),
}

# Prophet в канонической конфигурации стоит 3.5 с на подгонку против 0.08 с без
# годовой сезонности — разница в 47 раз. На 24 точках Фурье-компоненты годового
# цикла оставляют оптимизатору почти вырожденную задачу, и это та же
# недоопределённость, о которой Prophet сам предупреждает в логе.
# Обе конфигурации считаются как отдельные модели: выключить сезонность и
# назвать результат базовой моделью значило бы подогнать эталон под себя.
PROPHET_VARIANTS = {
    "prophet": {"yearly_seasonality": True},
    "prophet_no_yearly": {"yearly_seasonality": False, "uncertainty_samples": 0},
}


def _build_model(name: str):
    if name in PROPHET_VARIANTS:
        from src.models.prophet_model import ProphetModel

        return ProphetModel(**PROPHET_VARIANTS[name])
    if name not in REGISTRY:
        raise KeyError(f"модель не зарегистрирована: {name}")
    return REGISTRY[name]()


def _score_series(task: tuple) -> list[dict]:
    """Считает одну пару ряд-модель по всем фолдам. Верхнего уровня — чтобы пиклилось."""
    model_name, col, series, index, folds, horizon = task
    out = []
    for fold in folds:
        y_train = series[: fold.train_end]
        y_test = series[fold.test_start : fold.test_end]

        model = _build_model(model_name)
        if hasattr(model, "set_index"):
            model.set_index(index)
        try:
            y_pred = model.fit(y_train).predict(horizon)
        except Exception as exc:  # одна упавшая подгонка не должна ронять прогон
            out.append(
                {"model": model_name, "fold": fold.index, "mo": col, "error": str(exc)[:120]}
            )
            continue

        out.append(
            {
                "model": model_name,
                "fold": fold.index,
                "mo": col,
                "mae": mae(y_test, y_pred),
                "r2": r2(y_test, y_pred),
                "smape": smape(y_test, y_pred),
                "mase": mase(y_test, y_pred, y_train, season=1),
                "error": None,
            }
        )
    return out


def evaluate(
    wide: pd.DataFrame, model_name: str, horizon: int, n_folds: int, workers: int = 1
) -> pd.DataFrame:
    """Гоняет одну модель по всем рядам и фолдам.

    Параллелится по рядам, а не по фолдам: ряды независимы, а фолды внутри ряда
    делят обучающую историю и дробить их смысла нет.
    """
    folds = rolling_origin(n_obs=wide.shape[0], horizon=horizon, n_folds=n_folds)
    index = wide.index
    tasks = [
        (model_name, col, wide[col].to_numpy(dtype=float), index, folds, horizon)
        for col in wide.columns
    ]

    if workers <= 1:
        rows = [row for task in tasks for row in _score_series(task)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [row for chunk in pool.map(_score_series, tasks, chunksize=8) for row in chunk]
    return pd.DataFrame(rows)


def summarise(per_series: pd.DataFrame) -> pd.DataFrame:
    """Сводка по моделям. R² считается по объединённому пулу точек, а не усреднением по рядам:
    на трёх точках горизонта R² отдельного ряда неустойчив и его среднее ничего не значит."""
    ok = per_series.loc[per_series["error"].isna()]
    grouped = ok.groupby("model")
    summary = pd.DataFrame(
        {
            "MAE": grouped["mae"].mean(),
            "MASE": grouped["mase"].mean(),
            "sMAPE": grouped["smape"].mean(),
            "серий": grouped["mo"].nunique(),
            "отказов": per_series.loc[per_series["error"].notna()].groupby("model").size(),
        }
    )
    summary["отказов"] = summary["отказов"].fillna(0).astype(int)

    _warn_identical(ok)
    return summary.sort_values("MAE")


def _warn_identical(ok: pd.DataFrame) -> None:
    """Кричит, если две модели дали совпадающие прогнозы на всех рядах.

    На рядах длиной 24 модель с порогом вида «нужно не меньше 24 точек» молча
    вырождается в соседнюю: условие никогда не выполняется, ветка не включается,
    в таблице появляются две одинаковые строки под разными именами. Поймано
    трижды — на seasonal_drift, на ets_seasonal и едва не на третьей. Глазами
    такое замечать ненадёжно, поэтому проверка живёт в каркасе.
    """
    wide = ok.pivot_table(index=["mo", "fold"], columns="model", values="mae")
    models = list(wide.columns)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pair = wide[[a, b]].dropna()
            if len(pair) and np.allclose(pair[a], pair[b], rtol=1e-9, atol=1e-9):
                print(
                    f"  ВНИМАНИЕ: {a} и {b} дали идентичные прогнозы на всех {len(pair)} парах "
                    f"ряд-фолд. Скорее всего одна из моделей вырождается в другую."
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон моделей по единому протоколу")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="переопределить список моделей")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    panel = load_panel(ROOT / cfg["data"]["path"])
    wide, report = build_matrix(panel, cfg["data"]["category"], max_gap=cfg["data"]["max_gap"])
    print(report.as_text(), end="\n\n")

    wide = sample_series(wide, cfg["sample"]["n_series"], seed=cfg["sample"]["seed"])
    folds = rolling_origin(wide.shape[0], cfg["split"]["horizon"], cfg["split"]["n_folds"])
    print(f"рядов в прогоне: {wide.shape[1]} | периодов: {wide.shape[0]}")
    for f in folds:
        print(f"  фолд {f.index}: обучение 1..{f.train_end}, тест {f.test_start + 1}..{f.test_end}")
    print()

    workers = int(cfg.get("compute", {}).get("workers", 0)) or max(1, (os.cpu_count() or 2) - 1)
    print(f"параллельно процессов: {workers}\n")

    names = args.models if args.models else cfg["models"]
    parts = []
    for name in names:
        started = time.perf_counter()
        part = evaluate(wide, name, cfg["split"]["horizon"], cfg["split"]["n_folds"], workers)
        parts.append(part)
        print(f"{name}: {time.perf_counter() - started:.1f} с")

    per_series = pd.concat(parts, ignore_index=True)
    summary = summarise(per_series)

    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_series.to_csv(out_dir / "per_series.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")

    print("\n" + summary.to_string(float_format=lambda v: f"{v:,.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
