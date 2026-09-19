"""Прогон всех моделей по единому протоколу. Точка входа каркаса."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_matrix, load_panel, sample_series  # noqa: E402
from src.metrics import mae, mase, r2, smape  # noqa: E402
from src.models.naive import Drift, NaiveLast, SeasonalDrift, SeasonalNaive  # noqa: E402
from src.split import rolling_origin  # noqa: E402

REGISTRY = {
    "naive_last": lambda: NaiveLast(),
    "seasonal_naive": lambda: SeasonalNaive(season=12),
    "drift": lambda: Drift(),
    "seasonal_drift": lambda: SeasonalDrift(season=12),
}


def _build_model(name: str):
    if name == "prophet":
        from src.models.prophet_model import ProphetModel

        return ProphetModel()
    if name not in REGISTRY:
        raise KeyError(f"модель не зарегистрирована: {name}")
    return REGISTRY[name]()


def evaluate(wide: pd.DataFrame, model_name: str, horizon: int, n_folds: int) -> pd.DataFrame:
    """Гоняет одну модель по всем рядам и фолдам. Возвращает результат по каждой паре."""
    folds = rolling_origin(n_obs=wide.shape[0], horizon=horizon, n_folds=n_folds)
    index = wide.index
    rows = []

    for fold in folds:
        for col in wide.columns:
            series = wide[col].to_numpy(dtype=float)
            y_train = series[: fold.train_end]
            y_test = series[fold.test_start : fold.test_end]

            model = _build_model(model_name)
            if hasattr(model, "set_index"):
                model.set_index(index)
            try:
                y_pred = model.fit(y_train).predict(horizon)
            except Exception as exc:  # одна упавшая модель не должна ронять прогон
                rows.append(
                    {"model": model_name, "fold": fold.index, "mo": col, "error": str(exc)[:120]}
                )
                continue

            rows.append(
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
    return summary.sort_values("MAE")


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

    names = args.models if args.models else cfg["models"]
    parts = []
    for name in names:
        started = time.perf_counter()
        part = evaluate(wide, name, cfg["split"]["horizon"], cfg["split"]["n_folds"])
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
