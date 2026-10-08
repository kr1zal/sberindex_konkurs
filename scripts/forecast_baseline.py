"""Прогноз эталона конкурса на 2025 год по каждому ряду панели.

`scripts/forecast_forward.py` прогнозирует вперёд рекомендуемыми моделями;
здесь то же для эталона — Prophet по умолчанию, та самая модель, с которой
сравниваются остальные в основном протоколе. Обучение — все 24 месяца ряда,
прогноз — на `baseline.horizon` месяцев вперёд, ряд за рядом (`src.run._build_model`
и `set_index`, как при оценке на фолдах). Протокол — в `configs/forecast_forward.yaml`,
раздел `baseline`; вывод — `results/<baseline.output>`, столбцы как у таблицы
прогноза вперёд без признаков роли (`recommended`, `uses_published_aggregate`:
эталон ни то, ни другое).

Ряд, на котором модель упала, остаётся в файле строками с пустым прогнозом
и причиной в `error` — отказы видны, а не выброшены.

    .venv/bin/python -u scripts/forecast_baseline.py [--config configs/forecast_forward.yaml]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.forecast_forward import build_regions_frame  # noqa: E402
from src.data import build_matrix, load_panel  # noqa: E402
from src.results_guard import failure_reason  # noqa: E402
from src.run import _build_model  # noqa: E402

BASELINE_COLUMNS = ["series_id", "region", "oktmo", "model", "horizon", "month", "forecast", "error"]


def _forecast_series(task: tuple) -> dict:
    """Одна подгонка на ряд. Верхнего уровня — чтобы пиклилось в пул процессов."""
    model_name, col, series, index, horizon = task
    model = _build_model(model_name)
    if hasattr(model, "set_index"):
        model.set_index(index)
    try:
        forecast = np.asarray(model.fit(series).predict(horizon), dtype=float)
    except Exception as exc:  # одна упавшая подгонка не должна ронять прогон
        return {"series_id": col, "forecast": np.full(horizon, np.nan), "error": failure_reason(exc)[:120]}
    return {"series_id": col, "forecast": forecast, "error": None}


def forecast_baseline(wide: pd.DataFrame, regions: pd.DataFrame, cfg: dict, workers: int = 1) -> pd.DataFrame:
    """Прогноз эталона по всем рядам `wide` от её последнего месяца на `cfg['baseline']['horizon']` месяцев."""
    base = cfg["baseline"]
    model_name, horizon = base["model"], int(base["horizon"])
    origin = pd.Period(cfg["origin"], freq="M")
    index = pd.to_datetime(wide.index)
    if index[-1].to_period("M") != origin:
        raise ValueError(
            f"origin в конфиге ({origin}) не совпадает с последним месяцем матрицы "
            f"({index[-1].to_period('M')}): обучение здесь всегда идёт по всем месяцам панели"
        )
    months = [str(origin + step) for step in range(1, horizon + 1)]
    tasks = [(model_name, col, wide[col].to_numpy(dtype=float), index, horizon) for col in wide.columns]

    if workers <= 1:
        results = [_forecast_series(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_forecast_series, tasks, chunksize=8))

    frame = pd.DataFrame({
        "series_id": np.repeat([r["series_id"] for r in results], horizon),
        "model": model_name,
        "horizon": horizon,
        "month": months * len(results),
        "forecast": np.concatenate([r["forecast"] for r in results]),
        "error": np.repeat([r["error"] for r in results], horizon),
    })
    frame = frame.merge(regions[["series_id", "region", "oktmo"]], on="series_id", how="left")
    return frame[BASELINE_COLUMNS]


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Прогноз эталона конкурса на 2025 год по рядам панели")
    parser.add_argument("--config", default="configs/forecast_forward.yaml")
    parser.add_argument("--workers", type=int, default=0, help="процессов; 0 — все ядра минус одно")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    panel = load_panel(ROOT / cfg["data"]["path"])
    wide, report = build_matrix(panel, cfg["data"]["category"], max_gap=cfg["data"]["max_gap"])
    print(report.as_text(), end="\n\n")
    regions = build_regions_frame(wide.columns)
    base = cfg["baseline"]
    print(f"рядов: {wide.shape[1]} | origin: {cfg['origin']} | эталон: {base['model']} | "
          f"горизонт: {base['horizon']} | параллельно процессов: {workers}\n", flush=True)

    forecast = forecast_baseline(wide, regions, cfg, workers)

    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / base["output"]
    forecast.to_csv(path, index=False)

    failed = forecast.loc[forecast["error"].notna(), "series_id"].nunique()
    print(f"отказов: {failed} из {wide.shape[1]} рядов")
    print(f"NaN в прогнозе: {int(forecast['forecast'].isna().sum())}")
    print(f"сохранено: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
