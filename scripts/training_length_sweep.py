"""Как ценность переноса зависит от длины обучающей истории.

Три фолда протокола дали одинаковый по форме результат у всех четырёх источников
переноса: на фолде 0 (15 месяцев обучения) они помогают сильно, на фолде 1 слабее,
на фолде 2 (21 месяц) уже мешают. Трёх точек мало, чтобы называть это
закономерностью, поэтому здесь то же самое считается на всех origin'ах, которые
панель вообще допускает.

Это **диагностика, а не протокол оценки**. Протокол — три фолда в
`configs/*.yaml`, и итоговые числа берутся оттуда. Здесь origin'ы идут подряд
и пересекаются, поэтому их нельзя усреднять как независимые наблюдения; смысл
только в форме зависимости.

    .venv/bin/python -u scripts/training_length_sweep.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_matrix, load_panel  # noqa: E402
from src.external import ExternalFeatures, load_aggregate, load_industry  # noqa: E402
from src.models.global_model import GlobalGBM, PanelContext  # noqa: E402

REFERENCE = ROOT / "data" / "reference" / "sberindex"
HORIZON = 3
FIRST_ORIGIN = 12  # раньше на горизонт 3 остаётся слишком мало обучающих пар

VARIANTS = {
    "базовая": {},
    "доли категорий": {"categories": True},
    "общий фактор": {"common_factor": True},
    "стек категорий": {"stack_categories": True},
    "стек + фактор": {"stack_categories": True, "common_factor": True},
    "предобучение": {"pretrain": True},
}


def main() -> int:
    warnings.filterwarnings("ignore")
    panel = load_panel(ROOT / "data/raw/mo_spending_2023-2024.parquet")
    wide, _ = build_matrix(panel, "Все категории", max_gap=2)
    index = pd.to_datetime(wide.index).to_period("M")
    values = wide.to_numpy(dtype=float).T

    categories = {
        name: build_matrix(panel, name, max_gap=2)[0].reindex(index=wide.index)
        for name in sorted(panel["category_15"].unique())
        if name != "Все категории"
    }
    context = PanelContext(
        index=index,
        categories=categories,
        external=ExternalFeatures(REFERENCE),
        aggregate=load_aggregate(REFERENCE),
        long_series=load_industry(REFERENCE),
    )

    origins = list(range(FIRST_ORIGIN, len(index) - HORIZON + 1))
    print(f"origin'ов: {len(origins)}, обучение от {FIRST_ORIGIN} до {origins[-1]} месяцев")
    print(f"ряды: {wide.shape[1]}, горизонт {HORIZON}\n")

    rows = []
    for train_end in origins:
        test = values[:, train_end : train_end + HORIZON]
        base_level = values[:, train_end - 1]
        record = {
            "обучение, мес": train_end,
            "тест": f"{index[train_end]}..{index[train_end + HORIZON - 1]}",
            "наивная": float(np.nanmean(np.abs(base_level[:, None] - test))),
        }
        for label, kwargs in VARIANTS.items():
            model = GlobalGBM(**kwargs)
            model.set_context(context)
            model.fit(wide, train_end, HORIZON)
            predicted = model.predict(wide, train_end, HORIZON)
            record[label] = float(np.nanmean(np.abs(predicted - test)))
        rows.append(record)
        print("  " + " | ".join(
            f"{k}: {v:.0f}" if isinstance(v, float) else f"{k}: {v}" for k, v in record.items()
        ))

    table = pd.DataFrame(rows).set_index("обучение, мес")
    print("\n\nMAE ПО ДЛИНЕ ОБУЧЕНИЯ")
    print(table.drop(columns="тест").round(0).to_string())

    print("\n\nВЫИГРЫШ КАЖДОГО ИСТОЧНИКА К БАЗОВОЙ МОДЕЛИ, ₽")
    gain = table[[c for c in VARIANTS if c != "базовая"]].sub(table["базовая"], axis=0)
    print(gain.round(0).to_string())
    print("\nотрицательное число = источник помогает")

    # Рекомендация для практики держится на этой строке: какая конструкция
    # выигрывает при каждой длине обучения. Судить по среднему нельзя —
    # прогнозировать предстоит с самой длинной историей, какая есть.
    print("\n\nЛУЧШАЯ КОНСТРУКЦИЯ НА КАЖДОЙ ДЛИНЕ ОБУЧЕНИЯ")
    best = table[list(VARIANTS)].idxmin(axis=1)
    for length, name in best.items():
        print(f"  {length:2d} мес: {name:18s} MAE {table.loc[length, name]:6.0f} "
              f"(базовая {table.loc[length, 'базовая']:.0f})")

    out = ROOT / "results" / "training_length_sweep.csv"
    table.to_csv(out)
    print(f"\nсохранено: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
