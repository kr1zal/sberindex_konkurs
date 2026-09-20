"""Из чего состоит ошибка прогноза: общее смещение или разброс между рядами.

Средняя ошибка ничего не говорит о том, ошибается ли модель по-разному в разных
муниципалитетах или одинаково во всех сразу. А это разные болезни, и лечатся
они по-разному. Разброс между рядами — это предел информации в самих рядах.
Общее смещение — это когда модель промахнулась мимо того, что произошло со всей
страной, и тогда лечится оно внешним источником, а не лучшим алгоритмом.

Разделение простое. Для каждого шага горизонта считается медианное по рядам
отношение прогноза к последнему известному значению и такое же отношение факта.
Прогноз домножается на их частное — то есть калибруется идеально, задним числом,
одним множителем на всю панель. Насколько от этого упала ошибка, настолько она
и состояла из общего смещения; остаток — разброс.

Калибровка задним числом, разумеется, не модель: множитель берётся из тестовых
месяцев. Это измерительный инструмент, и его результат — верхняя граница того,
что вообще можно выиграть, угадав общее движение.

    .venv/bin/python -u scripts/error_decomposition.py
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
from src.regions import attach_regions, load_dictionary  # noqa: E402
from src.split import rolling_origin  # noqa: E402

REFERENCE = ROOT / "data" / "reference" / "sberindex"
HORIZON, N_FOLDS = 3, 3

MODELS = {
    "naive_last": None,  # считается без модели: прогноз равен последнему значению
    "global_gbm": {},
    "global_gbm_cat": {"categories": True},
    "global_gbm_factor": {"common_factor": True},
    "global_gbm_stack_factor": {"stack_categories": True, "common_factor": True},
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
    entities = pd.DataFrame(
        {"series_id": wide.columns, "mo": [str(c).split(" #")[0] for c in wide.columns]}
    )
    joined, _ = attach_regions(entities, load_dictionary())
    context = PanelContext(
        index=index,
        categories=categories,
        external=ExternalFeatures(REFERENCE),
        aggregate=load_aggregate(REFERENCE),
        long_series=load_industry(REFERENCE),
        regions={
            row.series_id: row.region_name
            for row in joined.itertuples()
            if isinstance(row.region_name, str) and row.region_name
        },
    )

    rows = []
    for fold in rolling_origin(len(index), HORIZON, N_FOLDS):
        origin = fold.train_end - 1
        base = values[:, origin]
        actual = values[:, fold.test_start : fold.test_end]
        actual_ratio = np.array(
            [np.nanmedian(actual[:, h] / base) for h in range(HORIZON)]
        )

        for name, kwargs in MODELS.items():
            if kwargs is None:
                predicted = np.repeat(base[:, None], HORIZON, axis=1)
            else:
                model = GlobalGBM(**kwargs)
                model.set_context(context)
                predicted = model.fit(wide, fold.train_end, HORIZON).predict(
                    wide, fold.train_end, HORIZON
                )
            predicted_ratio = np.array(
                [np.nanmedian(predicted[:, h] / base) for h in range(HORIZON)]
            )
            mae = float(np.nanmean(np.abs(predicted - actual)))
            calibrated = predicted / predicted_ratio[None, :] * actual_ratio[None, :]
            residual = float(np.nanmean(np.abs(calibrated - actual)))

            rows.append({
                "фолд": fold.index,
                "обучение, мес": fold.train_end,
                "тест": f"{index[fold.test_start]}..{index[fold.test_end - 1]}",
                "модель": name,
                "MAE": mae,
                "смещение": mae - residual,
                "разброс": residual,
                "доля смещения": (mae - residual) / mae if mae else np.nan,
                "прогноз h3": predicted_ratio[-1],
                "факт h3": actual_ratio[-1],
            })
            print(f"  фолд {fold.index} {name:24s} MAE {mae:7.0f} = смещение {mae - residual:6.0f} "
                  f"+ разброс {residual:6.0f} | отношение на h3: прогноз "
                  f"{predicted_ratio[-1]:.3f}, факт {actual_ratio[-1]:.3f}")

    table = pd.DataFrame(rows)
    out = ROOT / "results" / "error_decomposition.csv"
    table.to_csv(out, index=False)

    print("\n\nДОЛЯ ОБЩЕГО СМЕЩЕНИЯ В ОШИБКЕ, %")
    share = table.pivot_table(index="модель", columns="фолд", values="доля смещения") * 100
    print(share.round(0).to_string())
    print(f"\nсохранено: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
