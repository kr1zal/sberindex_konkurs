"""Проверка, что модели с длинными рядами не заглядывают в будущее.

Длинные ряды СберИндекса идут до августа 2026, а прогнозируем мы 2024 год.
Ряд, обрезанный неправильно, не уронит прогон и не выдаст исключения — он просто
улучшит метрики, и улучшение будет бессмысленным. Такую ошибку глазами по коду
не ловят, поэтому она проверяется машиной.

Способ прямой: подгоняем модель дважды — на полном ряде и на ряде, заранее
обрезанном по origin. Если результаты совпадают до последнего знака, значит
данных после origin модель не касалась.

    .venv/bin/python -u scripts/check_no_leakage.py
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
from src.external import CommonFactor, ExternalFeatures, load_aggregate, load_industry  # noqa: E402
from src.models.global_model import GlobalGBM, PanelContext  # noqa: E402

REFERENCE = ROOT / "data" / "reference" / "sberindex"
ORIGINS = (15, 18, 21)  # train_end трёх фолдов протокола


def main() -> int:
    warnings.filterwarnings("ignore")
    panel = load_panel(ROOT / "data/raw/mo_spending_2023-2024.parquet")
    wide, _ = build_matrix(panel, "Все категории", max_gap=2)
    index = pd.to_datetime(wide.index).to_period("M")

    aggregate = load_aggregate(REFERENCE)
    industry = load_industry(REFERENCE)
    print(f"федеральный ряд: {len(aggregate)} мес, до {aggregate.index.max()}")
    print(f"панель: {len(index)} мес, до {index.max()}")
    print(f"отраслевых рядов: {len(industry)}\n")

    failures = []

    for train_end in ORIGINS:
        origin = index[train_end - 1]

        full = CommonFactor(aggregate=aggregate).fit(wide, train_end, 3)
        cut = CommonFactor(aggregate=aggregate.loc[:origin]).fit(wide, train_end, 3)
        if np.allclose(full.multipliers, cut.multipliers, rtol=1e-12, atol=1e-12):
            print(f"общий фактор, origin {origin}: будущее не используется")
        else:
            failures.append(f"общий фактор, origin {origin}")
            print(f"общий фактор, origin {origin}: РАЗЛИЧАЕТСЯ — есть утечка")

        context_full = PanelContext(index=index, long_series=industry)
        context_cut = PanelContext(
            index=index, long_series={k: v.loc[:origin] for k, v in industry.items()}
        )
        sizes = []
        for context in (context_full, context_cut):
            model = GlobalGBM(pretrain=True)
            model.set_context(context)
            sizes.append(len(model._pretrain_set(wide, train_end, 3)[1][1]))
        if sizes[0] == sizes[1]:
            print(f"предобучение, origin {origin}: {sizes[0]} примеров, все до origin")
        else:
            failures.append(f"предобучение, origin {origin}")
            print(f"предобучение, origin {origin}: {sizes[0]} против {sizes[1]} — есть утечка")

    # Внешние признаки читаются в момент t и назад по лагам. Проверяем на самом
    # позднем t, который вообще встречается в прогоне: origin последнего фолда.
    external = ExternalFeatures(REFERENCE)
    latest = max(ORIGINS) - 1
    row_full = external.row(index, latest)
    trimmed = ExternalFeatures.__new__(ExternalFeatures)
    trimmed.series = {k: v.loc[: index[latest]] for k, v in external.series.items()}
    row_cut = trimmed.row(index, latest)
    if row_full == row_cut:
        print(f"внешние признаки, t={index[latest]}: {len(row_full)} шт, будущее не используется")
    else:
        failures.append("внешние признаки")
        print("внешние признаки: РАЗЛИЧАЮТСЯ — есть утечка")

    print()
    if failures:
        print("ПРОВАЛ: " + "; ".join(failures))
        return 1
    print("всё чисто: ни одна модель не обращается к данным после origin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
