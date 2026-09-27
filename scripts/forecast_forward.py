"""Прогноз вперёд от конца панели (декабрь 2024) на 2025 год.

В отличие от `scripts/horizons.py` (скользящий origin, сравнение моделей на
исторических фолдах) здесь origin один — последний месяц матрицы, обучение
идёт по всем 24 месяцам панели, а прогноз уходит за её пределы, в 2025 год,
которого в муниципальном разрезе ещё нет. Протокол — только в
`configs/forecast_forward.yaml`.

**Три модели на каждом горизонте.** Рекомендованная правилом отчёта
(`global_gbm_cat` до h=6, `two_stage` на h=12) и `two_stage` как второй столбец
для сравнения — на h=12 это одна и та же модель, и в файл идёт одна партия
строк, а не две одинаковые. Плюс `two_stage_known`: разнос ОПУБЛИКОВАННОГО
федерального агрегата, единственная модель, которой отдаётся будущее намеренно
(её описание — почему).

**Утечка.** Прогнозным моделям (`global_gbm_cat`, `two_stage`) достаётся контекст,
обрезанный по origin: агрегат и длинные ряды — `.loc[:origin]`, и данных после
origin в контексте просто нет — их оттуда физически нельзя прочитать, а не
«можно, но накажет исключением». Внешние признаки отключены совсем (`external=None`)
не по этой же причине, а потому что моделям этого протокола они не нужны; модели,
которой они бы понадобились, обрезка оставила бы работать без внешних признаков —
не с признаками из будущего.

**Проверка первого этапа по факту.** Федеральный агрегат СберИндекс публикует
регулярно, и в нём 2025 год уже есть (до августа 2026). Это разово позволяет
сверить прогноз агрегата `two_stage` с тем, что произошло на самом деле, —
не дожидаясь муниципального разреза за 2025 год. Ориентиры — классическая
наивная («как в декабре») и сезонно-наивная («как год назад»); обе тоже видят
только прошлое.

    .venv/bin/python -u scripts/forecast_forward.py [--config configs/forecast_forward.yaml]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_matrix, load_panel  # noqa: E402
from src.external import CANDIDATES, choose_model  # noqa: E402
from src.models.global_model import PanelContext  # noqa: E402
from src.models.two_stage import TwoStage, TwoStageKnownAggregate  # noqa: E402
from src.regions import attach_regions, load_dictionary  # noqa: E402
from src.run import GLOBAL_MODELS, build_context  # noqa: E402

# Как в scripts/horizons.py: модель оракула считается только здесь и в основной
# реестр не входит — с прогнозными моделями она не сравнивается на равных.
EXTRA_GLOBAL = {"two_stage_known": lambda: TwoStageKnownAggregate()}

FORECAST_COLUMNS = [
    "series_id", "region", "oktmo", "model", "horizon", "month",
    "forecast", "recommended", "uses_published_aggregate",
]
CHECK_COLUMNS = ["horizon", "method", "aggregate_model", "month", "forecast", "actual", "error_pct"]


def _future_index(index: pd.Index, n: int) -> pd.DatetimeIndex:
    """`n` месяцев после конца `index`, того же типа (начало месяца), что и `index` сам.

    Месяц считается через `pd.Period`, а не `DateOffset(months=1)` + `date_range(freq="MS")`:
    если индекс панели когда-нибудь станет концом месяца, а не началом (например,
    2025-01-31), `date_range` с `freq="MS"` от такой даты молча стартует со следующего
    начала месяца — с 2025-02-01, а не с 2025-01-01, — и весь хвост дополнения
    незаметно сдвинется на месяц. `Period` знает календарный месяц как единицу,
    а не как дату внутри него, и такой ошибки не допускает по построению.
    """
    origin = pd.to_datetime(index[-1]).to_period("M")
    return pd.period_range(start=origin + 1, periods=n, freq="M").to_timestamp()


def _long_rows(
    matrix: np.ndarray, columns: pd.Index, origin: pd.Period, horizon: int, model_name: str,
    *, recommended: bool, uses_published_aggregate: bool,
) -> pd.DataFrame:
    """Матрица (рядов × horizon) → длинный формат, по строке на пару (ряд, месяц).

    `matrix.reshape(-1)` разворачивает построчно (ряд за рядом), поэтому `series_id`
    повторяется блоками по `horizon`, а список месяцев — целиком на каждый ряд:
    это ровно тот же порядок.
    """
    months = [str(origin + step) for step in range(1, horizon + 1)]
    return pd.DataFrame({
        "series_id": np.repeat(np.asarray(columns), horizon),
        "model": model_name,
        "horizon": horizon,
        "month": months * len(columns),
        "forecast": np.asarray(matrix, dtype=float).reshape(-1),
        "recommended": recommended,
        "uses_published_aggregate": uses_published_aggregate,
    })


def _aggregate_check(
    context: PanelContext, context_cut: PanelContext,
    two_stage_by_horizon: dict[int, TwoStage], origin: pd.Period, horizons: list[int],
) -> pd.DataFrame:
    """Проверка первого этапа по факту: прогноз `two_stage` и два классических ориентира
    против фактического федерального агрегата (он один на все горизонты и уже знает 2025-й).

    Прогноз агрегата пересчитывается здесь же той же парой функций
    (`choose_model` + `CANDIDATES`), что и внутри `TwoStage._forecast_aggregate`, и сверяется
    с `model.forecast` уже подогнанного экземпляра `two_stage` этого горизонта — проверка
    обязана описывать именно тот первый этап, который дал прогноз, а не какой-то похожий.
    """
    # `context_cut.aggregate` уже обрезан по origin — это и есть «история до origin»
    # для бэктеста. Пересчитывать обрезку заново смысла нет: результат тот же самый ряд.
    history = np.log(context_cut.aggregate.to_numpy(dtype=float))
    last_level = float(context_cut.aggregate.iloc[-1])

    rows = []
    for h in horizons:
        ts_model = two_stage_by_horizon[h]
        name, _explained = choose_model(history, h)
        two_stage_forecast = np.exp(CANDIDATES[name](history, h))
        if not np.allclose(two_stage_forecast, ts_model.forecast, rtol=0, atol=0):
            raise ValueError(
                f"h={h}: пересчитанный прогноз агрегата не совпал с model.forecast "
                "экземпляра two_stage — проверка описывала бы не тот первый этап, "
                "который на самом деле дал прогноз"
            )

        months = [origin + step for step in range(1, h + 1)]
        # Факт — БЕЗ обрезки: это проверка по факту, а не обучение.
        actual = context.aggregate.reindex(months).to_numpy(dtype=float)
        naive_forecast = np.full(h, last_level)
        seasonal_forecast = np.array(
            [float(context_cut.aggregate.get(m - 12, np.nan)) for m in months]
        )

        for method, aggregate_model, values in (
            ("two_stage", name, two_stage_forecast),
            ("naive", "как в декабре", naive_forecast),
            ("seasonal_naive", "как год назад", seasonal_forecast),
        ):
            for step, month in enumerate(months):
                f, a = float(values[step]), float(actual[step])
                rows.append({
                    "horizon": h, "method": method, "aggregate_model": aggregate_model,
                    "month": str(month), "forecast": f, "actual": a,
                    "error_pct": (f - a) / a * 100.0,
                })
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def forecast_forward(
    wide: pd.DataFrame, context: PanelContext, regions: pd.DataFrame, cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Прогноз на 2025 год по протоколу `cfg` плюс проверка первого этапа по факту.

    `wide` — полная матрица панели (24 месяца), `context` — ПОЛНЫЙ контекст
    (`src.run.build_context`, агрегат до августа 2026 без обрезки), `regions` — кадр
    `series_id, region, oktmo` для колонок вывода (строится `attach_regions`, здесь
    только приклеивается). Возвращает кадр прогноза, кадр проверки агрегата
    и заметки моделей (`model.notes`, уже с привязкой к горизонту и модели).
    """
    origin = pd.Period(cfg["origin"], freq="M")
    index = pd.to_datetime(wide.index).to_period("M")
    if index[-1] != origin:
        raise ValueError(
            f"origin в конфиге ({origin}) не совпадает с последним месяцем матрицы "
            f"({index[-1]}): обучение здесь всегда идёт по всем месяцам панели, "
            "и расхождение значит прогноз не от того месяца"
        )
    if context.aggregate is None:
        raise ValueError(
            "в контексте нет федерального агрегата (context.aggregate is None) — "
            "без него не работают ни two_stage/two_stage_known, ни проверка по факту"
        )
    train_end = len(wide)

    # Обрезка по origin для прогнозных моделей: данных после origin в контексте
    # просто нет. external=None — моделям этого протокола он не нужен; модели,
    # которой он бы понадобился, обрезка оставила бы работать без внешних
    # признаков, а не с признаками из будущего.
    context_cut = dataclasses.replace(
        context,
        aggregate=context.aggregate.loc[:origin],
        long_series={name: s.loc[:origin] for name, s in context.long_series.items()},
        external=None,
    )

    # two_stage_known прогнозирует известный агрегат за месяцы 2025-го и должен находить
    # их в индексе — поэтому ей одной передаётся wide, дополненный NaN-строками на 2025 год.
    # Дальше max(horizons) месяцев хватает на самый длинный горизонт протокола.
    future_index = _future_index(wide.index, max(cfg["horizons"]))
    padded = pd.concat([wide, pd.DataFrame(np.nan, index=future_index, columns=wide.columns)])

    known_name = cfg["known_aggregate"]
    forecast_parts: list[pd.DataFrame] = []
    notes: list[dict] = []
    two_stage_by_horizon: dict[int, TwoStage] = {}

    def _fit_and_collect(model, data, model_context, name, h, *, recommended, uses_published_aggregate):
        if hasattr(model, "set_context"):
            model.set_context(model_context)
        model.fit(data, train_end, h)
        matrix = model.predict(data, train_end, h)
        for note in getattr(model, "notes", []):
            print(f"  h={h} {name}: {note}")
            notes.append({"horizon": h, "model": name, "note": note})
        forecast_parts.append(_long_rows(
            matrix, wide.columns, origin, h, name,
            recommended=recommended, uses_published_aggregate=uses_published_aggregate,
        ))

    for h in cfg["horizons"]:
        recommended_name = cfg["recommended"][h]
        comparison_name = cfg["comparison"]
        # dict.fromkeys дедуплицирует, сохраняя порядок: на h=12, где обе роли —
        # одна и та же модель, это ровно одна партия строк, а не две одинаковые.
        names = list(dict.fromkeys([recommended_name, comparison_name]))
        for name in names:
            model = GLOBAL_MODELS[name]()
            _fit_and_collect(
                model, wide, context_cut, name, h,
                recommended=(name == recommended_name), uses_published_aggregate=False,
            )
            if name == "two_stage":
                two_stage_by_horizon[h] = model
        print(f"h={h}: {', '.join(names)} готовы")

        known_model = (GLOBAL_MODELS | EXTRA_GLOBAL)[known_name]()
        _fit_and_collect(
            known_model, padded, context, known_name, h,
            recommended=False, uses_published_aggregate=True,
        )
        print(f"h={h}: {known_name} готова")

    forecast = pd.concat(forecast_parts, ignore_index=True)
    n_nan = int(forecast["forecast"].isna().sum())
    if n_nan:
        print(f"ВНИМАНИЕ: в прогнозе {n_nan} пропусков (NaN) — строки оставлены, не выброшены")

    forecast = forecast.merge(regions[["series_id", "region", "oktmo"]], on="series_id", how="left")
    forecast = forecast[FORECAST_COLUMNS]

    aggregate_check = _aggregate_check(context, context_cut, two_stage_by_horizon, origin, cfg["horizons"])
    return forecast, aggregate_check, notes


def build_regions_frame(columns: pd.Index) -> pd.DataFrame:
    """`series_id, region, oktmo` по справочнику муниципалитетов СберИндекса.

    Сущность — как в `src.run._series_regions`: `mo` — имя до суффикса гомонима
    (`" #N"`). У гомонимов регион и ОКТМО остаются пустыми — `attach_regions`
    решает так же и здесь, и в основном прогоне.
    """
    entities = pd.DataFrame({
        "series_id": columns, "mo": [str(c).split(" #")[0] for c in columns],
    })
    joined, report = attach_regions(entities, load_dictionary())
    print(report.as_text())
    return joined[["series_id", "region_name", "oktmo"]].rename(columns={"region_name": "region"})


def growth_stats(forecast: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Рост прогноза к тому же календарному месяцу 2024: медиана и 10/90 перцентили
    по объединённым парам (ряд, месяц) горизонта — отдельно по модели и горизонту."""
    by_period = wide.set_axis(pd.to_datetime(wide.index).to_period("M"), axis=0)
    by_period.index.name = "month_2024"
    molten = by_period.reset_index().melt(
        id_vars="month_2024", var_name="series_id", value_name="actual_2024"
    )
    rows = forecast.copy()
    rows["month_2024"] = pd.PeriodIndex(rows["month"], freq="M") - 12
    rows = rows.merge(molten, on=["series_id", "month_2024"], how="left")
    rows["growth"] = rows["forecast"] / rows["actual_2024"] - 1.0
    return rows.groupby(["model", "horizon"])["growth"].agg(
        медиана="median",
        p10=lambda s: float(s.quantile(0.10)),
        p90=lambda s: float(s.quantile(0.90)),
        n="count",
    )


def growth_stats_annual(forecast: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Только горизонт 12: рост суммы прогноза за 2025 год к сумме факта за 2024-й, по рядам."""
    h12 = forecast.loc[forecast["horizon"] == 12]
    forecast_sum = h12.groupby(["model", "series_id"])["forecast"].sum()
    # Последние 12 строк 24-месячной панели — это и есть календарный 2024 год.
    actual_sum = wide.iloc[-12:].sum(axis=0)
    actual_sum.index.name = "series_id"
    growth = forecast_sum.divide(actual_sum, level="series_id") - 1.0
    return growth.groupby("model").agg(
        медиана="median",
        p10=lambda s: float(s.quantile(0.10)),
        p90=lambda s: float(s.quantile(0.90)),
        n="count",
    )


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Прогноз вперёд от конца панели на 2025 год")
    parser.add_argument("--config", default="configs/forecast_forward.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    panel = load_panel(ROOT / cfg["data"]["path"])
    wide, report = build_matrix(panel, cfg["data"]["category"], max_gap=cfg["data"]["max_gap"])
    print(report.as_text(), end="\n\n")

    context = build_context(panel, wide, cfg)
    regions = build_regions_frame(wide.columns)
    print(f"рядов: {wide.shape[1]} | origin: {cfg['origin']} | горизонты: {cfg['horizons']}\n")

    forecast, aggregate_check, notes = forecast_forward(wide, context, regions, cfg)

    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = out_dir / cfg["output"]["forecast"]
    check_path = out_dir / cfg["output"]["aggregate_check"]
    forecast.to_csv(forecast_path, index=False)
    aggregate_check.to_csv(check_path, index=False)

    print("\nМОДЕЛЬ АГРЕГАТА И ПРАВИЛО ДОЛИ ПО ГОРИЗОНТАМ")
    for entry in notes:
        print(f"  h={entry['horizon']} {entry['model']}: {entry['note']}")

    pd.options.display.float_format = lambda v: f"{v:,.2f}"
    print("\nСРЕДНЯЯ АБСОЛЮТНАЯ ОШИБКА АГРЕГАТА, % (по методам и горизонтам)")
    mae_table = (
        aggregate_check.assign(abs_error=aggregate_check["error_pct"].abs())
        .groupby(["horizon", "method"])["abs_error"].mean().unstack("method")
    )
    print(mae_table.to_string())

    print("\nГОРИЗОНТ 12, ПОМЕСЯЧНО (прогноз / факт / ошибка,% по трём методам)")
    h12 = aggregate_check.loc[aggregate_check["horizon"] == 12]
    monthly = h12.pivot(index="month", columns="method", values=["forecast", "actual", "error_pct"])
    print(monthly.to_string())

    pd.options.display.float_format = lambda v: f"{v:,.4f}"
    print("\nРОСТ ПРОГНОЗА К ТОМУ ЖЕ МЕСЯЦУ 2024 (медиана, 10-й/90-й перцентиль по рядам)")
    print(growth_stats(forecast, wide).to_string())

    print("\nГОРИЗОНТ 12: РОСТ СУММЫ ЗА 2025 К СУММЕ ЗА 2024 (по рядам)")
    print(growth_stats_annual(forecast, wide).to_string())

    n_nan = int(forecast["forecast"].isna().sum())
    print(f"\nNaN в прогнозе: {n_nan}")
    print(f"сохранено: {forecast_path}, {check_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
