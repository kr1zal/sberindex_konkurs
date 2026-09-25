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
from src.external import ExternalFeatures, load_aggregate, load_industry  # noqa: E402
from src.regions import attach_regions, load_dictionary  # noqa: E402
from src.metrics import mae, mase, r2, smape  # noqa: E402
from src.models.classical import ARIMA, ETS, Theta  # noqa: E402
from src.models.foundation import Chronos, ChronosPanel, Moirai, TimesFM  # noqa: E402
from src.models.global_model import GlobalGBM, PanelContext  # noqa: E402
from src.models.naive import Drift, NaiveLast, SeasonalDrift, SeasonalNaive  # noqa: E402
from src.models.two_stage import TwoStage, TwoStageNews  # noqa: E402
from src.results_guard import check_same_panel, uneven_series  # noqa: E402
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
    "chronos_tiny": lambda: Chronos(size="tiny"),
    "chronos_small": lambda: Chronos(size="small"),
    "chronos_base": lambda: Chronos(size="base"),
    "chronos_small_log": lambda: Chronos(size="small", transform="log"),
    "chronos_base_log": lambda: Chronos(size="base", transform="log"),
}

# `prophet` — Prophet как есть, без единого нашего аргумента: на коротких окнах
# его собственный режим 'auto' не включает ни годовой, ни какой-либо другой
# сезонности. Это и есть эталон конкурса.
#
# `prophet_forced_yearly` — та же модель с насильно включённой годовой
# сезонностью. На 24 точках Фурье-компоненты годового цикла оставляют
# оптимизатору почти вырожденную задачу: подгонка дорожает в десятки раз,
# а прогноз уходит в отрицательное потребление. Это ловушка для практика,
# рассуждающего «данные месячные, значит нужна годовая сезонность», —
# и Prophet от неё защищается сам, выводя предупреждение.
# Панельные модели учатся на всей панели сразу и потому не влезают в интерфейс
# fit(y)/predict(h), рассчитанный на один ряд. Обучение идёт на ПОЛНОЙ панели,
# оценка - на той же выборке рядов, что и у остальных моделей: иначе сравнение
# считалось бы на разных множествах.
#
# Варианты различаются ровно одним источником признаков каждый — чтобы в итоговой
# таблице было видно, что именно дало выигрыш, а не «добавили всё сразу, стало лучше».
GLOBAL_MODELS = {
    "global_gbm": lambda: GlobalGBM(),
    "global_gbm_ext": lambda: GlobalGBM(external=True),
    "global_gbm_factor": lambda: GlobalGBM(common_factor=True),
    "global_gbm_pretrain": lambda: GlobalGBM(pretrain=True),
    "global_gbm_cat": lambda: GlobalGBM(categories=True),
    "global_gbm_all": lambda: GlobalGBM(common_factor=True, categories=True),
    "global_gbm_region": lambda: GlobalGBM(common_factor=True, regional_factor=True),
    "global_gbm_stack": lambda: GlobalGBM(stack_categories=True),
    "global_gbm_stack_factor": lambda: GlobalGBM(stack_categories=True, common_factor=True),
    # Фундаментальные модели тоже панельные: дообучение одно на фолд, а не своё
    # на каждый из 2028 рядов по пятнадцати точкам.
    "chronos_panel": lambda: ChronosPanel(size="small"),
    "chronos_ft": lambda: ChronosPanel(size="small", finetune=True),
    "chronos_ft_base": lambda: ChronosPanel(size="base", finetune=True),
    "timesfm": lambda: TimesFM(),
    "moirai": lambda: Moirai(),
    # Двухэтапный прогноз: тезис «одно число и разнос» в виде конструкции,
    # а не только измерения. Две версии агрегата — сырой и сезонно сглаженный.
    "two_stage": lambda: TwoStage(),
    "two_stage_sa": lambda: TwoStage(
        slug="consumper-spending-index-sa", where={"type": "Всего"}
    ),
    "two_stage_news": lambda: TwoStageNews(),
}

PROPHET_VARIANTS = {
    "prophet": {},
    "prophet_forced_yearly": {"yearly_seasonality": True},
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


def evaluate_global(
    full: pd.DataFrame, subset: pd.DataFrame, model_name: str, horizon: int, n_folds: int,
    context: PanelContext | None = None,
) -> pd.DataFrame:
    """Панельная модель: одна подгонка на фолд, прогноз сразу для всех рядов выборки."""
    folds = rolling_origin(n_obs=full.shape[0], horizon=horizon, n_folds=n_folds)
    values = subset.to_numpy(dtype=float).T
    rows = []

    for fold in folds:
        model = GLOBAL_MODELS[model_name]()
        if context is not None and hasattr(model, "set_context"):
            model.set_context(context)
        model.fit(full, fold.train_end, horizon)
        # Как именно подогнан общий фактор, надо видеть в логе: множитель, уехавший
        # не туда, портит прогноз тихо и одинаково на всех рядах сразу.
        for note in getattr(model, "notes", []):
            print(f"  фолд {fold.index}: {note}")
        predicted = model.predict(subset, fold.train_end, horizon)

        for i, col in enumerate(subset.columns):
            y_pred = predicted[i]
            y_test = values[i, fold.test_start : fold.test_end]
            y_train = values[i, : fold.train_end]
            if np.isnan(y_pred).any():
                rows.append(
                    {"model": model_name, "fold": fold.index, "mo": col,
                     "error": "недостаточно истории для признаков"}
                )
                continue
            rows.append(
                {
                    "model": model_name, "fold": fold.index, "mo": col,
                    "mae": mae(y_test, y_pred), "r2": r2(y_test, y_pred),
                    "smape": smape(y_test, y_pred),
                    "mase": mase(y_test, y_pred, y_train, season=1),
                    "error": None,
                }
            )
    return pd.DataFrame(rows)


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


def _warn_uneven(per_series: pd.DataFrame) -> None:
    """Кричит, если модели в файле гонялись на разном числе рядов.

    Сигнал, а не блокер: через `check_same_panel` такой файл уже не собрать,
    расхождение остаётся от прогонов до проверки — 21.09 `naive_last` на выборке
    из 300 рядов затёр свои полнопанельные строки. Сводка пишется всё равно.
    """
    uneven = uneven_series(per_series)
    if uneven is None:
        return
    by_count: dict[int, list[str]] = {}
    for model, n in uneven.items():
        by_count.setdefault(int(n), []).append(model)
    print(
        "ВНИМАНИЕ: модели гонялись на разном числе рядов (считая отказы), и сводка "
        "сравнивает их на разных множествах — "
        + "; ".join(f"{n}: {', '.join(models)}" for n, models in sorted(by_count.items()))
        + ". Модели с меньшим числом рядов стоит перегнать."
    )


def build_context(panel: pd.DataFrame, wide: pd.DataFrame, cfg: dict) -> PanelContext:
    """Данные, общие для всех панельных моделей: категории трат и длинные ряды.

    Собирается один раз на прогон. Категории раскладываются тем же ``build_matrix``
    и с тем же правилом пропусков, что и основная матрица, иначе доли считались бы
    по разным множествам рядов.
    """
    reference = ROOT / "data" / "reference" / "sberindex"
    categories: dict[str, pd.DataFrame] = {}
    for name in sorted(panel["category_15"].unique()):
        if name == cfg["data"]["category"]:
            continue
        part, _ = build_matrix(panel, name, max_gap=cfg["data"]["max_gap"])
        categories[name] = part.reindex(index=wide.index)

    regions = _series_regions(wide)
    external, aggregate, long_series = None, None, {}
    if reference.exists():
        external = ExternalFeatures(reference)
        aggregate = load_aggregate(reference)
        long_series = load_industry(reference)
        print(
            f"длинные ряды: агрегат {len(aggregate)} мес "
            f"({aggregate.index.min()}..{aggregate.index.max()}), "
            f"отраслевых рядов {len(long_series)}"
        )
    else:
        # Молчаливое отключение источника — ровно тот класс ошибки, от которого
        # в этом проекте уже пострадали трижды. Пусть видно будет в логе.
        print(f"  ВНИМАНИЕ: нет {reference}, модели с длинными рядами работать не смогут")
    print(f"категории трат: {', '.join(categories)}\n")

    return PanelContext(
        index=pd.to_datetime(wide.index).to_period("M"),
        categories=categories,
        external=external,
        aggregate=aggregate,
        long_series=long_series,
        regions=regions,
    )


def _series_regions(wide: pd.DataFrame) -> dict[str, str]:
    """Регион каждого ряда. Омонимы остаются без региона — так решено и не меняется.

    Идентификатор ряда для омонимичных названий выглядит как «Сергиевский #2»:
    суффикс добавлен при восстановлении сущностей, а в справочнике его нет.
    """
    entities = pd.DataFrame(
        {"series_id": wide.columns, "mo": [str(c).split(" #")[0] for c in wide.columns]}
    )
    joined, report = attach_regions(entities, load_dictionary())
    print(report.as_text())
    return {
        row.series_id: row.region_name
        for row in joined.itertuples()
        if isinstance(row.region_name, str) and row.region_name
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон моделей по единому протоколу")
    parser.add_argument(
        "--config", default="configs/full.yaml",
        help="протокол оценки, по умолчанию %(default)s — полная панель. Пилот на выборке "
             "из 300 рядов — configs/baseline.yaml, у него собственный каталог результатов",
    )
    parser.add_argument("--models", nargs="*", default=None, help="переопределить список моделей")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    panel = load_panel(ROOT / cfg["data"]["path"])
    wide, report = build_matrix(panel, cfg["data"]["category"], max_gap=cfg["data"]["max_gap"])
    print(report.as_text(), end="\n\n")

    full_panel = wide
    context = build_context(panel, full_panel, cfg)
    wide = sample_series(wide, cfg["sample"]["n_series"], seed=cfg["sample"]["seed"])
    folds = rolling_origin(wide.shape[0], cfg["split"]["horizon"], cfg["split"]["n_folds"])
    print(f"рядов в прогоне: {wide.shape[1]} | периодов: {wide.shape[0]}")
    for f in folds:
        print(f"  фолд {f.index}: обучение 1..{f.train_end}, тест {f.test_start + 1}..{f.test_end}")
    print()

    workers = int(cfg.get("compute", {}).get("workers", 0)) or max(1, (os.cpu_count() or 2) - 1)
    print(f"параллельно процессов: {workers}\n")

    known = set(REGISTRY) | set(PROPHET_VARIANTS) | set(GLOBAL_MODELS)
    configured = set(cfg["models"])
    if unknown := configured - known:
        raise KeyError(f"в конфиге есть незарегистрированные модели: {sorted(unknown)}")
    if missing := known - configured:
        # Модель, зарегистрированная в коде, но забытая в конфиге, тихо выпадает
        # из итоговой таблицы сравнения — заметить это можно только по её отсутствию,
        # а отсутствие в глаза не бросается. Уже случилось однажды с классическими моделями.
        print(f"  вне конфига (в прогон не войдут): {', '.join(sorted(missing))}\n")

    names = args.models if args.models else cfg["models"]
    parts = []
    for name in names:
        started = time.perf_counter()
        if name in GLOBAL_MODELS:
            part = evaluate_global(
                full_panel, wide, name,
                cfg["split"]["horizon"], cfg["split"]["n_folds"], context,
            )
        else:
            part = evaluate(wide, name, cfg["split"]["horizon"], cfg["split"]["n_folds"], workers)
        parts.append(part)
        print(f"{name}: {time.perf_counter() - started:.1f} с")

    per_series = pd.concat(parts, ignore_index=True)
    summary = summarise(per_series)

    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Результаты предыдущих прогонов не затираются, а дополняются: модели часто
    # гоняются частями (дорогие отдельно от дешёвых), и перезапись молча оставляла
    # бы в итоговой таблице только последнюю партию. Строки одной и той же модели
    # заменяются новыми — повторный прогон обновляет, а не дублирует.
    # Замена честна, только если партия считана на тех же рядах и фолдах, что весь
    # файл: сверка идёт до отбрасывания строк и до записи (src/results_guard.py).
    per_path = out_dir / "per_series.csv"
    if per_path.exists():
        previous = pd.read_csv(per_path)
        check_same_panel(previous, per_series, path=per_path)
        previous = previous[~previous["model"].isin(per_series["model"].unique())]
        per_series = pd.concat([previous, per_series], ignore_index=True)

    per_series.to_csv(per_path, index=False)
    summary = summarise(per_series)
    _warn_uneven(per_series)
    summary.to_csv(out_dir / "summary.csv")

    print("\n" + summary.to_string(float_format=lambda v: f"{v:,.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
