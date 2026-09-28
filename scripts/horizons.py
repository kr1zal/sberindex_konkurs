"""Пять горизонтов прогнозирования: наукаст и 1, 3, 6, 12 месяцев.

Организаторы (ответ от 24.09.2026) просят сравнить модели для наукастинга
и ещё как минимум на четырёх горизонтах: 1, 3, 6 и 12 месяцев. Основной протокол
работы — горизонт 3 и три фолда. Здесь та же конструкция разбиения — скользящий
origin с расширяющимся окном, шаг равен горизонту — применяется к каждому
горизонту отдельно. Протокол задан в `configs/horizons.yaml` и больше нигде.

**Наукаст.** Внутримесячных данных в панели нет, поэтому наукаст месяца t —
это прогноз по данным до t−1, то есть горизонт 1. Отдельной строкой идёт
`two_stage_known`: федеральный агрегат за месяцы горизонта считается известным,
прогнозируется только разнос. На горизонте 1 это наукаст в узком смысле —
СберИндекс публикует федеральный ряд раньше муниципального разреза; на остальных
горизонтах это оракул общего движения, то есть нижняя граница ошибки, которую
оставляет межрядовый разброс.

**Во что упирается.** На горизонте 12 из 24 точек получается один фолд
с обучением 12 месяцев, и доверительных интервалов там нет. Прямая многошаговая
стратегия панельного бустинга на этом фолде не имеет обучающих примеров
по построению — горизонт равен длине истории. Это фиксируется как результат,
а не чинится: отказ модели попадает в таблицу с причиной.

Помимо MAE по фолду сохраняется ошибка по каждому шагу горизонта: она отвечает
на вопрос «как растёт ошибка с расстоянием от origin», который среднее по
горизонту прячет.

    .venv/bin/python -u scripts/horizons.py [--config configs/horizons.yaml] [--models ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_matrix, load_panel, sample_series  # noqa: E402
from src.metrics import (  # noqa: E402
    ROW_METRICS, SUMMARY_COLUMNS, r2_and_gains, row_metrics, warn_uneven_pairs,
)
from src.models.two_stage import TwoStageKnownAggregate  # noqa: E402
from src.results_guard import (  # noqa: E402
    check_plan, check_same_panel, read_results, realization_stamp, save_realization,
)
from src.run import (  # noqa: E402
    GLOBAL_MODELS, _build_model, build_context, failure_reason, refused,
)
from src.split import Fold, rolling_origin  # noqa: E402

# Модели только этого прогона. В основной реестр не входят: оракул агрегата
# с прогнозными моделями за одно место не соревнуется.
EXTRA_GLOBAL = {
    "two_stage_known": lambda: TwoStageKnownAggregate(),
}


def _record(
    model_name: str, horizon: int, fold: Fold, col: str,
    y_train: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray,
) -> dict:
    """Строка успеха: метрики той же функцией, что в src/run.py, — со слагаемыми R² пула."""
    return {
        "model": model_name, "horizon": horizon, "fold": fold.index,
        "train_end": fold.train_end, "mo": col,
        **row_metrics(y_test, y_pred, y_train), "error": None,
    }


def _failure(model_name: str, horizon: int, fold: Fold, col: str, reason: str) -> dict:
    return {
        "model": model_name, "horizon": horizon, "fold": fold.index,
        "train_end": fold.train_end, "mo": col,
        **dict.fromkeys(ROW_METRICS, np.nan), "error": reason[:120],
    }


def _score_series(task: tuple) -> tuple[list[dict], list[tuple]]:
    """Одна пара ряд-модель по всем фолдам одного горизонта. Верхнего уровня — чтобы пиклилось."""
    model_name, col, series, index, folds, horizon = task
    rows, steps = [], []
    for fold in folds:
        y_train = series[: fold.train_end]
        y_test = series[fold.test_start : fold.test_end]
        model = _build_model(model_name)
        if hasattr(model, "set_index"):
            model.set_index(index)
        try:
            y_pred = np.asarray(model.fit(y_train).predict(horizon), dtype=float)
        except Exception as exc:  # одна упавшая подгонка не должна ронять прогон
            rows.append(_failure(model_name, horizon, fold, col, failure_reason(exc)))
            continue
        rows.append(_record(model_name, horizon, fold, col, y_train, y_test, y_pred))
        steps.append((fold.index, np.abs(y_test - y_pred)))
    return rows, steps


def evaluate_series_models(
    wide: pd.DataFrame, model_name: str, horizon: int, folds: list[Fold], workers: int,
) -> tuple[pd.DataFrame, dict[int, list[np.ndarray]]]:
    index = wide.index
    tasks = [
        (model_name, col, wide[col].to_numpy(dtype=float), index, folds, horizon)
        for col in wide.columns
    ]
    if workers <= 1:
        results = [_score_series(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_score_series, tasks, chunksize=8))

    rows, per_fold_steps = [], {fold.index: [] for fold in folds}
    for part_rows, part_steps in results:
        rows.extend(part_rows)
        for fold_index, errors in part_steps:
            per_fold_steps[fold_index].append(errors)
    return pd.DataFrame(rows), per_fold_steps


def evaluate_panel_model(
    full: pd.DataFrame, subset: pd.DataFrame, model_name: str, horizon: int,
    folds: list[Fold], context,
) -> tuple[pd.DataFrame, dict[int, list[np.ndarray]]]:
    """Панельная модель: одна подгонка на фолд. Отказ на фолде — строка с причиной у каждого ряда."""
    values = subset.to_numpy(dtype=float).T
    rows, per_fold_steps = [], {fold.index: [] for fold in folds}
    for fold in folds:
        model = (GLOBAL_MODELS | EXTRA_GLOBAL)[model_name]()
        if hasattr(model, "set_context"):
            model.set_context(context)
        try:
            model.fit(full, fold.train_end, horizon)
            predicted = model.predict(subset, fold.train_end, horizon)
        except Exception as exc:
            reason = failure_reason(exc)
            print(f"  h={horizon} фолд {fold.index} (обучение {fold.train_end}): ОТКАЗ — {reason[:100]}")
            rows.extend(_failure(model_name, horizon, fold, col, reason) for col in subset.columns)
            continue
        for note in getattr(model, "notes", []):
            print(f"  h={horizon} фолд {fold.index}: {note}")
        for i, col in enumerate(subset.columns):
            y_pred = np.asarray(predicted[i], dtype=float)
            y_test = values[i, fold.test_start : fold.test_end]
            y_train = values[i, : fold.train_end]
            if np.isnan(y_pred).any():
                rows.append(_failure(model_name, horizon, fold, col, "недостаточно истории для признаков"))
                continue
            rows.append(_record(model_name, horizon, fold, col, y_train, y_test, y_pred))
            per_fold_steps[fold.index].append(np.abs(y_test - y_pred))
    return pd.DataFrame(rows), per_fold_steps


def steps_frame(
    model_name: str, horizon: int, folds: list[Fold], per_fold_steps: dict[int, list[np.ndarray]],
) -> pd.DataFrame:
    """Средняя по рядам абсолютная ошибка на каждом шаге горизонта."""
    rows = []
    for fold in folds:
        errors = per_fold_steps.get(fold.index, [])
        if not errors:
            continue
        stacked = np.vstack(errors)
        for step in range(horizon):
            rows.append({
                "model": model_name, "horizon": horizon, "fold": fold.index,
                "train_end": fold.train_end, "step": step + 1,
                "mae": float(np.nanmean(stacked[:, step])), "серий": stacked.shape[0],
            })
    return pd.DataFrame(rows)


def merge_into(
    path: Path, fresh: pd.DataFrame, replace: dict[str, object], *,
    same_panel: bool = False, stamp: str | None = None,
) -> None:
    """Заменяет в файле строки сочетания `replace` (модель × горизонт) строками `fresh`,
    остальное сохраняет.

    Модели гоняются партиями, и перезапись файла оставляла бы только последнюю.
    Заменяемое сочетание передаётся явно, а не выводится из `fresh`: у модели,
    отказавшей на всех фолдах горизонта, кадр шагов пуст, и её старые шаги иначе
    остались бы в файле рядом с одними отказами в файле по рядам.

    Прежние строки читаются через `read_results` (`resolve_results`), а не по
    `path.exists()`: у файла по рядам (`horizons_per_series.csv`) есть снимок `.gz`,
    и на чистом клоне партия из одной пары модель × горизонт должна слиться с ним,
    а не лечь в файл одна — иначе файл остался бы с этой парой в одиночестве, а отчёт
    молча потерял бы остальные пары снимка. У файла шагов (`horizons_steps.csv`)
    снимка нет, и для него `FileNotFoundError` срабатывает ровно как прежний
    `.exists()`: пустой кадр без строк не пишется, кадр с данными ложится первым
    файлом.

    `same_panel` — до записи сверить ряды и номера фолдов партии со всем файлом
    (`src.results_guard.check_same_panel`; сверка по номерам, не по границам обучения
    `train_end` — предел описан там же); при несовпадении файл не меняется.
    Только для файла по рядам: в файле шагов колонки `mo` нет.

    Файл читается точно (`read_results`): строки остальных пар записываются обратно
    теми же числами до последнего разряда.

    `stamp` — снять строки заменяемых пар, уже лежащие в файле, после сверки и до замены:
    `realizations/<модель>__h<горизонт>__<stamp>.csv` рядом с файлом
    (`src.results_guard.save_realization`). main передаёт штамп, один на весь вызов,
    только для файла по рядам: реализации сравниваются по строкам ряд × фолд, и файл
    шагов не снимается.
    """
    try:
        previous = read_results(path)
    except FileNotFoundError:
        if not fresh.empty:
            fresh.to_csv(path, index=False)
        return  # пустой кадр без колонок записался бы файлом, который потом не читается
    if same_panel:
        check_same_panel(previous, fresh, path=path)
    keys = list(replace)
    replaced = {tuple(replace.values())}
    if not fresh.empty:
        # Сочетания, пришедшие в самой партии, заменяются тоже — иначе их строки задвоятся.
        replaced |= set(fresh[keys].itertuples(index=False, name=None))
    is_replaced = pd.MultiIndex.from_frame(previous[keys]).isin(list(replaced))
    if stamp is not None:
        for (model, horizon), rows in previous.loc[is_replaced].groupby(["model", "horizon"]):
            save_realization(rows, path.parent / "realizations", f"{model}__h{int(horizon)}", stamp)
    kept = previous.loc[~is_replaced]
    # Пустые кадры в concat не передаются: pandas предупреждает о выводе типов по ним.
    # Если не осталось ничего, файл всё равно переписывается — одним заголовком,
    # иначе заменяемые строки в нём бы и остались.
    frames = [frame for frame in (kept, fresh) if not frame.empty]
    merged = pd.concat(frames, ignore_index=True) if frames else kept
    merged.to_csv(path, index=False)


def summarise(per_series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сводки по фолдам и по горизонтам. Отказ модели на фолде виден числом отказов.

    Колонки — `src.metrics.SUMMARY_COLUMNS`, как в сводке src/run.py, и той же функцией
    (`src.metrics.r2_and_gains`): R² пул — по всем рядам и фолдам модели на горизонте
    (в пофолдовой — на фолде), выигрыш к Prophet и к наивной — внутри горизонта
    (в пофолдовой — внутри горизонта и фолда). В файле, посчитанном до слагаемых пула,
    колонок r2 и sse нет, и R²-колонки — NaN.

    `фолдов` — плановое число фолдов пары горизонт × модель, `фолдов зачтено` — из скольких
    фолдов её MAE, то есть с успешными строками. Модели, посчитанные внутри горизонта
    на разных парах ряд × фолд, отмечаются строкой `ВНИМАНИЕ:` (`src.metrics.warn_uneven_pairs`)."""
    refusals = refused(per_series)  # правило одно на run.py и этот скрипт
    ok = per_series.loc[~refusals]
    failed = per_series.loc[refusals]
    warn_uneven_pairs(ok)  # на уровне горизонта: пофолдовое сравнение в него входит

    def _summary(group_keys: list[str]) -> pd.DataFrame:
        grouped = ok.groupby(group_keys)
        out = pd.DataFrame({
            "MAE": grouped["mae"].mean(), "MASE": grouped["mase"].mean(),
            "sMAPE": grouped["smape"].mean(), "серий": grouped["mo"].nunique(),
        })
        refusals = failed.groupby(group_keys).size()
        # Модель, отказавшая на всём фолде, в `ok` не встречается вовсе, и без
        # внешнего объединения её строка пропала бы из сводки молча.
        all_keys = per_series[group_keys].drop_duplicates().set_index(group_keys).index
        out = out.reindex(all_keys)
        out["отказов"] = refusals.reindex(all_keys).fillna(0).astype(int)
        out["серий"] = out["серий"].fillna(0).astype(int)
        return out.join(r2_and_gains(ok, out["MAE"]))[SUMMARY_COLUMNS].reset_index()

    by_fold = _summary(["horizon", "model", "fold", "train_end"])
    by_horizon = _summary(["horizon", "model"])
    keys = pd.MultiIndex.from_frame(by_horizon[["horizon", "model"]])
    by_horizon["фолдов"] = by_fold.groupby(["horizon", "model"])["fold"].nunique().reindex(keys).to_numpy()
    # Плановое число прячет отказ целого фолда: у модели, отказавшей на первом фолде h=6,
    # «фолдов» 2, а MAE посчитана по одному.
    counted = by_fold.loc[by_fold["серий"] > 0].groupby(["horizon", "model"])["fold"].nunique()
    by_horizon["фолдов зачтено"] = counted.reindex(keys).fillna(0).astype(int).to_numpy()
    return by_fold.sort_values(["horizon", "fold", "MAE"]), by_horizon.sort_values(["horizon", "MAE"])


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Прогон моделей по пяти горизонтам")
    parser.add_argument("--config", default="configs/horizons.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="переопределить список моделей")
    parser.add_argument("--horizons", nargs="*", type=int, default=None, help="подмножество горизонтов")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    panel = load_panel(ROOT / cfg["data"]["path"])
    wide, report = build_matrix(panel, cfg["data"]["category"], max_gap=cfg["data"]["max_gap"])
    print(report.as_text(), end="\n\n")

    full_panel = wide
    context = build_context(panel, full_panel, cfg)
    wide = sample_series(wide, cfg["sample"]["n_series"], seed=cfg["sample"]["seed"])
    n_obs = wide.shape[0]
    print(f"рядов в прогоне: {wide.shape[1]} | периодов: {n_obs}")

    plan = []
    for item in cfg["horizons"]:
        if args.horizons and item["horizon"] not in args.horizons:
            continue
        folds = rolling_origin(n_obs, item["horizon"], item["n_folds"])
        plan.append((item["horizon"], folds))
        print(f"  горизонт {item['horizon']:2d}: {len(folds)} фолдов, обучение "
              + ", ".join(str(f.train_end) for f in folds))
    print()

    workers = int(cfg.get("compute", {}).get("workers", 0)) or max(1, (os.cpu_count() or 2) - 1)
    known = set(GLOBAL_MODELS) | set(EXTRA_GLOBAL)
    names = args.models if args.models else cfg["models"]

    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg["output"].get("prefix", "horizons")
    per_path = out_dir / f"{prefix}_per_series.csv"
    steps_path = out_dir / f"{prefix}_steps.csv"
    stamp = realization_stamp()  # один на вызов main: снимки всех пар партии — под одним именем
    # Сверка нужна и на чистом клоне, где обычного файла ещё нет, а есть только снимок
    # `.gz` (per_path.exists() его не увидел бы) — read_results находит его сам через
    # resolve_results; FileNotFoundError (нет ни файла, ни снимка) — план сверять не с чем.
    try:
        previous_plan = read_results(per_path)
    except FileNotFoundError:
        previous_plan = pd.DataFrame()
    # Перед каждой записью партия сверяется с файлом ещё раз (merge_into), но там
    # несовпадение всплывает, когда первая пара модель × горизонт уже посчитана.
    plan_folds = {horizon: [f.index for f in folds] for horizon, folds in plan}
    check_plan(previous_plan, wide.columns, plan_folds, path=per_path)

    for name in names:
        for horizon, folds in plan:
            started = time.perf_counter()
            if name in known:
                part, per_fold_steps = evaluate_panel_model(
                    full_panel, wide, name, horizon, folds, context
                )
            else:
                part, per_fold_steps = evaluate_series_models(wide, name, horizon, folds, workers)
            steps = steps_frame(name, horizon, folds, per_fold_steps)
            # Файл по рядам сверяется с партией первым: при несовпадении исключение
            # выходит до записи, и файл шагов тоже остаётся нетронутым.
            pair = {"model": name, "horizon": horizon}
            merge_into(per_path, part, pair, same_panel=True, stamp=stamp)
            merge_into(steps_path, steps, pair)
            is_refusal = refused(part)  # то же правило, что в сводке
            ok = part.loc[~is_refusal]
            by_fold = ok.groupby("fold")["mae"].mean()
            print(f"{name} h={horizon}: MAE {ok['mae'].mean():,.0f} | по фолдам "
                  + ", ".join(f"{v:,.0f}" for v in by_fold)
                  + f" | отказов {int(is_refusal.sum())} | {time.perf_counter() - started:.0f} с")
        print()

    per_series = read_results(per_path)
    by_fold, by_horizon = summarise(per_series)
    by_fold.to_csv(out_dir / f"{prefix}_folds.csv", index=False)
    by_horizon.to_csv(out_dir / f"{prefix}_summary.csv", index=False)

    pd.options.display.float_format = lambda v: f"{v:,.0f}"
    print("\nMAE ПО ГОРИЗОНТАМ (строки — модели)")
    print(by_horizon.pivot(index="model", columns="horizon", values="MAE").to_string())
    print("\nMASE ПО ГОРИЗОНТАМ")
    pd.options.display.float_format = lambda v: f"{v:,.3f}"
    print(by_horizon.pivot(index="model", columns="horizon", values="MASE").to_string())
    pd.options.display.float_format = lambda v: f"{v:,.0f}"
    for horizon in sorted(by_fold["horizon"].unique()):
        sub = by_fold[by_fold["horizon"] == horizon]
        print(f"\nГОРИЗОНТ {horizon}: MAE ПО ФОЛДАМ (в шапке — длина обучения)")
        print(sub.pivot(index="model", columns="train_end", values="MAE").to_string())
    print(f"\nсохранено в {out_dir}: {per_path.name}, {steps_path.name}, "
          f"{prefix}_folds.csv, {prefix}_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
