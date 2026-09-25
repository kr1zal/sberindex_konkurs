"""Бэкфилл R²: слагаемые пула для результатов, посчитанных до их появления.

**Зачем.** Задача конкурса — превзойти эталон по MAE и R². R² сводок считается
по объединённому пулу тестовых точек (`src/metrics.py::r2_and_gains`) из сумм,
которые хранит каждая строка ряд × фолд: sse, sst, n, y_sum, y_sq. Прогоны пишут
их сами, начиная с изменения, где появились сводки с R². `results/per_series.csv`
посчитан раньше, в нём только r2 ряда. Перегонять ради сумм 29 моделей — часы
счёта, а для уже посчитанных строк они выводятся без единого прогона модели.

**Что делает.**

1. Строит панель тем же путём, что `src/run.py`: `load_panel`, `build_matrix`
   с категорией и `max_gap`, `sample_series` с `n_series`, фолды — `rolling_origin`
   с `horizon` и `n_folds`. Всё из конфига (`--config`, по умолчанию `configs/full.yaml`).
2. Для каждой успешной строки берёт тест её ряда и фолда из панели: n, y_sum, y_sq,
   sst — по фактическим значениям той же функцией, что в прогонах
   (`src.metrics.target_sums`); sse = (1 − r2) · sst. При r2 = NaN (тест ряда ровный,
   SST ноль) sse не восстановить: он NaN, число таких строк печатается, в пул они
   не входят. Строки отказа — NaN во всех пяти.
3. Проверяет себя на моделях с прогнозом, известным без прогона: `naive_last` —
   последнее значение обучения на все шаги, `seasonal_naive` — значение сезоном
   раньше, как в `src/models/naive.py`. SSE по этим прогнозам, посчитанный напрямую,
   должен совпасть с выведенным с относительной точностью 1e−6 в каждой строке.
   Границ фолдов в файле нет, только номера: что тест взят из того же окна,
   что у прогона, подтверждает именно эта проверка.
4. Сверяет инварианты: строк столько же, ряды, фолды, mae, mase, smape, r2 и порядок
   строк те же, добавлены только новые колонки.
5. Пересчитывает `results/summary.csv` (`src.run.summarise`), `horizons_summary.csv`
   и `horizons_folds.csv` (`scripts.horizons.summarise`, пути — из `--horizons-config`);
   прежние колонки сводок должны совпасть с тем, что лежит в файлах. В
   `horizons_per_series.csv` нет ни r2, ни слагаемых — `scripts/horizons.py` писал только
   MAE, MASE и sMAPE, — поэтому R²-колонки сводок горизонтов остаются NaN до перепрогона
   `scripts/horizons.py`. Выигрыш к Prophet и к наивной считается по MAE и заполняется
   сразу.
6. Копирует четыре файла в `results/backup_2026-09-25/` и только потом пишет их.
   Копию, сделанную раньше, не перезаписывает: повторный запуск иначе заменил бы
   исходные файлы уже переписанными.

Любое расхождение останавливает скрипт до записи. `--dry-run` — все расчёты
и проверки и печать будущей сводки, без записи и без копии.

**Идемпотентен:** считает только по панели и колонке r2, свои прошлые выходы
в расчёт не берёт (прежние сводки читаются лишь для сверки), и повторный запуск
даёт те же байты. **Разовый:** после него прогоны
пишут слагаемые сами, и прямой sse точнее выведенного — у ряда с ровным тестом
он есть, у выведенного NaN. На файл, куда уже дописаны новые прогоны, бэкфилл
не запускать.

    .venv/bin/python -u scripts/backfill_r2.py [--config configs/full.yaml]
        [--horizons-config configs/horizons.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import horizons  # noqa: E402
from src.data import build_matrix, load_panel, sample_series  # noqa: E402
from src.metrics import GAINS, R2_MEDIAN, R2_POOL, target_sums  # noqa: E402
from src.results_guard import refused  # noqa: E402
from src.run import REGISTRY, summarise  # noqa: E402
from src.split import Fold, rolling_origin  # noqa: E402

# Слагаемые пула в порядке строк прогона (`src.metrics.ROW_METRICS`).
PARTS = ("sse", "sst", "n", "y_sum", "y_sq")
BACKUP = "backup_2026-09-25"
TOLERANCE = 1e-6
KNOWN = ("naive_last", "seasonal_naive")
SEASON = REGISTRY["seasonal_naive"]().season
# Разбор чисел CSV по умолчанию в pandas не обратим: примерно каждое восьмое число читается
# на единицу последнего разряда не тем, что записано. Файлы, которые run.py пишет из памяти
# (per_series.csv, summary.csv), и прежние сводки читаются точно: иначе переписанный файл
# сменил бы последние разряды прежних колонок, а сверка сводок споткнулась бы о шум разбора.
EXACT = {"float_precision": "round_trip"}
FORMATS = {
    "MAE": "{:,.1f}", "MASE": "{:.4f}", "sMAPE": "{:.3f}", R2_POOL: "{:.4f}", R2_MEDIAN: "{:.4f}",
    **{column: "{:+.2f}" for column in GAINS},
}


def known_forecast(model: str, series: np.ndarray, fold: Fold, season: int) -> np.ndarray:
    """Прогноз, известный без прогона, — повтор `src/models/naive.py`.

    `naive_last` — последнее значение обучения на все шаги. `seasonal_naive` — шаг i
    берёт y[train_end − season + i mod season]: при горизонте не длиннее сезона это тот
    же месяц сезоном раньше. Обучение короче сезона — последнее значение, как
    в `SeasonalNaive.predict`."""
    train = series[: fold.train_end]
    if model == "naive_last" or len(train) < season:
        return np.full(fold.horizon, float(train[-1]))
    return train[len(train) - season + np.arange(fold.horizon) % season].astype(float)


def direct_sse(model: str, series: np.ndarray, fold: Fold) -> float:
    """SSE по известному прогнозу — напрямую, без r2."""
    y_test = series[fold.test_start : fold.test_end]
    return float(np.sum((y_test - known_forecast(model, series, fold, SEASON)) ** 2))


def fill_parts(per_series: pd.DataFrame, wide: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Копия `per_series` со слагаемыми пула, выведенными из панели и колонки r2.

    Слагаемые, которые уже есть в кадре, не читаются: всё считается заново, поэтому
    повторный запуск даёт тот же кадр. Колонки встают перед `error`, как в строках
    прогона. Ряд или фолд, которых нет в панели, — ошибка: суммы по другой панели
    были бы молча неверны."""
    base = per_series.drop(columns=[column for column in PARTS if column in per_series.columns])
    by_index = {fold.index: fold for fold in folds}
    problems = []
    if missing := sorted(set(base["mo"]) - set(wide.columns)):
        problems.append(f"рядов нет в панели: {len(missing)} ({', '.join(map(str, missing[:5]))})")
    if missing := sorted(set(base["fold"]) - set(by_index)):
        problems.append(f"фолды {missing} не входят в фолды конфига {sorted(by_index)}")
    if problems:
        raise ValueError(
            "файл результатов посчитан не на панели из конфига: " + "; ".join(problems)
            + ". В файл ничего не записано."
        )

    pairs = base[["mo", "fold"]].drop_duplicates()
    values = {mo: wide[mo].to_numpy(dtype=float) for mo in pairs["mo"].unique()}
    sums = pd.DataFrame(
        [target_sums(values[mo][by_index[f].test_start : by_index[f].test_end])
         for mo, f in zip(pairs["mo"], pairs["fold"])],
        index=pd.MultiIndex.from_frame(pairs),
    )
    parts = sums.reindex(pd.MultiIndex.from_frame(base[["mo", "fold"]])).set_axis(base.index)
    parts["sse"] = (1.0 - base["r2"]) * parts["sst"]
    parts = parts[list(PARTS)].where(~refused(base), axis=0)

    columns = list(base.columns)
    at = columns.index("error") if "error" in columns else len(columns)
    return pd.concat([base.iloc[:, :at], parts, base.iloc[:, at:]], axis=1)


def check_known(filled: pd.DataFrame, wide: pd.DataFrame, folds: list[Fold]) -> float:
    """Сверка выведенного sse с SSE по известному прогнозу, строка за строкой.

    Печатает число сверенных строк и наибольшее относительное расхождение по каждой
    модели и падает, если хоть в одной строке оно больше `TOLERANCE`. Строки с NaN sse
    (ровный тест) сверять не с чем — они посчитаны отдельно."""
    by_index = {fold.index: fold for fold in folds}
    usable = ~refused(filled) & filled["sse"].notna()
    worst, worst_row = 0.0, None
    for model in KNOWN:
        rows = filled.loc[usable & (filled["model"] == model)]
        if rows.empty:
            print(f"  {model}: строк для сверки нет")
            continue
        direct = np.array([
            direct_sse(model, wide[mo].to_numpy(dtype=float), by_index[f])
            for mo, f in zip(rows["mo"], rows["fold"])
        ])
        derived = rows["sse"].to_numpy(dtype=float)
        scale = np.maximum(np.abs(direct), np.abs(derived))
        deviation = np.divide(np.abs(derived - direct), scale, out=np.zeros_like(scale), where=scale > 0)
        at = int(np.argmax(deviation))
        print(f"  {model}: сверено {len(rows)} строк, максимальное расхождение {deviation[at]:.2e}")
        if deviation[at] >= worst:
            row = rows.iloc[at]
            worst, worst_row = float(deviation[at]), (model, row["mo"], row["fold"], derived[at], direct[at])
    if worst_row is None:
        raise ValueError(
            f"проверять нечем: в файле нет успешных строк {', '.join(KNOWN)} с sse. "
            "В файл ничего не записано."
        )
    if worst > TOLERANCE:
        model, mo, fold, derived, direct = worst_row
        raise ValueError(
            f"выведенный sse расходится с SSE известного прогноза на {worst:.2e} "
            f"(допуск {TOLERANCE:.0e}): {model}, ряд {mo}, фолд {fold} — {derived!r} "
            f"против {direct!r}. r2 в файле не сходится с панелью и фолдами конфига; "
            "в файл ничего не записано."
        )
    return worst


def check_invariants(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Бэкфилл только дописывает колонки. Любое другое изменение — ошибка до записи."""
    base = [column for column in before.columns if column not in PARTS]
    problems = []
    if len(after) != len(before):
        problems.append(f"строк было {len(before)}, стало {len(after)}")
    if [column for column in after.columns if column not in PARTS] != base:
        problems.append("изменился набор или порядок прежних колонок")
    if set(after.columns) != set(base) | set(PARTS):
        problems.append(f"колонки: {sorted(set(after.columns) ^ (set(base) | set(PARTS)))}")
    if not problems:
        if set(after["mo"]) != set(before["mo"]):
            problems.append("другое множество рядов")
        if set(after["fold"]) != set(before["fold"]):
            problems.append("другое множество фолдов")
        # equals сравнивает и порядок строк, и NaN на тех же местах.
        problems.extend(
            f"изменились значения или порядок строк в колонке {column}"
            for column in base if not after[column].equals(before[column])
        )
    if problems:
        raise ValueError("бэкфилл изменил бы прежние данные: " + "; ".join(problems)
                         + ". В файл ничего не записано.")


def check_same(old: pd.DataFrame, new: pd.DataFrame, path: Path) -> None:
    """Прежние колонки сводки `path` не изменились: бэкфилл сводкам только добавляет
    колонки. Оба кадра — с ключами сводки в индексе."""
    problems = []
    if missing := [column for column in old.columns if column not in new.columns]:
        problems.append(f"в новой сводке нет колонок {missing}")
    if set(old.index) != set(new.index):
        problems.append(f"строк было {len(old)}, стало {len(new)}, ключи различаются")
    if not problems:
        aligned = new.loc[old.index]
        for column in old.columns:
            a = old[column].to_numpy(dtype=float)
            b = aligned[column].to_numpy(dtype=float)
            if not ((a == b) | (np.isnan(a) & np.isnan(b))).all():
                problems.append(f"изменилась колонка {column}")
    if problems:
        raise ValueError(f"{path}: пересчитанная сводка не сходится с прежней — "
                         + "; ".join(problems) + ". В файлы ничего не записано.")


def backup(paths: list[Path], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = folder / path.name
        if target.exists():
            print(f"  {shown(target)}: копия уже есть, не перезаписана")
            continue
        shutil.copy2(path, target)
        print(f"  {shown(target)}: скопирован")


def shown(path: Path) -> str:
    """Путь от корня репозитория, если файл внутри него."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def show(table: pd.DataFrame) -> str:
    formats = {column: fmt.format for column, fmt in FORMATS.items() if column in table.columns}
    return table.to_string(formatters=formats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Слагаемые R² пула для результатов, посчитанных до них")
    parser.add_argument("--config", default="configs/full.yaml",
                        help="протокол, на котором посчитан per_series.csv, по умолчанию %(default)s")
    parser.add_argument("--horizons-config", default="configs/horizons.yaml",
                        help="откуда взять каталог и префикс файлов горизонтов, по умолчанию %(default)s")
    parser.add_argument("--dry-run", action="store_true",
                        help="все расчёты и проверки, печать будущей сводки, без записи")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hcfg = yaml.safe_load(Path(args.horizons_config).read_text(encoding="utf-8"))

    panel = load_panel(ROOT / cfg["data"]["path"])
    wide, report = build_matrix(panel, cfg["data"]["category"], max_gap=cfg["data"]["max_gap"])
    print(report.as_text(), end="\n\n")
    wide = sample_series(wide, cfg["sample"]["n_series"], seed=cfg["sample"]["seed"])
    folds = rolling_origin(wide.shape[0], cfg["split"]["horizon"], cfg["split"]["n_folds"])
    print(f"рядов в панели: {wide.shape[1]} | периодов: {wide.shape[0]}")
    for f in folds:
        print(f"  фолд {f.index}: обучение 1..{f.train_end}, тест {f.test_start + 1}..{f.test_end}")

    out_dir = ROOT / cfg["output"]["dir"]
    per_path, summary_path = out_dir / "per_series.csv", out_dir / "summary.csv"
    before = pd.read_csv(per_path, **EXACT)
    failed = refused(before)
    print(f"\n{shown(per_path)}: строк {len(before)}, моделей {before['model'].nunique()}, "
          f"рядов {before['mo'].nunique()}, фолды {sorted(before['fold'].unique().tolist())}, "
          f"отказов {int(failed.sum())}")

    filled = fill_parts(before, wide, folds)
    unknown = int((~failed & filled["sse"].isna()).sum())
    print(f"строк с NaN sse: {unknown} — успешные строки с r2 = NaN (тест ряда ровный, SST ноль); "
          f"у строк отказа все пять слагаемых — NaN")
    print(f"проверка на известных прогнозах (допуск {TOLERANCE:.0e}):")
    worst = check_known(filled, wide, folds)
    print(f"  максимальное расхождение по обеим моделям: {worst:.2e}")
    check_invariants(before, filled)
    print(f"инварианты: строк {len(filled)}, ряды, фолды, mae, mase, smape, r2 и порядок строк те же; "
          f"добавлены колонки {', '.join(PARTS)}")

    summary = summarise(filled)
    if summary_path.exists():
        check_same(pd.read_csv(summary_path, index_col=0, **EXACT), summary, summary_path)
        print(f"{shown(summary_path)}: прежние колонки совпадают ({len(summary)} моделей)")

    h_dir = ROOT / hcfg["output"]["dir"]
    prefix = hcfg["output"].get("prefix", "horizons")
    h_per_path = h_dir / f"{prefix}_per_series.csv"
    h_paths = {"summary": h_dir / f"{prefix}_summary.csv", "folds": h_dir / f"{prefix}_folds.csv"}
    h_tables, missing = {}, []
    if h_per_path.exists():
        # Разбор по умолчанию — как в scripts/horizons.py::main: сводка горизонтов
        # пересчитывается из тех же чисел, из которых её посчитал прогон.
        h_per = pd.read_csv(h_per_path)
        missing = [column for column in ("r2", *PARTS) if column not in h_per.columns]
        by_fold, by_horizon = horizons.summarise(h_per)
        h_tables = {"summary": (by_horizon, ["horizon", "model"]),
                    "folds": (by_fold, ["horizon", "model", "fold", "train_end"])}
        print(f"\n{shown(h_per_path)}: строк {len(h_per)}"
              + (f"; колонок {', '.join(missing)} в файле нет — R²-колонки сводок горизонтов "
                 "останутся NaN до перепрогона scripts/horizons.py" if missing else ""))
        for name, (table, keys) in h_tables.items():
            path = h_paths[name]
            if path.exists():
                check_same(pd.read_csv(path, **EXACT).set_index(keys), table.set_index(keys), path)
            filled_r2 = int(table[R2_POOL].notna().sum())
            print(f"{shown(path)}: прежние колонки совпадают; R² пул заполнен "
                  f"в {filled_r2} строках из {len(table)}, выигрыш к Prophet и к наивной — по MAE")
    else:
        print(f"\n{shown(h_per_path)} нет — сводки горизонтов не пересчитываются")

    print(f"\nСВОДКА ({shown(summary_path)})")
    print(show(summary))
    if h_tables:
        print(f"\nСВОДКА ГОРИЗОНТОВ ({shown(h_paths['summary'])}"
              + ("; R²-колонки — NaN до перепрогона scripts/horizons.py)" if missing else ")"))
        print(show(h_tables["summary"][0].set_index(["horizon", "model"])))

    if args.dry_run:
        print("\n--dry-run: ничего не записано и не скопировано")
        return 0

    targets = [per_path, summary_path, *(h_paths[name] for name in h_tables)]
    print(f"\nкопия перед записью в {shown(out_dir / BACKUP)}:")
    backup([path for path in targets if path.exists()], out_dir / BACKUP)
    filled.to_csv(per_path, index=False)
    summary.to_csv(summary_path)
    for name, (table, _) in h_tables.items():
        table.to_csv(h_paths[name], index=False)
    print("записано: " + ", ".join(shown(path) for path in targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
