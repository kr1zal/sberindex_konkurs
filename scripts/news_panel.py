"""Две новостные конструкции на готовых результатах: признаки в модели, корреляция с изломами.

Модели этот скрипт не гоняет — только читает то, что уже посчитано (`src/run.py`,
`scripts/horizons.py`, `scripts/changepoints.py`) и `data/news/monthly.parquet`.

## Конструкция 1 — новостные признаки в модели

`global_gbm_news` отличается от `global_gbm` ровно одним источником признаков —
долями новостных тем региона (`src/models/global_model.py::_load_news`). Сравнение
парное, ряд × фолд, и только там, где обе модели посчитаны (отказ исключает пару
целиком): на рядах, покрытых новостным корпусом (регион ряда есть в корпусе), и на
всех рядах панели. Два горизонта — основной протокол (h=3, `results/per_series.csv`,
3 фолда) и наукаст (h=1, `results/horizons_per_series.csv`, 9 фолдов — самая мощная
из двух проверок).

Разница считается как новости минус база: отрицательная разница значит, что новости
понижают ошибку. Интервал по парам (t, n − 1 пар) не годится для вывода — пары внутри
фолда делят один и тот же тестовый месяц и не независимы, он завышает точность.
Интервал по фолдам (t, фолдов − 1, по средним фолдов) — основная проверка.

## Конструкция 2 — корреляция новостей со структурными изменениями

Единица — регион × месяц: регионы новостного корпуса, представленные в панели
(есть хотя бы один ряд с этим регионом), и месяцы, где офлайновый детектор
(`results/cp_offline_series.csv`, протокол v2, штраф 1,0 — та же картина, что в отчёте)
способен поставить излом при `min_size=3` (см. `MIN_SIZE` ниже). Доля излома —
доля рядов региона (с определённым регионом) с изломом в этом месяце.

Признаки — те же шесть долей тем, что у конструкции 1, лаг 0 (тот же месяц, что
у излома) и лаг 1 (месяц перед изломом) — 12 проверок, порог Бонферрони
`0,05 / 12` объявлен здесь, до расчёта.

**Основной дизайн — `within_month`.** Изломы при штрафе 1,0 сосредоточены в трёх
общих месяцах (см. `results/cp_summary_by_position.csv`), и корреляция в пуле мерила
бы общий календарный рисунок — его уже проверяют событийная (`scripts/news_event_study.py`)
и национальная (`scripts/news_national.py`) конструкции. Вопрос этой конструкции —
региональный: если новостной фон региона в этом месяце необычен на фоне остальных
регионов той же панели, чаще ли там ломаются ряды. Обе величины центрируются средним
по регионам месяца (фиксированный эффект месяца), и после этого остаётся
`n − месяцев − 1` степеней свободы, а не `n − 2`: центрирование по месяцу тратит
степени свободы так же, как фиктивные переменные месяца в LSDV-регрессии — это та же
оценка, только в два шага вместо одной регрессии со всеми фиктивными переменными сразу.
`pooled` — тот же расчёт без центрирования, вторым столбцом для сравнения;
решения по нему не принимается.

    .venv/bin/python -u scripts/news_panel.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cp_bench import month_offset  # noqa: E402
from src.data import build_matrix, load_panel  # noqa: E402
from src.results_guard import read_results, refused  # noqa: E402
from src.run import _series_regions  # noqa: E402

BASE_MODEL = "global_gbm"
NEWS_MODEL = "global_gbm_news"

# ---------------------------------------------------------------------------
# Конструкция 1 — новостные признаки в модели
# ---------------------------------------------------------------------------

NEWS_PANEL_COLUMNS = [
    "horizon", "subset", "n_series", "n_pairs", "n_folds", "mae_base", "mae_news",
    "diff_mean", "pair_ci_low", "pair_ci_high",
    "fold_diff_mean", "fold_ci_low", "fold_ci_high",
    "share_series_improved", "folds_won",
]


def _t_ci(values: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал среднего по t-распределению, df = len(values) − 1."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return mean, mean
    margin = float(stats.t.ppf(1 - (1 - conf) / 2, n - 1) * se)
    return mean - margin, mean + margin


def paired_rows(
    frame: pd.DataFrame, base: str = BASE_MODEL, treatment: str = NEWS_MODEL,
    subset: set[str] | None = None,
) -> pd.DataFrame:
    """Пары (ряд, фолд) с MAE обеих моделей — только там, где обе посчитаны (не отказ).

    `frame` — кадр вида `per_series.csv`/`horizons_per_series.csv` (уже отфильтрованный
    по одному горизонту), колонки `model, fold, mo, mae, error`. Отказ одной модели
    выбрасывает пару целиком: сравнивать с NaN нечего, а частичная пара завысила бы
    точность одной из моделей на тех рядах, где вторая просто не считалась.
    """
    both = frame.loc[frame["model"].isin((base, treatment))].copy()
    both = both.loc[~refused(both)]
    wide = both.pivot_table(index=["mo", "fold"], columns="model", values="mae")
    wide = wide.dropna(subset=[base, treatment])
    if subset is not None:
        wide = wide.loc[wide.index.get_level_values("mo").isin(subset)]
    out = wide.reset_index().rename(columns={base: "mae_base", treatment: "mae_news"})
    out["diff"] = out["mae_news"] - out["mae_base"]
    return out


def pair_stats(pairs: pd.DataFrame, horizon: int, subset_name: str) -> dict:
    """Одна строка `news_panel.csv`: сводка по уже отобранным парам одного горизонта/подмножества."""
    fold_means = pairs.groupby("fold")["diff"].mean()
    by_series = pairs.groupby("mo")[["mae_base", "mae_news"]].mean()
    by_fold = pairs.groupby("fold")[["mae_base", "mae_news"]].mean()
    pair_low, pair_high = _t_ci(pairs["diff"].to_numpy())
    fold_low, fold_high = _t_ci(fold_means.to_numpy())
    return {
        "horizon": horizon, "subset": subset_name,
        "n_series": int(pairs["mo"].nunique()), "n_pairs": int(len(pairs)),
        "n_folds": int(pairs["fold"].nunique()),
        "mae_base": float(pairs["mae_base"].mean()), "mae_news": float(pairs["mae_news"].mean()),
        "diff_mean": float(pairs["diff"].mean()),
        "pair_ci_low": pair_low, "pair_ci_high": pair_high,
        "fold_diff_mean": float(fold_means.mean()),
        "fold_ci_low": fold_low, "fold_ci_high": fold_high,
        "share_series_improved": float((by_series["mae_news"] < by_series["mae_base"]).mean()),
        "folds_won": int((by_fold["mae_news"] < by_fold["mae_base"]).sum()),
    }


def build_news_panel(
    per_series: pd.DataFrame, horizons_per_series: pd.DataFrame, covered_series: set[str],
) -> pd.DataFrame:
    """`results/news_panel.csv`: h=3 из `per_series.csv`, наукаст h=1 из `horizons_per_series.csv`,
    каждый — на покрытых новостями рядах и на всех."""
    sources = {3: per_series, 1: horizons_per_series.loc[horizons_per_series["horizon"] == 1]}
    rows = []
    for horizon, frame in sources.items():
        for subset_name, subset in (("covered", covered_series), ("all", None)):
            pairs = paired_rows(frame, subset=subset)
            rows.append(pair_stats(pairs, horizon, subset_name))
    return pd.DataFrame(rows)[NEWS_PANEL_COLUMNS]


# ---------------------------------------------------------------------------
# Конструкция 2 — корреляция новостей со структурными изменениями
# ---------------------------------------------------------------------------

# Шесть долей тем — те же, что у global_gbm_news (src/models/global_model.py::NEWS_TOPICS).
NEWS_TOPICS = ("t_ceny", "t_dohody", "t_zanjatost", "t_proizvodstvo", "t_kredit", "t_torgovlja")
LAGS = (0, 1)
ALPHA = 0.05
N_TESTS = len(NEWS_TOPICS) * len(LAGS)  # 12
BONFERRONI_ALPHA = ALPHA / N_TESTS

BREAKS_PENALTY = 1.0
BREAKS_PROTOCOL = "v2"

# `min_size` детекторов офлайновой картины (Pelt/Binseg/BottomUp/KernelCPD) — константа
# вызова ruptures в src/changepoints.py, а не протокола: configs/changepoints.yaml её не
# хранит. Точка излома b делит ряд на сегмент [0, b) и сегмент [b, L): чтобы оба были не
# короче min_size, b должно лежать в [min_size, L − min_size] включительно —
# короче с обоих краёв не бывает сегмента.
MIN_SIZE = 3

# Офлайновая картина (scripts/changepoints.py::offline_breaks) детектирует не по самому
# ряду, а по темпу роста (`configs/changepoints.yaml: realtime.mode`) — тот же режим, что
# и у потокового сигнала, иначе offline и realtime отвечали бы на разные вопросы. Темп
# роста короче ряда на одну точку, и `month_offset` (src/cp_bench.py) сдвигает индекс
# излома b обратно к месяцу исходной панели: month = b + offset. Строка не вписана
# руками — берётся из того же места, что и offline_breaks, чтобы расхождение режимов
# не прошло тихо.
OFFLINE_MODE = "ratio"

BREAKS_CORR_COLUMNS = ["design", "feature", "lag", "n", "df", "r", "t", "p", "alpha", "t_threshold", "r_critical", "passed"]


def allowed_break_months(index: pd.PeriodIndex, min_size: int = MIN_SIZE) -> pd.PeriodIndex:
    """Месяцы панели, где офлайновый детектор способен поставить излом.

    Детектор видит темп роста — ряд длиной `L = len(index) − offset` (`offset =
    month_offset(OFFLINE_MODE)`), и на нём излом b возможен только при b in
    [min_size, L − min_size]. Перевод в месяц исходной панели — month = b + offset.
    Нижняя граница (b = min_size) даёт month = min_size + offset — offset добавляется
    без сокращения. Верхняя (b = L − min_size) даёт month = (len(index) − offset)
    − min_size + offset = len(index) − min_size — offset входит и вычитается, поэтому
    верхняя граница от режима подготовки не зависит, а нижняя зависит. При min_size=3
    на 24 месяцах и offset=1 (темп роста) — позиции 4…21, 2023-05…2024-10, 18 месяцев;
    на самом ряду (offset=0) было бы 19, начиная с 2023-04.
    """
    offset = month_offset(OFFLINE_MODE)
    return index[min_size + offset : len(index) - min_size + 1]


def check_breaks_within_range(
    cp_series: pd.DataFrame, allowed: pd.PeriodIndex,
    penalty: float = BREAKS_PENALTY, protocol: str = BREAKS_PROTOCOL,
) -> None:
    """Падает, если в файле есть излом при `penalty` вне `allowed` — сигнал, что окно
    здесь разошлось с детектором в src/changepoints.py."""
    subset = cp_series.loc[(cp_series["penalty"] == penalty) & (cp_series["protocol"] == protocol)]
    months = pd.PeriodIndex(subset["month"], freq="M")
    outside = sorted(set(months[~months.isin(allowed)].astype(str)))
    if outside:
        raise ValueError(
            f"изломы {protocol}/{penalty} вне диапазона [{allowed.min()}, {allowed.max()}]: "
            f"{outside}. Похоже, окно в scripts/news_panel.py разошлось с src/changepoints.py."
        )


def check_window_boundaries_reachable(cp_series: pd.DataFrame, allowed: pd.PeriodIndex) -> None:
    """Падает, если у краёв `allowed` нет ни одного реального излома ни при одном штрафе файла.

    `check_breaks_within_range` ловит только окно, которое УЖЕ слишком узкое (реальный
    излом вне вычисленной границы). Слишком широкое — где граница на самом деле
    недостижима детектором — эта проверка не поймала бы: ни одного излома вне узкого
    окна там и не бывает. Смотрим по всем штрафам файла, а не только `BREAKS_PENALTY`:
    достижимость границы — свойство детектора и `min_size`, а не конкретного штрафа,
    и при штрафе 1,0 крайний месяц мог не встретиться просто по недостатку изломов.
    """
    months = pd.PeriodIndex(cp_series["month"], freq="M")
    missing = [str(m) for m in (allowed.min(), allowed.max()) if not (months == m).any()]
    if missing:
        raise ValueError(
            f"на краю окна [{allowed.min()}, {allowed.max()}] нет ни одного излома ни при "
            f"одном штрафе файла: {missing}. Похоже, окно шире того, что детектор способен "
            "поставить — MIN_SIZE/OFFLINE_MODE в scripts/news_panel.py стоит перепроверить."
        )


def _break_flags(
    cp_series: pd.DataFrame, penalty: float = BREAKS_PENALTY, protocol: str = BREAKS_PROTOCOL,
) -> dict[str, set]:
    """Ряд → множество месяцев излома при данном штрафе."""
    subset = cp_series.loc[(cp_series["penalty"] == penalty) & (cp_series["protocol"] == protocol)]
    months = pd.PeriodIndex(subset["month"], freq="M")
    out: dict[str, set] = {}
    for series_id, month in zip(subset["series_id"], months):
        out.setdefault(series_id, set()).add(month)
    return out


def break_share_table(
    cp_series: pd.DataFrame, regions: dict[str, str], corpus_regions: set[str],
    months: pd.PeriodIndex, penalty: float = BREAKS_PENALTY, protocol: str = BREAKS_PROTOCOL,
) -> pd.DataFrame:
    """Доля рядов региона (с определённым регионом) с изломом в этом месяце. Только регионы,
    представленные и в новостном корпусе, и в панели (`corpus_regions`)."""
    members: dict[str, list[str]] = {}
    for series_id, region in regions.items():
        if region in corpus_regions:
            members.setdefault(region, []).append(series_id)
    flags = _break_flags(cp_series, penalty, protocol)
    rows = []
    for region, series_ids in members.items():
        n = len(series_ids)
        for month in months:
            hits = sum(1 for sid in series_ids if month in flags.get(sid, ()))
            rows.append({"region_name": region, "month": month, "break_share": hits / n, "n_series": n})
    return pd.DataFrame(rows)


def news_lagged_table(
    monthly: pd.DataFrame, regions: list[str], months: pd.PeriodIndex, lags: tuple[int, ...] = LAGS,
) -> pd.DataFrame:
    """Доли тем региона на месяц излома (лаг 0) и на месяц перед ним (лаг 1)."""
    indexed = monthly.set_index(["region_name", "month"])
    rows = []
    for region in regions:
        for month in months:
            row = {"region_name": region, "month": month}
            for lag in lags:
                key = (region, month - lag)
                present = key in indexed.index
                for topic in NEWS_TOPICS:
                    row[f"{topic}_lag{lag}"] = float(indexed.loc[key, topic]) if present else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    if denom == 0:
        return np.nan
    return float((xc * yc).sum() / denom)


def _t_stats(r: float, df: int, alpha: float) -> dict:
    """t, двустороннее p, критический t и критическое |r| (мощность) при данных df."""
    if df <= 0 or not np.isfinite(r) or abs(r) >= 1:
        return {"t": np.nan, "p": np.nan, "t_threshold": np.nan, "r_critical": np.nan}
    t = r * np.sqrt(df) / np.sqrt(1 - r**2)
    p = float(2 * stats.t.sf(abs(t), df))
    t_threshold = float(stats.t.ppf(1 - alpha / 2, df))
    r_critical = float(t_threshold / np.sqrt(df + t_threshold**2))
    return {"t": float(t), "p": p, "t_threshold": t_threshold, "r_critical": r_critical}


def _demean_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Каждое значение минус среднее его группы (тут — месяца)."""
    return values - pd.Series(values).groupby(np.asarray(groups)).transform("mean").to_numpy()


def design_stats(x: np.ndarray, y: np.ndarray, months: np.ndarray, design: str, alpha: float) -> dict:
    """r/df/t/p одного дизайна. `within_month` центрирует обе величины средним по месяцу
    (фиксированный эффект месяца) и теряет на этом `месяцев` степеней свободы вместо одной,
    как у обычного парного `pooled`."""
    n = len(x)
    if design == "within_month":
        x_use, y_use = _demean_by_group(x, months), _demean_by_group(y, months)
        df = n - len(set(months)) - 1
    elif design == "pooled":
        x_use, y_use = x, y
        df = n - 2
    else:
        raise ValueError(f"неизвестный дизайн: {design}")
    r = _pearson(x_use, y_use)
    return {"n": n, "df": df, "r": r, **_t_stats(r, df, alpha)}


def corr_rows(panel: pd.DataFrame, alpha: float = BONFERRONI_ALPHA) -> pd.DataFrame:
    """Строки `news_breaks_corr.csv`: `within_month` (основной) и `pooled` (для сравнения)
    на каждую из 12 пар признак × лаг."""
    rows = []
    for topic in NEWS_TOPICS:
        for lag in LAGS:
            col = f"{topic}_lag{lag}"
            sub = panel.dropna(subset=[col, "break_share"])
            x = sub[col].to_numpy(dtype=float)
            y = sub["break_share"].to_numpy(dtype=float)
            months = sub["month"].to_numpy()

            within = design_stats(x, y, months, "within_month", alpha)
            rows.append({
                "design": "within_month", "feature": topic, "lag": lag, **within, "alpha": alpha,
                "passed": bool(np.isfinite(within["p"]) and within["p"] < alpha),
            })

            pooled = design_stats(x, y, months, "pooled", alpha)
            rows.append({
                "design": "pooled", "feature": topic, "lag": lag, **pooled, "alpha": alpha,
                # Решения по пулу не принимается (см. докстринг модуля) — не False, а пусто,
                # чтобы не выглядело подсчитанным вердиктом.
                "passed": None,
            })
    return pd.DataFrame(rows)[BREAKS_CORR_COLUMNS]


def build_news_breaks_corr(
    cp_series: pd.DataFrame, monthly: pd.DataFrame, regions: dict[str, str], index: pd.PeriodIndex,
) -> pd.DataFrame:
    """Полный конвейер конструкции 2: от офлайновых изломов и новостей до `corr_rows`."""
    allowed = allowed_break_months(index)
    check_breaks_within_range(cp_series, allowed)
    check_window_boundaries_reachable(cp_series, allowed)
    corpus_regions = set(monthly["region_name"].unique()) & set(regions.values())
    breaks = break_share_table(cp_series, regions, corpus_regions, allowed)
    news = news_lagged_table(monthly, sorted(corpus_regions), allowed)
    panel = breaks.merge(news, on=["region_name", "month"], how="left")
    missing = panel[[f"{t}_lag{l}" for t in NEWS_TOPICS for l in LAGS]].isna().to_numpy().sum()
    if missing:
        print(f"  ВНИМАНИЕ: {missing} пропусков в новостных признаках региона × месяца × лага "
              "— ожидалось 0, корпус и панель проверялись как полные")
    return corr_rows(panel)


# ---------------------------------------------------------------------------


def main() -> int:
    warnings.filterwarnings("ignore")
    full_cfg = yaml.safe_load((ROOT / "configs/full.yaml").read_text(encoding="utf-8"))
    horizons_cfg = yaml.safe_load((ROOT / "configs/horizons.yaml").read_text(encoding="utf-8"))

    panel = load_panel(ROOT / full_cfg["data"]["path"])
    wide, _ = build_matrix(panel, full_cfg["data"]["category"], max_gap=full_cfg["data"]["max_gap"])
    index = pd.to_datetime(wide.index).to_period("M")
    regions = _series_regions(wide)

    monthly = pd.read_parquet(ROOT / "data/news/monthly.parquet")
    monthly["month"] = pd.to_datetime(monthly["month"]).dt.to_period("M")
    corpus_regions = set(monthly["region_name"].unique())
    covered_regions = set(regions.values()) & corpus_regions
    covered_series = {sid for sid, region in regions.items() if region in covered_regions}
    print(f"\nрядов с определённым регионом: {len(regions)} из {wide.shape[1]}")
    print(f"регионов в новостном корпусе: {len(corpus_regions)}, общих с панелью: {len(covered_regions)}")
    print(f"рядов, покрытых новостями: {len(covered_series)}\n")

    # --- конструкция 1 ---
    per_series = read_results(ROOT / full_cfg["output"]["dir"] / "per_series.csv")
    horizons_path = ROOT / horizons_cfg["output"]["dir"] / f"{horizons_cfg['output']['prefix']}_per_series.csv"
    horizons_per_series = read_results(horizons_path)
    news_panel = build_news_panel(per_series, horizons_per_series, covered_series)
    out1 = ROOT / "results" / "news_panel.csv"
    news_panel.to_csv(out1, index=False)
    print("=== results/news_panel.csv ===")
    print(news_panel.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"сохранено: {out1.relative_to(ROOT)}\n")

    # --- конструкция 2 ---
    cp_series = read_results(ROOT / "results" / "cp_offline_series.csv")
    breaks_corr = build_news_breaks_corr(cp_series, monthly, regions, index)
    out2 = ROOT / "results" / "news_breaks_corr.csv"
    breaks_corr.to_csv(out2, index=False)
    print("\n=== results/news_breaks_corr.csv ===")
    print(breaks_corr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    within = breaks_corr.loc[breaks_corr["design"] == "within_month"]
    passed = within.loc[within["passed"].fillna(False)]
    print(f"\nпорог Бонферрони {BONFERRONI_ALPHA:.5f}, прошло (within_month): {len(passed)} из {len(within)}")
    if len(passed):
        print(passed.to_string(index=False))
    else:
        best = within.loc[within["p"].idxmin()]
        print(f"лучшее p = {best['p']:.4f} у {best['feature']} (лаг {best['lag']}), "
              f"|r| = {abs(best['r']):.4f} при критическом {best['r_critical']:.4f}")
    print(f"сохранено: {out2.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
