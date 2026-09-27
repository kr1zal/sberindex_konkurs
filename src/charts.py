"""Графики для отчёта и презентации. Один модуль на оба документа, чтобы картинки
не разъехались между артефактами так же, как не расходятся числа.

Палитра взята из проверенного эталона и прогнана валидатором: по трём слотам
пройдены все шесть проверок, включая разделимость при дальтонизме (worst ΔE 9,2)
и порог для обычного зрения (24,0). Бирюзовый даёт контраст к фону 2,74 против
порога 3:1, поэтому везде, где он используется, стоят подписи или рядом есть
таблица с теми же числами.

Два правила, которые здесь соблюдаются намеренно:

* **Никаких двух осей Y.** Доля МО со структурным изменением и ключевая ставка —
  величины разного масштаба, и совмещение их на одной картинке с двумя шкалами
  позволяет подогнать видимую «связь» выбором пределов. Рисуем две панели
  с общей осью времени.
* **Цвет закреплён за сущностью, а не за местом в рейтинге.** Пересортировка
  таблицы не перекрашивает ряды.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # слоты 1-3 эталонной палитры
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
SURFACE = "#fcfcfb"


def _style(ax, title: str = "", ylabel: str = "") -> None:
    """Рецессивная сетка и оси: данные на переднем плане, разметка позади."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)


def prophet_failure(wide: pd.DataFrame, column: str, canonical, fixed) -> plt.Figure:
    """История обучения, два прогноза и факт. Отрицательный прогноз виден глазом."""
    y = wide[column].to_numpy(float)
    months = wide.index
    fig, ax = plt.subplots(figsize=(8, 3.9), facecolor=SURFACE)

    ax.plot(months[:15], y[:15], color=INK_SOFT, linewidth=2, label="история обучения")
    # Факт и Prophet по умолчанию почти совпадают — в этом и результат, но одна
    # линия полностью перекрывала другую, и «факт» пропадал с картинки. Факт
    # рисуется шире и ниже по слою, дефолтный — тоньше поверх, с кольцом цвета
    # фона на маркерах.
    ax.plot(months[14:18], np.r_[y[14], y[15:18]], color=SERIES[0], linewidth=5,
            marker="o", markersize=11, label="факт", zorder=2, alpha=0.95)
    ax.plot(months[14:18], np.r_[y[14], canonical], color=SERIES[1], linewidth=2,
            marker="o", markersize=8, linestyle="--", label="годовая сезонность включена насильно", zorder=3)
    ax.plot(months[14:18], np.r_[y[14], fixed], color=SERIES[2], linewidth=2,
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
            label="Prophet по умолчанию", zorder=4)

    ax.axhline(0, color=INK_SOFT, linewidth=1)
    worst = int(np.argmin(canonical))
    ax.annotate(f"{canonical[worst]:,.0f} ₽".replace(",", " "),
                xy=(months[15 + worst], canonical[worst]),
                xytext=(10, -6), textcoords="offset points",
                color=INK, fontsize=10, fontweight="bold")
    _style(ax, "Прогноз Prophet на обучении в 15 точек", "руб. на человека в месяц")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT,
              loc="upper left", bbox_to_anchor=(0.0, 0.98))
    fig.tight_layout()
    return fig


def model_comparison(summary: pd.DataFrame, baseline: str = "prophet") -> plt.Figure:
    """Горизонтальные столбцы MAE. Планка — опорная линия, а не другой цвет:
    подсветка строки цветом кодировала бы место в рейтинге, а не сущность.

    Планка — `prophet`, то есть Prophet с его собственными настройками
    по умолчанию: на пятнадцати месяцах он сам не включает годовую сезонность.
    """
    s = summary.sort_values("MAE", ascending=True)
    bar = float(s.loc[baseline, "MAE"])
    # Высота под число строк: моделей стало вдвое больше, и фиксированная высота
    # схлопывала подписи в нечитаемую кашу.
    fig, ax = plt.subplots(figsize=(8, max(4.8, 0.34 * len(s))), facecolor=SURFACE)

    ax.barh(s.index, s["MAE"], color=SERIES[0], height=0.62, zorder=2)
    ax.axvline(bar, color=INK_SOFT, linewidth=1.5, linestyle="--", zorder=3)
    ax.text(bar, len(s) - 0.3, f"  планка {bar:,.0f}".replace(",", " "),
            color=INK, fontsize=9, va="top")
    span = float(max(s["MAE"]))
    for name, value in s["MAE"].items():
        # Отодвигаем подпись, если она попадает на пунктир планки
        offset = span * 0.055 if abs(value - bar) < span * 0.05 else span * 0.012
        ax.text(value + offset, name, f"{value:,.0f}".replace(",", " "),
                va="center", color=INK_SOFT, fontsize=9)

    ax.invert_yaxis()
    ax.set_xlim(0, max(s["MAE"]) * 1.12)
    _style(ax, "Средняя абсолютная ошибка прогноза, полная панель", "")
    ax.set_xlabel("MAE, руб.", color=INK_SOFT, fontsize=9)
    fig.tight_layout()
    return fig


def aggregate_check(actual: pd.Series, check: pd.DataFrame, start: str = "2022-01") -> plt.Figure:
    """Федеральный агрегат: факт, прогноз первого этапа двухэтапной модели от конца панели
    и два ориентира — на все месяцы проверки.

    `actual` — ряд с месячным PeriodIndex, целиком, без обрезки: факт после origin здесь —
    проверка, а не обучение, модель его не видела. `check` — строки одного горизонта из
    `forecast_2025_aggregate_check.csv`. Прогноз сплошной, как факт: сравнивается он с фактом,
    а ориентиры — фон, поэтому штриховые. В подписях легенды — средняя абсолютная ошибка
    по месяцам, те же числа, что в таблице под картинкой.
    """
    months = pd.PeriodIndex(sorted(check["month"].unique()), freq="M")
    origin = months[0] - 1
    shown = actual.loc[pd.Period(start, "M"): months[-1]]
    fig, ax = plt.subplots(figsize=(8, 3.9), facecolor=SURFACE)

    ax.plot(shown.index.to_timestamp(), shown.to_numpy(float), color=INK_SOFT, linewidth=2,
            label="факт", zorder=2)
    # Факт после origin — точками поверх той же линии: видно, где кончается то, на чём
    # учились, и начинается то, чем проверяют.
    checked = shown.loc[months[0]:]
    ax.plot(checked.index.to_timestamp(), checked.to_numpy(float), linestyle="none", marker="o",
            markersize=5, color=INK_SOFT, zorder=3)

    # Точки — только у факта после origin: маркеры с кольцом на линии прогноза рвали её
    # в пунктир, и сплошной прогноз читался как ещё один штриховой ориентир.
    styles = {
        "two_stage": (SERIES[0], "-", "двухэтапная, первый этап"),
        "naive": (SERIES[1], "--", "как в декабре"),
        "seasonal_naive": (SERIES[2], "--", "как год назад"),
    }
    for method, (color, line, label) in styles.items():
        part = check.loc[check["method"] == method].sort_values("month")
        if part.empty:
            continue
        x = [origin.to_timestamp()] + list(pd.PeriodIndex(part["month"], freq="M").to_timestamp())
        y = [float(actual.loc[origin])] + list(part["forecast"].astype(float))
        mean_error = f"{part['error_pct'].abs().mean():.1f}".replace(".", ",")
        ax.plot(x, y, color=color, linestyle=line, linewidth=2,
                label=f"{label}: средняя ошибка {mean_error}%",
                zorder=4 if method == "two_stage" else 3)

    ax.axvline(origin.to_timestamp(), color=INK_SOFT, linewidth=1, linestyle=":", zorder=1)
    ax.annotate("конец панели", xy=(origin.to_timestamp(), 1.0), xycoords=("data", "axes fraction"),
                xytext=(-4, -2), textcoords="offset points", ha="right", va="top",
                color=INK_SOFT, fontsize=8)
    _style(ax, "", "млрд руб. в месяц")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper left")
    fig.tight_layout()
    return fig


def breaks_vs_rate(shares: dict[str, pd.Series], rate: pd.Series, threshold: float,
                   marks: Sequence[str] = ()) -> plt.Figure:
    """Доли МО по месяцам — по панели на каждую картину — и ставка под ними, с общей осью времени.

    Намеренно НЕ две шкалы на одной картинке: совмещение величин разного масштаба на двух
    осях Y позволяет подогнать видимую «связь» выбором пределов, и читатель не может это
    проверить. Разнесённые панели показывают ровно то, что есть: совпадение моментов,
    а не форму зависимости.

    Все доли — на одной шкале от нуля до ста, со штриховой линией порога события: картина,
    где доля не поднимается выше десяти процентов, при своей шкале растянулась бы на всю
    высоту панели и читалась бы как всплеск. Пунктирные вертикали — месяцы событий `marks`:
    по ним видно, на сколько сигнал отстаёт от события. Подписаны столбцы выше двадцати
    процентов и наибольший в каждой панели.
    """
    months = next(iter(shares.values())).index
    n = len(shares)
    fig, axes = plt.subplots(n + 1, 1, figsize=(8, 1.55 * n + 2.2), sharex=True,
                             facecolor=SURFACE, height_ratios=[1] * n + [1.2])
    x = np.arange(len(months))
    for k, (ax, (title, share)) in enumerate(zip(axes, shares.items())):
        values = share.reindex(months).to_numpy(dtype=float)
        ax.bar(x, values, color=SERIES[0], width=0.62, zorder=2)
        top = int(np.nanargmax(values)) if np.isfinite(values).any() else -1
        for i, v in enumerate(values):
            if v > 20 or (i == top and v > 0):
                text = f"{v:.0f}%" if v >= 10 else f"{v:.1f}%".replace(".", ",")
                ax.text(i, v + 3, text, ha="center", color=INK, fontsize=8, fontweight="bold")
        ax.axhline(threshold, color=INK_SOFT, linewidth=1, linestyle="--", zorder=1)
        if k == 0:
            ax.text(-0.3, threshold + 3, f"порог {threshold:g}%", ha="left", color=INK_SOFT, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, threshold, 100])
        _style(ax, "", "% МО")
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)

    bottom = axes[-1]
    bottom.plot(x, rate.values, color=SERIES[1], linewidth=2, marker="o", markersize=5)
    _style(bottom, "", "%")
    bottom.set_title("Ключевая ставка в реальном выражении, СберИндекс", color=INK, fontsize=10,
                     loc="left", pad=8)
    # Вертикаль связывает панели по моменту времени. Цветом сетки она была не видна
    # и своей работы не делала.
    position = {month: i for i, month in enumerate(months)}
    for month in marks:
        if month in position:
            for panel in axes:
                panel.axvline(position[month], color=INK_SOFT, linewidth=1, linestyle=":", zorder=1)
    bottom.set_xticks(x[::3])
    bottom.set_xticklabels(months[::3], rotation=45, ha="right")
    fig.tight_layout()
    return fig


def series_with_breaks(values: np.ndarray, months, breaks, title: str) -> plt.Figure:
    """Один реальный ряд с отмеченными структурными изменениями — метод на настоящих данных."""
    fig, ax = plt.subplots(figsize=(8, 3.3), facecolor=SURFACE)
    ax.plot(months, values, color=SERIES[0], linewidth=2, marker="o", markersize=5)
    for b in breaks:
        if 0 <= b < len(months):
            ax.axvline(months[b], color=SERIES[1], linewidth=2, linestyle="--", zorder=1)
            ax.text(months[b], ax.get_ylim()[1], f" {months[b]:%Y-%m}",
                    color=INK, fontsize=9, va="top")
    _style(ax, title, "руб. на человека в месяц")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig
