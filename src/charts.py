"""Графики для отчёта и презентации. Один модуль на оба документа, чтобы картинки
не разъехались между артефактами так же, как не расходятся числа.

Палитра взята из проверенного эталона и прогнана валидатором: по трём слотам
пройдены все шесть проверок, включая разделимость при дальтонизме (worst ΔE 9,2)
и порог для обычного зрения (24,0). Бирюзовый даёт контраст к фону 2,74 против
порога 3:1, поэтому везде, где он используется, стоят подписи или рядом есть
таблица с теми же числами.

Два правила, которые здесь соблюдаются намеренно:

* **Никаких двух осей Y.** Доля муниципалитетов с разладкой и ключевая ставка —
  величины разного масштаба, и совмещение их на одной картинке с двумя шкалами
  позволяет подогнать видимую «связь» выбором пределов. Рисуем две панели
  с общей осью времени.
* **Цвет закреплён за сущностью, а не за местом в рейтинге.** Пересортировка
  таблицы не перекрашивает ряды.
"""
from __future__ import annotations

import matplotlib as mpl
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
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)

    ax.plot(months[:15], y[:15], color=INK_SOFT, linewidth=2, label="история обучения")
    # Факт и починенный Prophet почти совпадают — в этом и результат, но одна линия
    # полностью перекрывала другую, и «факт» пропадал с картинки. Факт рисуется шире
    # и ниже по слою, починенный — тоньше поверх, с кольцом цвета фона на маркерах.
    ax.plot(months[14:18], np.r_[y[14], y[15:18]], color=SERIES[0], linewidth=5,
            marker="o", markersize=11, label="факт", zorder=2, alpha=0.95)
    ax.plot(months[14:18], np.r_[y[14], canonical], color=SERIES[1], linewidth=2,
            marker="o", markersize=8, linestyle="--", label="Prophet канонический", zorder=3)
    ax.plot(months[14:18], np.r_[y[14], fixed], color=SERIES[2], linewidth=2,
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
            label="Prophet без годовой сезонности", zorder=4)

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


def model_comparison(summary: pd.DataFrame, baseline: str = "prophet_no_yearly") -> plt.Figure:
    """Горизонтальные столбцы MAE. Планка — опорная линия, а не другой цвет:
    подсветка строки цветом кодировала бы место в рейтинге, а не сущность."""
    s = summary.sort_values("MAE", ascending=True)
    bar = float(s.loc[baseline, "MAE"])
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)

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


def breaks_vs_rate(share: pd.Series, rate: pd.Series) -> plt.Figure:
    """Две панели с общей осью времени — намеренно НЕ две шкалы на одной картинке.

    Совмещение величин разного масштаба на двух осях Y позволяет подогнать видимую
    «связь» выбором пределов, и читатель не может это проверить. Разнесённые панели
    показывают ровно то, что есть: совпадение моментов, а не форму зависимости.
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                                      facecolor=SURFACE, height_ratios=[1, 1])
    x = np.arange(len(share))
    top.bar(x, share.values, color=SERIES[0], width=0.62, zorder=2)
    for i, v in enumerate(share.values):
        if v > 20:
            top.text(i, v + 2, f"{v:.0f}%", ha="center", color=INK, fontsize=9, fontweight="bold")
    _style(top, "Доля муниципалитетов с обнаруженной разладкой", "%")

    bottom.plot(x, rate.values, color=SERIES[1], linewidth=2, marker="o", markersize=5)
    for i, v in enumerate(share.values):
        if v > 20:
            # Вертикаль связывает панели по моменту времени. Цветом сетки она
            # была не видна и своей работы не делала.
            for panel in (top, bottom):
                panel.axvline(i, color=INK_SOFT, linewidth=1, linestyle=":", zorder=1)
    _style(bottom, "Ключевая ставка в реальном выражении, СберИндекс", "%")
    bottom.set_xticks(x[::3])
    bottom.set_xticklabels(share.index[::3], rotation=45, ha="right")
    fig.tight_layout()
    return fig


def series_with_breaks(values: np.ndarray, months, breaks, title: str) -> plt.Figure:
    """Один реальный ряд с отмеченными разладками — метод на настоящих данных."""
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
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
