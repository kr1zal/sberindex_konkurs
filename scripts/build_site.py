"""Единственный источник главной страницы (`index.html`) и данных демонстрационного
стенда (`demo/data/*.json`) для GitHub Pages.

    .venv/bin/python scripts/build_site.py [--out DIR]

Пишет `<DIR>/index.html` из `site/index.template.html` (`--out` по умолчанию — корень
репозитория; тесты собирают во временный каталог) и `<DIR>/demo/data/**`. Сам стенд
(`demo/index.html`, `demo/demo.js`, графики) и скрипт главной (`site/landing.js`) —
рукописные и читают только эти данные.

## Пять чисел главной и фраза об изломах

Резюме отчёта (`report/report.qmd`) объясняет тот же прогон теми же файлами. Отсюда
перенесена ЛОГИКА, а не результат: `compute_placeholders` пересчитывает её на текущих
`results/*.csv`, и при другом прогоне подставит другие, но так же верные числа.
Помощники форматирования (`rub`, `num`, `on_folds`, `plural`, `and_join`, `in_words_m`)
и словарь русских названий моделей (`MODEL_LABELS`) — те же самые, что в отчёте
(`report/report.qmd`, строки 42-102 и 1682-1692), с тем же поведением.

Фраза раздела «Изломы» — пункт резюме отчёта об обнаружении точек структурных изменений
(`report/report.qmd`, ~строки 545-1060): `changepoint_claim` пересчитывает его условия на
`results/cp_*.csv` и пишет ту же фразу без ссылки «[об обнаружении]». Совпадение с отрисованным
резюме отчёта проверяет тест.

## Формат данных стенда — контракт с `demo/demo.js`

Все значения — JSON без NaN (пропуск — ``null``), рубли — целыми, проценты и
млрд руб. — с двумя знаками, месяцы — строки ``"YYYY-MM"``. Файлы компактные
(``separators=(",", ":")``, ``ensure_ascii=False``).

``demo/data/index.json``::

    built                дата сборки, "YYYY-MM-DD"
    unit                 "руб. на человека в месяц"
    origin               последний месяц панели (configs/forecast_forward.yaml::origin)
    panel_months         месяцы панели, индекс матрицы (24 штуки)
    forecast_months      месяцы прогноза: origin+1 .. origin+max(horizons)
    n_series             рядов всего (столбцов матрицы)
    n_no_region          рядов без региона (region пуст в forecast_2025.csv)
    n_homonym_names      названий (series_id без " #N"), у которых больше одного ряда
    n_homonym_series     рядов, относящихся к таким названиям
    n_hash_names         названий среди рядов с суффиксом " #N" в series_id, уникальных
    n_hash_series        рядов с таким суффиксом — подсказка поиска говорит про них
                         «их различает номер после «#»», и это не то же самое, что
                         n_homonym_series: у части омонимов вторая копия выпала из
                         панели по пропускам, оставшийся ряд с суффиксом "#N" остался
                         без пары, и счёт по имени (n_homonym_*) такой ряд омонимом
                         больше не считает
    default_mo           МО, которое стенд открывает по умолчанию
    quick                [{id, short}, …] — быстрые кнопки под полем поиска: ряд (series_id) и
                         короткая подпись кнопки, порядок как у STAND_QUICK_MO
    forecast_rule        [{from, to, model, horizon, label}, …] — отрезки месяцев
                         прогноза подряд с одной моделью и горизонтом; label — как
                         в models[].label (или сырой model, если такой модели там нет)
    known_model          модель пунктира (uses_published_aggregate), одна на все горизонты
    breaks               {protocol, penalty, detector, mode} — источник изломов ниже
    folds                [{fold, train_months, test_from, test_to}, …] — фолды
                         основного протокола (configs/full.yaml, src.split.rolling_origin)
    models               [{id, role, label}, …] — модели таблицы ошибок, см. ниже
    panel_mae            {model_id: [fold0, fold1, fold2, среднее]}, руб.
    series               [[series_id, регион|null, ОКТМО|null, номер файла mo/], …]; у рядов без региона
                         пятым элементом — средние расходы за последние MEAN_MONTHS месяцев панели,
                         тыс. руб. на человека в месяц, готовой строкой ``num(v, 1)``: одноимённые
                         ряды («Михайловский муниципальный район #1 … #4») подсказка поиска иначе
                         отличить не может, а «#N» — лишь порядок в выгрузке

``demo/data/mo/<номер>.json`` — ряды одного региона (номер — позиция региона
в отсортированном списке уникальных регионов; ряды без региона — под отдельным,
следующим по счёту номером)::

    {series_id: {fact: [24], forecast: [12], known: [12], breaks: [месяцы],
                 mae: {model_id: [fold0, fold1, fold2]}}, …}

``demo/data/aggregate.json``::

    unit, origin, horizon, model             — как в forecast_2025_aggregate_check.csv
    model_label                             — название model, как в отчёте (_fc_agg_ru)
    history: {months, values}               — агрегат с AGGREGATE_CHART_START по origin
                                               включительно (как на графике отчёта,
                                               src/charts.py::aggregate_check)
    check: {months, actual, actual_rub,
            forecast: {two_stage, naive, seasonal_naive}, forecast_rub: {two_stage},
            error_pct: {…те же ключи, что forecast}}  — горизонт = max(forecast_forward.yaml::recommended)
    rule_names: {naive, seasonal_naive}     — из колонки aggregate_model
    mape: {method: …}                       — средняя |ошибка| за check.months

    ``error_pct``, ``mape``, ``actual_rub`` и ``forecast_rub`` — уже готовые строки,
    отформатированные функциями этого модуля (``num(v, 1, sign=True)``, ``num(v, 1)``,
    ``rub(v)``), не числа: стенд показывает их как есть. Округлять на JS ещё раз
    (`Math.round` после округления питоном) нельзя — у отчёта и стенда тогда может
    разойтись последний знак. Числа графика (``history``, ``check.actual``,
    ``check.forecast``) остаются числами — по ним строятся линии; рублёвые строки
    (``actual_rub``, ``forecast_rub``) — те же значения текстом, только для рублёвых
    колонок таблицы стенда.

Роли ``models`` (в этом порядке): ``prophet`` — reference;
``naive_last`` — naive; recommended-модель самого короткого горизонта
`forecast_forward.yaml` — recommended; ``summary.csv`` ``MAE.idxmin()`` — best_mean;
recommended-модель самого длинного горизонта — two_stage. Если две роли достаются
одной модели — она входит одной строкой, ``role`` — их имена через ``+``.

Откуда что: ряды — `forecast_2025.csv` (их множество сверяется со столбцами матрицы,
иначе сборка падает); факт — `build_matrix`; прогноз/пунктир — строки
`recommended`/`uses_published_aggregate`, месяц берётся с самого короткого горизонта,
который его покрывает (`report/report.qmd::_fc_steps`); изломы — `cp_offline_series.csv`
при протоколе `v{changepoints.yaml::protocol_version}` и штрафе
`scripts/news_event_study.py::PENALTY` (модуль грузится по пути, как в отчёте) — так
отчёт выбирает офлайновую картину; фолды — `src.split.rolling_origin` на
`configs/full.yaml`; MAE — `per_series.csv` без отказов (`src.results_guard.refused`).

## Данные главной — контракт с `site/landing.js`

Лежат в `index.html` прямо в странице, `<script type="application/json" id="landing-data">`
(подстановка `${landing_json}`; `<` экранирован как `\\u003c`, чтобы данные не закрыли
тег): страница работает без сети и без лишнего запроса. Формат, как у данных стенда:
пропуск — ``null``, месяцы — ``"YYYY-MM"``, значения рядов — до трёх знаков.

    story     месяцы панели, 30 рядов для обложки и раздела «Почему это прогноз одного числа»
              и медиана по всем рядам матрицы:
              {months, base_months, ids, series, median, n_total, n_sample}.
              series[j] — ряд ids[j], делённый на среднее его первых base_months месяцев;
              median[t] — медиана таких значений по ВСЕМ n_total рядам матрицы в месяце t.
              Ряды выбраны сидом LANDING_SAMPLE_SEED.
    horizons  {list, main, labels, unit, models, notes}: горизонты из configs/horizons.yaml,
              main — горизонт основного протокола (его показывают первым), четыре модели
              в порядке prophet, naive_last, лучшая основного протокола (summary.csv
              MAE.idxmin()), two_stage — [{id, role, label, note, mae}], mae[i] — MAE
              на горизонте list[i], руб., по horizons_summary.csv (нет — null);
              notes[i] — пояснение под полосами на горизонте list[i], с числом фолдов этого
              горизонта (`horizon_folds`).
    fact      то же, что demo/data/aggregate.json (одна и та же структура, не пересчёт).
    teaser    {months, items}: три МО из TEASER_MO; items[k] — {id, short, region, fact,
              forecast, known, breaks} теми же значениями, что в demo/data/mo/*.json;
              months — 24 месяца панели и 12 месяцев прогноза подряд.
    breaks    {months, share, top, top_labels}: раздел «Изломы». share[t] — доля муниципалитетов
              (%, два знака), у которых по полному ряду, задним числом, найден излом в месяце
              months[t]: `results/cp_offline.csv` на протоколе `v{changepoints.yaml::protocol_version}`
              и штрафе `scripts/news_event_study.py::PENALTY` — той же офлайновой картине, что
              изломы стенда. top — месяцы массового согласия (столько наибольших долей, сколько
              событий в `changepoints.yaml::realtime.events`, по порядку месяцев, как `_cp_top`
              отчёта), top_labels — их доли готовыми строками, «85,8%».

Размер данных главной ограничен `MAX_LANDING_BYTES` — как и размер данных стенда, он проверяется
до записи.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from string import Template
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_matrix, load_panel  # noqa: E402
from src.external import load_aggregate  # noqa: E402
from src.results_guard import read_results, refused  # noqa: E402
from src.split import rolling_origin  # noqa: E402

# Порог размера стенда: суммарный размер demo/data/ не должен превышать это число байт.
MAX_DEMO_BYTES = 2_000_000

# Порог размера данных главной: JSON внутри index.html грузится вместе со страницей, а не
# по требованию, как данные стенда, поэтому лимит на порядок жёстче.
MAX_LANDING_BYTES = 40_000

# МО стенда по умолчанию — то же, чем отчёт иллюстрирует изломы одного ряда
# (report/report.qmd, ~строка 5008): готовая, проверенная на реальных данных иллюстрация.
DEFAULT_MO = "городской округ город Орёл"

# С какого месяца рисовать историю на графике федерального агрегата — как в
# фигуре отчёта (`src/charts.py::aggregate_check`, параметр по умолчанию
# `start="2022-01"`), чтобы год проверки (2025) не оказался прижат к правому
# краю: без обрезки график начинался бы в 2018 году.
AGGREGATE_CHART_START = "2022-01"

# Обложка и раздел «Почему это прогноз одного числа» рисуют ряды одной и той же выборки:
# сид фиксирован, чтобы страница не менялась от сборки к сборке. Размер выборки — число
# линий на графике; больше — и линии сливаются в пятно.
LANDING_SAMPLE_SEED = 20261007
LANDING_SAMPLE_SIZE = 30

# Ряды нормируются к среднему за первые месяцы панели: на обложке и в «Почему это прогноз
# одного числа» 100% — «расходы такие же, как в среднем за этот период».
STORY_BASE_MONTHS = 12

# Подпись третьего шага «Почему это прогноз одного числа»: доля значений панели (в процентах),
# которые лежат в полосе вокруг общего движения. Ширина полосы — процентиль отклонений этой доли,
# округлённый вверх до целого процента: полоса не уже, чем нужно, чтобы доля была честной.
STORY_SPREAD_SHARE = 90

# Муниципалитеты блока «Стенд»: названия — как в series_id панели, короткая подпись — для
# чипа. Нужен ряд с регионом (Орёл, Казань) и ряд-омоним с номером «#N» (Михайловский
# район): на нём видно, что в адрес стенда номер идёт закодированным. Ряд, которого нет
# в панели, роняет сборку (`build_teaser`).
TEASER_MO = [
    (DEFAULT_MO, "Орёл"),
    ("городской округ город Казань", "Казань"),
    ("Михайловский муниципальный район #2", "Михайловский р-н #2"),
]

# Быстрые кнопки под полем поиска на стенде: ряд (series_id панели) и короткая подпись кнопки.
# Набор подобран так, чтобы с первого нажатия были видны все случаи стенда: города с регионом
# и ОКТМО, ряд-омоним с номером «#N» (в адресе он идёт закодированным) и ряды без региона,
# у которых стенд объясняет, почему регион не приписан. Подписи общих с блоком «Стенд» главной
# рядов совпадают с TEASER_MO. Ряд, которого нет в панели, роняет сборку (`build_quick`).
STAND_QUICK_MO = [
    (DEFAULT_MO, "Орёл"),
    ("городской округ город Казань", "Казань"),
    ("городской округ город Новосибирск", "Новосибирск"),
    ("городской округ город Екатеринбург", "Екатеринбург"),
    ("Михайловский муниципальный район #2", "Михайловский р-н #2"),
    ("Ардатовский муниципальный район", "Ардатовский р-н"),
]

# Различитель рядов без региона в подсказках поиска: средние расходы за столько последних месяцев
# панели (год), тыс. руб. на человека в месяц. Входит в `index.json::series` пятым элементом.
MEAN_MONTHS = 12

# Наукаст на месячных данных — это горизонт 1 (README, отчёт, configs/horizons.yaml): внутри
# месяца данных нет, поэтому месяц t прогнозируется по данным до t−1. Год вперёд — горизонт 12,
# как в `compute_placeholders` (`h12_*`): подпись «Год» на переключателе и оговорка об одном фолде.
NOWCAST_HORIZON = 1
YEAR_HORIZON = 12

# Названия моделей — копия `_ru` отчёта (report/report.qmd, ~строка 1682): тот же
# читателю текст в обоих местах. Модель без записи здесь получает на странице
# свой сырой идентификатор — работает, но подписи не будет; такого сейчас не бывает.
MODEL_LABELS = {
    "naive_last": "наивная: оставить как в прошлом месяце",
    "prophet": "эталон конкурса (Prophet по умолчанию)",
    "two_stage": "двухэтапная: одно число и разнос долями",
    "global_gbm": "панельная модель, без внешних источников",
    "global_gbm_cat": "панельная + доли категорий",
    "global_gbm_factor": "панельная + общий фактор",
    "global_gbm_stack": "панельная + категории рядами",
    "global_gbm_stack_factor": "панельная + фактор + категории рядами",
    "chronos_ft": "Chronos-Bolt, дообученный",
}

# Кандидаты первого этапа двухэтапной модели (`src/external.py::CANDIDATES`) —
# копия `_fc_agg_ru` отчёта (report/report.qmd, ~строка 2081): та же модель,
# то же название на странице и в отчёте.
AGGREGATE_MODEL_LABELS = {
    "seasonal_median": "медианный по годам прирост того же месяца",
    "seasonal_naive": "прошлогодний прирост того же месяца",
    "ets": "экспоненциальное сглаживание с затухающим трендом",
    "sarima": "SARIMA",
    "random_walk": "последнее значение",
}

# ---------------------------------------------------------------------------
# Форматирование чисел — перенесено из report/report.qmd (строки 42-102, 212) с тем же
# поведением: неразрывный пробел в разрядах, запятая, минус «−», знак и NaN как там.
# Число страницы обязано быть той же строкой, что число отчёта, а не только тем же
# значением — отсюда копия функций, а не собственный форматтер.
# ---------------------------------------------------------------------------

_ORDINAL = {0: "первом", 1: "втором", 2: "третьем"}
_NUMBER_WORDS_M = ["ноль", "один", "два", "три", "четыре", "пять",
                    "шесть", "семь", "восемь", "девять", "десять"]
MONTH_OF = ["января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря"]
MONTH_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь",
             "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def rub(value: float) -> str:
    """Число с неразрывным пробелом в разряде тысяч, как в отчёте (`report.qmd::rub`)."""
    return f"{value:,.0f}".replace(",", " ")


def num(value: float, digits: int, sign: bool = False) -> str:
    """Число с `digits` знаками, минусом «−» и, при `sign`, «+» у положительных;
    NaN — прочерк (`report.qmd::num` — то же поведение, включая правило знака у нуля)."""
    if not np.isfinite(value):
        return "—"
    value = round(float(value), digits) + 0.0
    text = format(value, ("+" if sign and value > 0 else "") + f",.{digits}f")
    return text.replace(",", " ").replace(".", ",").replace("-", "−")


def on_folds(k: int, n: int) -> str:
    """«на 2 фолдах из 3»; «все» и «ни одного» — словами (`report.qmd::on_folds`)."""
    if k == 0:
        return "ни на одном фолде"
    if k == n:
        return "на всех фолдах"
    return f"на {k} {'фолде' if k == 1 else 'фолдах'} из {n}"


def plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова при числе: 1 ряд, 2 ряда, 5 рядов (`report.qmd::plural`)."""
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


def and_join(items) -> str:
    """«a», «a и b», «a, b и c» (`report.qmd::and_join`)."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " и " + items[-1]


def in_words_m(n: int) -> str:
    """Малое число мужского рода словом: «один фолд» (`report.qmd::in_words`, форма «им_м»)."""
    return _NUMBER_WORDS_M[n] if 0 <= n <= 10 else str(n)


def _date_words(value: dt.date) -> str:
    return f"{value.day} {MONTH_OF[value.month - 1]} {value.year}"


def file_size_label(path: Path) -> str:
    """Размер файла для подписи ссылки: «2,4 МБ» — мегабайты по 1 048 576 байт, как их показывает файловый
    менеджер, с неразрывным пробелом перед единицей; файл меньше 0,05 МБ (при округлении до десятых он
    стал бы «0,0 МБ») — в килобайтах. Читатель видит размер до того, как нажмёт: PDF и архив с прогнозом
    скачиваются целиком."""
    size = path.stat().st_size
    if size >= 0.05 * 1024 ** 2:
        return f"{num(size / 1024 ** 2, 1)}\u00a0МБ"
    return f"{num(size / 1024, 0)}\u00a0КБ"


def _ruble(value: float) -> int | None:
    """Рубль как в формате данных стенда — целым; пропуск (NaN, отказ модели) — null."""
    return None if not np.isfinite(value) else int(round(float(value)))


def _round2(value: float) -> float | None:
    """Проценты и млрд руб. данных стенда — с двумя знаками; пропуск — null."""
    return None if not np.isfinite(value) else round(float(value), 2)


def _rub_or_dash(value: float) -> str:
    """``rub(value)`` — то же готовое представление, что у чисел отчёта, а при пропуске
    (NaN) — «—», как ``num`` выше: рублёвые строки таблицы агрегата (``actual_rub``,
    ``forecast_rub``), чтобы JS показывал их как есть, не округляя ещё раз."""
    return rub(value) if np.isfinite(value) else "—"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_penalty() -> float:
    """Штраф офлайновой картины изломов: `scripts/news_event_study.py::PENALTY`.

    Модуль грузится по пути (`scripts/` — не пакет), как это делает сам отчёт
    (`report/report.qmd`, ~строка 589), а не обычным импортом.
    """
    spec = importlib.util.spec_from_file_location(
        "news_event_study", ROOT / "scripts" / "news_event_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return float(module.PENALTY)


# ---------------------------------------------------------------------------
# Пять чисел главной
# ---------------------------------------------------------------------------


def compute_placeholders(
    *, wide: pd.DataFrame, summary: pd.DataFrame, ok: pd.DataFrame,
    horizons_summary: pd.DataFrame, horizons_folds: pd.DataFrame,
    agg_check: pd.DataFrame, full_cfg: dict, forward_cfg: dict, today: dt.date,
) -> dict[str, str]:
    """Считает все подстановки `${имя}` шаблона `site/index.template.html`.

    Текст главной живёт в шаблоне, а каждое число в нём — одна из этих
    подстановок; вписанных руками чисел в шаблоне нет. Пять чисел — те же, что
    в резюме отчёта: главное число, наукаст, год вперёд, проверка агрегата, панель.

    - ``built`` — дата сборки словами, «29 сентября 2026».
    - ``default_mo`` — МО, на котором открывается страница прогноза по муниципалитету без выбора:
      пустой запрос в поле обложки ведёт именно на него (`DEFAULT_MO`).
    - ``report_pdf_size`` / ``slides_pdf_size`` / ``forecast_size`` — размеры PDF отчёта, PDF слайдов
      и архива `results/forecast_2025.csv.gz` в подписях карточек материалов, «2,4 МБ» (`file_size_label`).
    - ``horizon_main`` — горизонт основного протокола, мес. (`configs/full.yaml::split.horizon`).
    - ``forecast_year`` — год прогноза вперёд, `origin.year + 1` (`configs/forecast_forward.yaml::origin`).
    - ``prophet_mae`` / ``best_mae`` — MAE эталона (Prophet) и лучшей модели на этом горизонте, ₽.
    - ``best_gain`` — выигрыш лучшей модели к эталону, %, без знака (главное число;
      `_rs["gain"]` отчёта).
    - ``r2_prophet`` / ``r2_best`` — R² пул эталона и лучшей модели там же.
    - ``folds_caveat`` — оговорка о фолдах целиком: «(Но )?выигрыш держится на N
      фолдах из M[: на таком-то эталон точнее]».
    - ``h1_best`` / ``h1_prophet`` / ``h1_gain`` — то же для наукаста, горизонт 1;
      «—», если лучшая модель основного протокола не входит в `configs/horizons.yaml`.
    - ``h12_gain_naive`` / ``h12_gain_prophet`` — выигрыш лучшей модели года вперёд
      (горизонт 12, без оракула) к наивной и к эталону, %. Лучшая выбирается минимумом
      MAE среди моделей, где есть обе, поэтому оба выигрыша не отрицательны.
    - ``h12_model`` — эта модель: «двухэтапная» или «название» из `MODEL_LABELS`.
    - ``h12_folds_caveat`` — «, но это один фолд» (число фолдов словом), если у неё
      зачтено меньше трёх фолдов, иначе пусто: хвост фразы, как в `_rs["h12"]` отчёта.
    - ``agg_own_pct`` / ``agg_rules_pct`` — ошибка первого этапа двухэтапной модели
      и простых правил, % (диапазон по горизонтам проверки, схлопывается в одно
      число, если границы совпадают после округления).
    - ``agg_rules`` — названия простых правил из файла проверки, в кавычках и через «и»:
      `«как в декабре» и «как год назад»`.
    - ``agg_horizon_range`` — горизонты проверки, «1–12».
    - ``agg_origin_label`` — месяц и год origin словами, «декабря 2024».
    - ``n_series_rub`` / ``n_months`` — рядов и месяцев панели, из формы матрицы
      `build_matrix` (`configs/forecast_forward.yaml::data`).
    - ``panel_shape`` — «N рядов × M месяцев» целиком, слова — через `plural`.
    - ``panel_span`` — первый и последний месяц панели словами, «январь 2023 — декабрь 2024».
    - ``folds_short`` — та же оговорка о фолдах одним оборотом, «на 2 фолдах из 3»: для узких
      экранов, где подпись числа сокращена, и для строки с главным числом на обложке.
    - ``fold_months`` — длина проверочного окна одного фолда основного протокола, мес.
      (`index.json::folds`): словарь под карточками объясняет «фолд» этим числом.
    - ``sample_label`` — «30 случайных муниципалитетов»: выборка обложки с согласованными формами.
    - ``spread_share`` / ``spread_pct`` — не менее чем у `spread_share`% значений по всей панели
      отклонение от общего движения (медианы по рядам) не больше `spread_pct`%: подпись третьего
      шага «Почему это прогноз одного числа». Доля — константа `STORY_SPREAD_SHARE`, ширина полосы —
      её процентиль отклонений, округлённый вверх. Считается на всей матрице, а не на 30
      нарисованных рядах.
    - ``story_base`` — период, к среднему за который нормированы ряды обложки и первого раздела:
      «2023 год» (первые `STORY_BASE_MONTHS` месяцев панели — календарный год) или
      «первые 12 месяцев панели».
    - ``landing_json`` — данные главной (формат — докстринг модуля); добавляет `main` после
      сборки, сюда он не входит. Так же `main` добавляет ``cp_claim`` (HTML фразы об изломах,
      `changepoint_claim`), ``cp_method`` («PELT на темпах роста») и ``cp_penalty`` (штраф
      офлайновой картины): они считаются не здесь, а `build_breaks` из файлов стенда разладок.
    """
    top = summary["MAE"].idxmin()
    horizon_main = int(full_cfg["split"]["horizon"])
    origin = pd.Period(forward_cfg["origin"], "M")
    # Год прогноза — origin (декабрь) + 1: настройка forecast_forward.yaml всегда
    # прогнозирует от конца года на следующий целиком, поэтому «год вперёд» — один
    # календарный год, а не только это число фолдов.
    forecast_year = origin.year + 1

    placeholders: dict[str, str] = {
        "built": _date_words(today),
        "default_mo": DEFAULT_MO,
        "report_pdf_size": file_size_label(ROOT / "report" / "report.pdf"),
        "slides_pdf_size": file_size_label(ROOT / "report" / "slides.pdf"),
        "forecast_size": file_size_label(ROOT / "results" / "forecast_2025.csv.gz"),
        "horizon_main": str(horizon_main),
        "forecast_year": str(forecast_year),
        "prophet_mae": rub(summary.loc["prophet", "MAE"]),
        "best_mae": rub(summary.loc[top, "MAE"]),
        # Выигрыш без знака, как `_rs["gain"]` отчёта: на странице число стоит перед словом
        # «точнее», и минус изменения MAE с ним бы спорил. top — минимум MAE сводки, где есть
        # и эталон, так что выигрыш не бывает отрицательным.
        "best_gain": num(summary.loc[top, "к Prophet, %"], 1),
        "r2_prophet": num(summary.loc["prophet", "R² пул"], 3),
        "r2_best": num(summary.loc[top, "R² пул"], 3),
    }

    # Оговорка о фолдах: держится ли выигрыш из summary.csv и на каком фолде, если
    # нет, — эталон точнее (report.qmd::_rs["won_short"], логика _vs_lost/_won_mae).
    fold_mae = (ok[ok["model"].isin([top, "prophet"])]
                .groupby(["fold", "model"])["mae"].mean().unstack())
    r2_folds = list(fold_mae.index)
    won_mae = [f for f in r2_folds if fold_mae.loc[f, top] < fold_mae.loc[f, "prophet"]]
    vs_lost = [f for f in r2_folds if fold_mae.loc[f, top] > fold_mae.loc[f, "prophet"]]
    vs_first = len(vs_lost) == 1 and vs_lost[0] == min(r2_folds)
    lead = "Но выигрыш держится " if vs_lost else "Выигрыш держится "
    # Окно проверки фолда — те же границы, что в index.json::folds (`src.split.rolling_origin`).
    first_fold = rolling_origin(len(wide), horizon_main, int(full_cfg["split"]["n_folds"]))[0]
    placeholders["fold_months"] = str(first_fold.test_end - first_fold.test_start)
    placeholders["folds_short"] = on_folds(len(won_mae), len(r2_folds))
    placeholders["folds_caveat"] = (
        lead + on_folds(len(won_mae), len(r2_folds))
        + (f": на {and_join(_ORDINAL.get(f, str(f)) for f in vs_lost)}"
           + (", с самой короткой историей," if vs_first else "") + " эталон точнее"
           if vs_lost else "")
        + "."
    )

    # Наукаст — горизонт 1 по требованию организаторов (configs/horizons.yaml), не
    # результат прогона: как и в отчёте, число горизонта здесь не выведено из файла.
    hzi = horizons_summary.set_index(["horizon", "model"])
    if (1, top) in hzi.index and (1, "prophet") in hzi.index:
        placeholders["h1_best"] = rub(hzi.loc[(1, top), "MAE"])
        placeholders["h1_prophet"] = rub(hzi.loc[(1, "prophet"), "MAE"])
        placeholders["h1_gain"] = num(hzi.loc[(1, top), "к Prophet, %"], 1)
    else:
        placeholders["h1_best"] = placeholders["h1_prophet"] = placeholders["h1_gain"] = "—"

    # Год вперёд — горизонт 12, без оракула two_stage_known (report.qmd::_rs["h12"]). На странице
    # у числа одна подпись, и оговорка о фолдах — в ней же, поэтому сюда идут части фразы
    # отчёта, а не она целиком: модель и хвост «, но это один фолд».
    year = horizons_summary[
        (horizons_summary["horizon"] == 12) & horizons_summary["MAE"].notna()
        & (horizons_summary["model"] != "two_stage_known")
    ].set_index("model")
    if len(year) and {"naive_last", "prophet"} <= set(year.index):
        year_top = year["MAE"].idxmin()
        placeholders["h12_gain_naive"] = num(year.loc[year_top, "к наивной, %"], 0)
        placeholders["h12_gain_prophet"] = num(year.loc[year_top, "к Prophet, %"], 0)
        placeholders["h12_model"] = ("двухэтапная" if year_top == "two_stage"
                                     else f"«{MODEL_LABELS.get(year_top, year_top)}»")
        # Столько фолдов, сколько допускают 24 точки: без их числа процент выигрыша
        # читался бы как устойчивый (та же оговорка, что в резюме отчёта).
        n_folds_top = int(horizons_folds.loc[
            (horizons_folds["horizon"] == 12) & (horizons_folds["model"] == year_top), "MAE"
        ].notna().sum())
        placeholders["h12_folds_caveat"] = (
            f", но это {in_words_m(n_folds_top)} {plural(n_folds_top, 'фолд', 'фолда', 'фолдов')}"
            if 0 < n_folds_top < 3 else ""
        )
    else:
        placeholders["h12_gain_naive"] = placeholders["h12_gain_prophet"] = "—"
        placeholders["h12_model"] = "—"
        placeholders["h12_folds_caveat"] = ""

    # Проверка первого этапа по факту 2025 года (report.qmd::_rs["agg"]).
    agg_mape = agg_check.assign(e=agg_check["error_pct"].abs()).groupby(["method", "horizon"])["e"].mean()
    agg_own, agg_rules = agg_mape.loc["two_stage"], agg_mape.drop(index="two_stage")
    agg_names = (agg_check.loc[agg_check["method"] != "two_stage"]
                 .drop_duplicates("method")["aggregate_model"].tolist())
    agg_hs = sorted(int(h) for h in agg_check["horizon"].unique())

    def _range(lo: float, hi: float) -> str:
        return num(lo, 1) if num(lo, 1) == num(hi, 1) else f"{num(lo, 1)}–{num(hi, 1)}"

    placeholders["agg_own_pct"] = _range(agg_own.min(), agg_own.max())
    placeholders["agg_rules_pct"] = _range(agg_rules.min(), agg_rules.max())
    placeholders["agg_rules"] = and_join(f"«{name}»" for name in agg_names)
    placeholders["agg_horizon_range"] = f"{agg_hs[0]}–{agg_hs[-1]}"
    placeholders["agg_origin_label"] = f"{MONTH_OF[origin.month - 1]} {origin.year}"

    # Панель — форма матрицы, а не результат прогона.
    n_series, n_months = wide.shape[1], wide.shape[0]
    placeholders["n_series_rub"] = rub(n_series)
    placeholders["n_months"] = str(n_months)
    placeholders["panel_shape"] = (
        f"{rub(n_series)} {plural(n_series, 'ряд', 'ряда', 'рядов')} × "
        f"{n_months} {plural(n_months, 'месяц', 'месяца', 'месяцев')}"
    )
    start, end = wide.index[0], wide.index[-1]
    placeholders["panel_span"] = (
        f"{MONTH_NOM[start.month - 1]} {start.year} — {MONTH_NOM[end.month - 1]} {end.year}"
    )

    placeholders["sample_label"] = (
        f"{LANDING_SAMPLE_SIZE} "
        + plural(LANDING_SAMPLE_SIZE, "случайный муниципалитет", "случайных муниципалитета",
                 "случайных муниципалитетов")
    )
    norm, median = _story_matrix(wide)
    deviation = norm.div(median, axis=0).sub(1).abs().to_numpy()
    # Округление вверх: при процентиле 7,03% полоса ±7% вмещала бы чуть меньше заявленной доли.
    # До ceil — округление до шести знаков, чтобы шум float не добавил лишний процент к целому.
    spread = float(np.percentile(deviation, STORY_SPREAD_SHARE)) * 100
    placeholders["spread_share"] = str(STORY_SPREAD_SHARE)
    placeholders["spread_pct"] = str(math.ceil(round(spread, 6)))
    placeholders["story_base"] = story_base_label(wide)
    return placeholders


# ---------------------------------------------------------------------------
# Изломы — точки структурных изменений: фраза резюме отчёта и доли территорий с изломом
# ---------------------------------------------------------------------------

# Названия детекторов и режимов — как в отчёте (`report/report.qmd`: `_CP_DET`, `_CP_DET_GEN`, `_CP_ON`,
# ~строки 617-623): тот же текст обязан стоять и там, и здесь.
CP_DETECTOR = {"pelt": "PELT", "binseg": "BinSeg", "window": "Window", "bottomup": "BottomUp",
               "kernel_rbf": "ядровой (RBF)", "cusum": "CUSUM"}
CP_DETECTOR_GEN = {**CP_DETECTOR, "kernel_rbf": "ядрового (RBF)"}
CP_MODE_ON = {"ratio": "на темпах роста", "raw": "на сырых значениях", "deseason": "без профиля месяца"}

# Таблицы стенда разладок, которые читает раздел: сводка по детекторам, потоковая доля по месяцам, события
# потокового сигнала и изломы по полному ряду. Остальные файлы стенда нужны отчёту, а не странице.
CP_TABLES = ("cp_summary", "cp_realtime", "cp_realtime_events", "cp_offline")


def _cp_true(column: pd.Series) -> pd.Series:
    """Флаг из CSV: True/False читаются булевыми, но колонка с пропусками становится строковой
    (`report.qmd::_cp_true`)."""
    return column.astype(str).str.lower() == "true"


def _cp_pct(value: float) -> str:
    """Процент с одним знаком; «100%», а не «100,0%»: ноль после запятой лишний (`report.qmd::_cp_pct`)."""
    text = num(value, 1)
    return f"{text[:-2] if text.endswith(',0') else text}%"


def _cp_range(values) -> str:
    """Диапазон процентов «53,3–73,3%»; при совпавших краях — одно число (`report.qmd::_cp_range`)."""
    lo, hi = _cp_pct(min(values)), _cp_pct(max(values))
    return hi if lo == hi else f"{lo[:-1]}–{hi}"


def _cp_pen(value: float) -> str:
    """Штраф в тексте: «3», «1», «2,5» (`report.qmd::_cp_pen`)."""
    return num(value, 0 if float(value).is_integer() else 1)


def load_cp_tables(cp_cfg: dict, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """`CP_TABLES` из `results/` — только версии протокола `changepoints.yaml::protocol_version` и на сетке
    штрафов конфига (`report.qmd::_cp_table` и проверка сетки там же). Файл другой версии или другой сетки
    страница не принимает: числа прежнего прогона в раздел действующего молча не встанут."""
    protocol = f"v{cp_cfg['protocol_version']}"
    grid = sorted(float(p) for p in cp_cfg["bench"]["penalties"])
    tables = {}
    for name in CP_TABLES:
        kwargs = {"dtype": {"crossed_month": object}} if name == "cp_realtime_events" else {}
        frame = read_results(root / "results" / f"{name}.csv", **kwargs)
        if "protocol" not in frame or set(frame["protocol"]) != {protocol}:
            raise ValueError(
                f"results/{name}.csv — не версии протокола {protocol}: запустите scripts/changepoints.py")
        if sorted(set(frame["penalty"].dropna().astype(float))) != grid:
            raise ValueError(
                f"results/{name}.csv посчитан на другой сетке штрафов, чем configs/changepoints.yaml: "
                "перезапустите scripts/changepoints.py")
        tables[name] = frame
    return tables


def changepoint_claim(cps: pd.DataFrame, cpe: pd.DataFrame, cp_cfg: dict) -> tuple[str, str]:
    """Фраза о точках структурных изменений — пункт резюме отчёта (`report/report.qmd`, `_cp_summary`,
    ~строки 790-1060), условия которого пересчитаны здесь на тех же файлах, без ссылки «[об обнаружении]».
    Возвращает (жирное начало, остальное). Тезис «решает преобразование ряда, а не алгоритм» держится, если у
    каждого метода преобразование снижает ложные тревоги и поднимает J Юдена, а разброс J между методами
    со штрафом меньше выигрыша любого из них; иначе фраза называет только невыполненное условие.

    - `cps` — `cp_summary.csv`, `cpe` — `cp_realtime_events.csv` (оба версии протокола конфига);
    - число обнаружения не называется без оговорки: «скромный» — пока лучший J не дотягивает до пятой
      части шкалы; «раннего предупреждения не показано» — если при выбранном штрафе порог не перейдён
      или все переходы пришлись на декабрь позже месяца события."""
    realtime, bench = cp_cfg["realtime"], cp_cfg["bench"]
    detector, mode = realtime["detector"], realtime["mode"]
    events = sorted(str(e) for e in realtime["events"])
    p_sel = float(cpe.loc[_cp_true(cpe["selected"]), "penalty"].iloc[0])
    event_table = cpe.set_index(["penalty", "event"])
    crossed = {
        p: [e for e in events if isinstance(event_table.loc[(p, e), "crossed_month"], str)]
        for p in sorted(set(cpe["penalty"]))
    }

    # Стенд при выбранном штрафе; CUSUM — со своим порогом, штраф у него пуст.
    selected_rows = cps[(cps["penalty"] == p_sel) | cps["penalty"].isna()]
    by_method = selected_rows.set_index(["detector", "mode"])
    youden, false_alarm = by_method["J Юдена"], by_method["ложных на чистых, %"]
    own = set(cps.loc[cps["penalty"].isna(), "detector"])
    dets = [d for d in bench["detectors"] if (d, "raw") in by_method.index and (d, mode) in by_method.index]
    pen_dets = [d for d in dets if d not in own]
    gain = {d: youden[(d, mode)] - youden[(d, "raw")] for d in pen_dets}
    ratio_j = [youden[(d, mode)] for d in pen_dets]
    spread = max(ratio_j) - min(ratio_j) if ratio_j else np.nan

    worse = [d for d in dets
             if not (youden[(d, mode)] > youden[(d, "raw")] and false_alarm[(d, mode)] < false_alarm[(d, "raw")])]
    why_not = "; ".join(
        ([f"у {and_join(CP_DETECTOR_GEN[d] for d in worse)} {CP_MODE_ON[mode]} ложных тревог не меньше "
          f"или J не выше, чем {CP_MODE_ON['raw']}"] if worse else [])
        + ([f"разброс J между методами со штрафом {CP_MODE_ON[mode]}, {num(spread, 1)} пункта, "
            f"не меньше наименьшего выигрыша от смены преобразования, {num(min(gain.values()), 1)}"]
           if gain and not spread < min(gain.values()) else [])
        + (["методов со штрафом при этом штрафе в сводке нет"] if not gain else []))
    thesis = not why_not

    # Лучший по J — с ничьими: у PELT и BinSeg на одних рядах числа бывают одинаковы, и лучший по первой
    # строке файла был бы одним из двух наугад. Первым — детектор потокового сигнала.
    best_j = selected_rows["J Юдена"].round(9)
    order = {d: i for i, d in enumerate(bench["detectors"])}
    best_keys = sorted(
        selected_rows.loc[best_j == best_j.max(), ["detector", "mode"]].itertuples(index=False, name=None),
        key=lambda k: (k != (detector, mode), order.get(k[0], len(order)), k[1]))
    level = "скромный" if by_method.loc[best_keys[0]]["J Юдена"] < 20 else "заметный"

    def december_only(p: float) -> bool:
        """Все переходы порога при штрафе p пришлись на декабрь позже месяца события."""
        return bool(crossed[p]) and all(
            event_table.loc[(p, e), "crossed_month"].endswith("-12") and event_table.loc[(p, e), "delay"] > 0
            for e in crossed[p])

    no_early = not crossed[p_sel] or december_only(p_sel)
    raw_pen = [false_alarm[(d, "raw")] for d in pen_dets]
    ratio_pen = [false_alarm[(d, mode)] for d in pen_dets]
    head = ("Точки структурных изменений: "
            + ("решает преобразование ряда, а не алгоритм" if thesis else "преобразование ряда решает не всё") + ".")
    body = (
        (f"{CP_MODE_ON['raw'][:1].upper() + CP_MODE_ON['raw'][1:]} методы со штрафом тревожат на "
         f"{_cp_range(raw_pen)} нетронутых рядов, {CP_MODE_ON[mode]} — "
         + ("ни разу" if max(ratio_pen) == 0 else f"на {_cp_range(ratio_pen)}") + ". "
         if thesis else
         f"Тезис «решает преобразование, а не алгоритм» этим прогоном не подтверждается: {why_not}. ")
        + f"Уровень обнаружения {level}"
        + (", и раннего предупреждения на двух годах данных не показано" if no_early else
           ", а в реальном времени видны все события" if len(crossed[p_sel]) == len(events) else
           f", а в реальном времени видно {len(crossed[p_sel])} из {len(events)} событий")
        + "."
    )
    return head, body


class BreaksBuild(NamedTuple):
    """Раздел «Изломы»: фраза резюме (HTML и текст), подписи графика и его данные."""

    claim_html: str
    claim_text: str
    method: str
    penalty: str
    data: dict


def build_breaks(tables: dict[str, pd.DataFrame], cp_cfg: dict, penalty: float) -> BreaksBuild:
    """Раздел «Изломы» главной: фраза резюме отчёта и доли территорий с изломом по месяцам
    (`landing-data.breaks`, формат — докстринг модуля).

    Доли — офлайновая картина отчёта: изломы по полному ряду, найденные задним числом, на протоколе конфига
    и штрафе `penalty` (`scripts/news_event_study.py::PENALTY`) — те же, что изломы на стенде прогноза
    по муниципалитету. Это не сигнал в реальном времени: так сказано и в подписи графика."""
    head, body = changepoint_claim(tables["cp_summary"], tables["cp_realtime_events"], cp_cfg)
    realtime = cp_cfg["realtime"]
    cpe, cpr, cpo = tables["cp_realtime_events"], tables["cp_realtime"], tables["cp_offline"]
    p_sel = float(cpe.loc[_cp_true(cpe["selected"]), "penalty"].iloc[0])
    months = list(cpr.loc[cpr["penalty"] == p_sel, "month"])
    offline = cpo.pivot(index="month", columns="penalty", values="share").reindex(months)
    if float(penalty) not in offline.columns or offline[float(penalty)].isna().any():
        raise ValueError(
            f"в results/cp_offline.csv нет долей изломов при штрафе {_cp_pen(penalty)} на все месяцы панели")
    share = offline[float(penalty)]
    # Месяцы массового согласия — столько наибольших долей, сколько событий в конфиге стенда, по порядку месяцев.
    top = sorted(share.nlargest(len(realtime["events"])).index, key=months.index)
    return BreaksBuild(
        claim_html=f"<strong>{html.escape(head, quote=False)}</strong> {html.escape(body, quote=False)}",
        claim_text=f"{head} {body}",
        method=f"{CP_DETECTOR[realtime['detector']]} {CP_MODE_ON[realtime['mode']]}",
        penalty=_cp_pen(penalty),
        data={
            "months": months,
            "share": [round(float(v), 2) for v in share],
            "top": top,
            "top_labels": [_cp_pct(share[m]) for m in top],
        },
    )


# ---------------------------------------------------------------------------
# Данные демонстрационного стенда
# ---------------------------------------------------------------------------


def _check_series_match(forecast: pd.DataFrame, wide: pd.DataFrame) -> None:
    """Ряды `forecast_2025.csv` обязаны совпасть со столбцами матрицы панели — иначе
    протокол или конфиг разошлись между прогонами, и дальнейшая сборка была бы на вранье."""
    forecast_series = set(forecast["series_id"].unique())
    matrix_series = set(wide.columns)
    if forecast_series != matrix_series:
        only_forecast = sorted(forecast_series - matrix_series)
        only_matrix = sorted(matrix_series - forecast_series)
        example = ", ".join((only_forecast + only_matrix)[:5])
        raise ValueError(
            "ряды results/forecast_2025.csv не совпадают со столбцами матрицы панели: "
            f"только в прогнозе {len(only_forecast)}, только в матрице {len(only_matrix)} "
            f"(пример: {example}) — прогон forecast_forward.py и текущая панель не совпадают"
        )


def _step_horizon_map(horizons: list[int]) -> dict[int, int]:
    """Шаг s (1..max) → наименьший горизонт, который его покрывает
    (`report/report.qmd::_fc_steps`, ~строка 2102)."""
    hs = sorted(horizons)
    return {s: min(h for h in hs if h >= s) for s in range(1, hs[-1] + 1)}


def _steps_frame(subset: pd.DataFrame, step_horizon: dict[int, int], origin: pd.Period,
                  columns: pd.Index) -> pd.DataFrame:
    """Матрица (ряд × шаг 1..max) значений `forecast` из `subset` (уже отфильтрованных
    строк — `recommended` или `uses_published_aggregate`): на каждый шаг — значение
    с месяца `origin + шаг` на покрывающем его горизонте. Ряд без строки на этот
    шаг — NaN (в JSON станет null)."""
    columns_by_step = {}
    for step, horizon in step_horizon.items():
        month = str(origin + step)
        part = subset.loc[(subset["horizon"] == horizon) & (subset["month"] == month)]
        columns_by_step[step] = part.set_index("series_id")["forecast"]
    frame = pd.DataFrame(columns_by_step).reindex(columns)
    return frame[sorted(frame.columns)]


def _build_forecast_rule(recommended: pd.DataFrame, step_horizon: dict[int, int],
                          origin: pd.Period) -> list[dict]:
    """`forecast_rule`: отрезки подряд идущих месяцев с одной моделью и горизонтом.

    `label` — то же название, что в `models[].label`, а не только сырой `model`:
    стенду не нужен свой словарь названий моделей рядом с генератором (одна модель
    без записи в `MODEL_LABELS` — сырой идентификатор, как и у `models[].label`)."""
    by_horizon = recommended.drop_duplicates(["horizon", "model"]).groupby("horizon")["model"].agg(list)
    for horizon, models in by_horizon.items():
        if len(models) != 1:
            raise ValueError(f"на горизонте {horizon} среди recommended-строк больше одной модели: {models}")
    model_of = by_horizon.map(lambda models: models[0])

    runs: list[dict] = []
    for step in sorted(step_horizon):
        horizon = step_horizon[step]
        model = model_of.loc[horizon]
        month = str(origin + step)
        if runs and runs[-1]["model"] == model and runs[-1]["horizon"] == horizon:
            runs[-1]["to"] = month
        else:
            runs.append({
                "from": month, "to": month, "model": model, "horizon": int(horizon),
                "label": MODEL_LABELS.get(model, model),
            })
    return runs


def _build_model_roles(summary: pd.DataFrame, forward_cfg: dict) -> list[dict]:
    """Пять ролей таблицы ошибок стенда, в порядке докстринга модуля. `role` — строка;
    модель в нескольких ролях сразу входит одной строкой, роли соединены «+»."""
    recommended = forward_cfg["recommended"]
    shortest, longest = min(recommended), max(recommended)
    roles_in_order = [
        ("prophet", "reference"),
        ("naive_last", "naive"),
        (recommended[shortest], "recommended"),
        (summary["MAE"].idxmin(), "best_mean"),
        (recommended[longest], "two_stage"),
    ]
    by_model: dict[str, list[str]] = {}
    for model_id, role in roles_in_order:
        by_model.setdefault(model_id, []).append(role)
    return [
        {"id": model_id, "role": "+".join(roles), "label": MODEL_LABELS.get(model_id, model_id)}
        for model_id, roles in by_model.items()
    ]


def _load_breaks(cp_cfg: dict, penalty: float) -> dict[str, list[str]]:
    """series_id → отсортированные месяцы изломов офлайновой картины: `cp_offline_series.csv`
    на протоколе `v{protocol_version}` и штрафе `scripts/news_event_study.py::PENALTY`
    (так отчёт выбирает офлайновую картину, `report/report.qmd` ~строки 587-600, 5005-5012)."""
    protocol = f"v{cp_cfg['protocol_version']}"
    offline = read_results(ROOT / "results" / "cp_offline_series.csv")
    subset = offline.loc[(offline["protocol"] == protocol) & np.isclose(offline["penalty"], penalty)]
    return {series: sorted(months) for series, months in subset.groupby("series_id")["month"]}


def build_quick(series_ids) -> list[dict[str, str]]:
    """Поле `quick` файла `demo/data/index.json`: быстрые кнопки стенда из `STAND_QUICK_MO`.

    Ряд, которого нет среди `series_ids`, роняет сборку: кнопка на несуществующий ряд
    открыла бы стенд с МО по умолчанию и сообщением «в данных стенда нет»."""
    available = set(series_ids)
    missing = [series_id for series_id, _ in STAND_QUICK_MO if series_id not in available]
    if missing:
        raise ValueError(f"МО быстрых кнопок стенда нет среди рядов панели: {', '.join(missing)}")
    return [{"id": series_id, "short": short} for series_id, short in STAND_QUICK_MO]


class DemoBuild(NamedTuple):
    """Данные стенда, собранные в памяти: что писать, сколько это весит и то, что нужно главной."""

    files: dict[Path, str]  # путь → текст; на диск ещё ничего не записано
    total_bytes: int
    entries: dict[str, dict]  # series_id → запись demo/data/mo/<номер>.json
    regions: dict[str, str | None]  # series_id → регион (None — не определён)


def build_demo_data(
    data_dir: Path, *, wide: pd.DataFrame, forecast: pd.DataFrame, ok: pd.DataFrame,
    summary: pd.DataFrame, full_cfg: dict, forward_cfg: dict, cp_cfg: dict,
    penalty: float, aggregate: dict, built: str,
) -> DemoBuild:
    """Собирает `demo/data/index.json`, `demo/data/mo/<номер>.json` и `demo/data/aggregate.json`
    (формат — докстринг модуля) и проверяет размер; на диск не пишет — это `write_demo_data`.

    `aggregate` приходит готовым (`build_aggregate_json`): ту же структуру главная кладёт
    в свои данные, и считать её дважды значило бы рисковать расхождением."""
    # Горизонты — по ключам recommended, как _fc_steps отчёта (report/report.qmd), а не
    # forward_cfg["horizons"]: тот список и recommended сейчас совпадают, но recommended —
    # источник истины (это она отвечает и за прогноз, и за пунктир), а horizons здесь был
    # бы совпадением, которое зеркалить в точности надёжнее, чем полагаться на то, что оно
    # не разойдётся.
    horizons = sorted(forward_cfg["recommended"])
    origin = pd.Period(forward_cfg["origin"], "M")
    step_horizon = _step_horizon_map(horizons)

    recommended_rows = forecast.loc[forecast["recommended"]]
    known_rows = forecast.loc[forecast["uses_published_aggregate"]]
    forecast_by_series = _steps_frame(recommended_rows, step_horizon, origin, wide.columns)
    known_by_series = _steps_frame(known_rows, step_horizon, origin, wide.columns)
    forecast_rule = _build_forecast_rule(recommended_rows, step_horizon, origin)

    regions = (forecast.drop_duplicates("series_id").set_index("series_id")[["region", "oktmo"]]
               .reindex(wide.columns))
    unique_regions = sorted(r for r in regions["region"].dropna().unique())
    region_number = {region: i for i, region in enumerate(unique_regions)}
    no_region_number = len(unique_regions)

    model_roles = _build_model_roles(summary, forward_cfg)
    model_ids = [role["id"] for role in model_roles]
    fold_mae = ok[ok["model"].isin(model_ids)].pivot_table(index=["mo", "model"], columns="fold", values="mae")
    n_folds = int(full_cfg["split"]["n_folds"])

    panel_mae = {}
    for model_id in model_ids:
        by_fold = ok.loc[ok["model"] == model_id].groupby("fold")["mae"].mean().reindex(range(n_folds))
        values = [_ruble(v) for v in by_fold.to_numpy()]
        panel_mae[model_id] = values + [_ruble(float(by_fold.mean()))]

    breaks_by_series = _load_breaks(cp_cfg, penalty)

    fold_specs = rolling_origin(len(wide), full_cfg["split"]["horizon"], n_folds)
    folds = [
        {
            "fold": f.index, "train_months": f.train_end,
            "test_from": wide.index[f.test_start].strftime("%Y-%m"),
            "test_to": wide.index[f.test_end - 1].strftime("%Y-%m"),
        }
        for f in fold_specs
    ]

    # Гомонимы — по совпадению базового имени (без " #N") у нескольких столбцов матрицы,
    # не по региону-омониму (src.regions.attach_regions — это n_no_region, другая причина).
    base_names = pd.Index(wide.columns).map(lambda s: s.split(" #")[0])
    homonym_counts = pd.Series(base_names).value_counts()
    homonym_counts = homonym_counts[homonym_counts > 1]

    # То же, но строго по факту суффикса " #N" в series_id: подсказка поиска говорит
    # «их различает номер после «#»», и это утверждение верно именно для рядов с таким
    # суффиксом — не для всех рядов с неуникальным базовым именем. У пяти названий
    # вторая копия выпала из панели по пропускам: оставшийся ряд с суффиксом "#N" без
    # пары, homonym_counts выше такой ряд омонимом уже не считает (группа из одного),
    # а суффикс на его series_id всё ещё есть.
    hash_series = pd.Index(wide.columns)[pd.Index(wide.columns).str.contains(r" #\d+$", regex=True)]
    hash_names = hash_series.map(lambda s: re.sub(r" #\d+$", "", s))

    if DEFAULT_MO not in wide.columns:
        raise ValueError(f"МО стенда по умолчанию {DEFAULT_MO!r} не найдено среди рядов панели")

    series_entries = []
    region_of: dict[str, str | None] = {}
    mo_payload: dict[int, dict] = defaultdict(dict)
    for series_id in wide.columns:
        region = regions.at[series_id, "region"]
        oktmo = regions.at[series_id, "oktmo"]
        region = None if pd.isna(region) else region
        oktmo = None if pd.isna(oktmo) else oktmo
        number = region_number.get(region, no_region_number)
        entry = [series_id, region, oktmo, number]
        if region is None:
            mean = float(wide[series_id].iloc[-MEAN_MONTHS:].mean())
            if np.isfinite(mean):
                entry.append(num(mean / 1000, 1))
        series_entries.append(entry)
        region_of[series_id] = region

        mae = {}
        for model_id in model_ids:
            try:
                row = fold_mae.loc[(series_id, model_id)]
            except KeyError:
                mae[model_id] = [None] * n_folds
                continue
            mae[model_id] = [_ruble(row.get(f, np.nan)) for f in range(n_folds)]

        mo_payload[number][series_id] = {
            "fact": [_ruble(v) for v in wide[series_id].to_numpy()],
            "forecast": [_ruble(v) for v in forecast_by_series.loc[series_id].to_numpy()],
            "known": [_ruble(v) for v in known_by_series.loc[series_id].to_numpy()],
            "breaks": breaks_by_series.get(series_id, []),
            "mae": mae,
        }

    index_payload = {
        "built": built,
        "unit": "руб. на человека в месяц",
        "origin": forward_cfg["origin"],
        "panel_months": [pd.Timestamp(m).strftime("%Y-%m") for m in wide.index],
        "forecast_months": [str(origin + step) for step in range(1, horizons[-1] + 1)],
        "n_series": int(wide.shape[1]),
        "n_no_region": int(regions["region"].isna().sum()),
        "n_homonym_names": int(len(homonym_counts)),
        "n_homonym_series": int(homonym_counts.sum()) if len(homonym_counts) else 0,
        "n_hash_names": int(hash_names.nunique()),
        "n_hash_series": int(len(hash_series)),
        "default_mo": DEFAULT_MO,
        "quick": build_quick(wide.columns),
        "forecast_rule": forecast_rule,
        "known_model": forward_cfg["known_aggregate"],
        "breaks": {
            "protocol": f"v{cp_cfg['protocol_version']}", "penalty": penalty,
            "detector": cp_cfg["realtime"]["detector"], "mode": cp_cfg["realtime"]["mode"],
        },
        "folds": folds,
        "models": model_roles,
        "panel_mae": panel_mae,
        "series": series_entries,
    }

    # Всё сериализуется в память и меряется ДО записи на диск: раньше порог проверялся
    # уже после того, как main() записал все файлы, и сборка, упавшая по размеру, всё
    # равно оставляла их на диске. Теперь либо пишется всё, либо ничего.
    files: dict[Path, str] = {data_dir / "index.json": _serialize_json(index_payload)}
    for number, payload in mo_payload.items():
        files[data_dir / "mo" / f"{number}.json"] = _serialize_json(payload)
    files[data_dir / "aggregate.json"] = _serialize_json(aggregate)

    total = sum(len(text.encode("utf-8")) for text in files.values())
    if total > MAX_DEMO_BYTES:
        raise ValueError(
            f"данные стенда {total} байт больше порога {MAX_DEMO_BYTES} — "
            "сократите состав или точность полей demo/data/, не поднимайте порог не глядя"
        )

    entries = {series_id: entry for payload in mo_payload.values() for series_id, entry in payload.items()}
    return DemoBuild(files=files, total_bytes=total, entries=entries, regions=region_of)


def write_demo_data(demo: DemoBuild, data_dir: Path) -> None:
    """Пишет собранные данные стенда в `data_dir`."""
    # Старые demo/data/mo/*.json чистятся перед записью новых: без этого при уменьшении
    # числа регионов лишние файлы прежних прогонов оставались бы в рабочем каталоге
    # (а при коммите — и в репозитории) неограниченно долго.
    mo_dir = data_dir / "mo"
    if mo_dir.exists():
        for old in mo_dir.glob("*.json"):
            old.unlink()
    for path, text in demo.files.items():
        _write_text(path, text)


def build_aggregate_json(agg_check: pd.DataFrame, forward_cfg: dict, root: Path = ROOT) -> dict:
    """`demo/data/aggregate.json` — федеральный агрегат: история и проверка первого
    этапа двухэтапной модели по факту 2025 года (формат — докстринг модуля)."""
    origin_str = forward_cfg["origin"]
    origin = pd.Period(origin_str, "M")
    # Тот же источник истины, что и в build_demo_data (см. её комментарий): ключи
    # recommended, а не forward_cfg["horizons"] отдельно.
    horizon = max(forward_cfg["recommended"])
    methods = ["two_stage", "naive", "seasonal_naive"]

    history = load_aggregate(root / "data" / "reference" / "sberindex").loc[AGGREGATE_CHART_START:origin]

    year = agg_check.loc[agg_check["horizon"] == horizon]
    months = sorted(year["month"].unique())
    actual_by_month = year.groupby("month")["actual"].first().reindex(months)
    forecast_by_method = year.pivot(index="month", columns="method", values="forecast").reindex(months)
    error_by_method = year.pivot(index="month", columns="method", values="error_pct").reindex(months)

    abs_error = agg_check.assign(e=agg_check["error_pct"].abs())
    mape = abs_error.loc[abs_error["horizon"] == horizon].groupby("method")["e"].mean()

    model_id = year.loc[year["method"] == "two_stage", "aggregate_model"].iloc[0]
    return {
        "unit": "млрд руб. в месяц",
        "origin": origin_str,
        "horizon": int(horizon),
        "model": model_id,
        "model_label": AGGREGATE_MODEL_LABELS.get(model_id, model_id),
        "history": {
            "months": [str(m) for m in history.index],
            "values": [_round2(v) for v in history.to_numpy()],
        },
        "check": {
            "months": months,
            "actual": [_round2(v) for v in actual_by_month],
            "forecast": {m: [_round2(v) for v in forecast_by_method[m]] for m in methods},
            # Строки, отформатированные rub()/num(v, 1, sign=True) — тем же, что таблица
            # отчёта (report/report.qmd), а не числа: JS показывает их как есть, без
            # своего округления (на стенде иначе было «−3,9%» на том месяце, где
            # в отчёте «−4,0»: JS округлял уже округлённые питоном сотые ещё раз).
            # actual/forecast выше остаются числами для графика (buildAggregateChartSpec);
            # рублёвые колонки таблицы стенда читают готовые строки ниже.
            "actual_rub": [_rub_or_dash(v) for v in actual_by_month],
            "forecast_rub": {"two_stage": [_rub_or_dash(v) for v in forecast_by_method["two_stage"]]},
            "error_pct": {m: [num(v, 1, sign=True) for v in error_by_method[m]] for m in methods},
        },
        "rule_names": {
            "naive": agg_check.loc[agg_check["method"] == "naive", "aggregate_model"].iloc[0],
            "seasonal_naive":
                agg_check.loc[agg_check["method"] == "seasonal_naive", "aggregate_model"].iloc[0],
        },
        # Тоже готовая строка num(v, 1), не число — см. actual_rub/forecast_rub/error_pct
        # выше. mape_by_horizon (та же ошибка на каждом горизонте файла, не только
        # check.horizon) убран: стенд его не показывает.
        "mape": {m: num(mape.get(m, np.nan), 1) for m in methods},
    }


# ---------------------------------------------------------------------------
# Данные главной (формат — докстринг модуля, раздел «Данные главной»)
# ---------------------------------------------------------------------------


def _story_matrix(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Ряды матрицы, делённые на среднее своих первых `STORY_BASE_MONTHS` месяцев, и их
    медиана по рядам в каждом месяце — «общее движение»."""
    norm = wide / wide.iloc[:STORY_BASE_MONTHS].mean()
    return norm, norm.median(axis=1)


def story_base_label(wide: pd.DataFrame) -> str:
    """Период нормировки словами, после предлога «за»: «2023 год», если первые
    `STORY_BASE_MONTHS` месяцев панели — целые календарные годы, иначе «первые 12 месяцев
    панели». Год на странице берётся из данных, а не пишется в шаблоне: панель может начаться
    не с января, и тогда «год» был бы неправдой."""
    start = wide.index[0]
    if start.month == 1 and STORY_BASE_MONTHS % 12 == 0:
        first, last = start.year, start.year + STORY_BASE_MONTHS // 12 - 1
        return f"{first} год" if first == last else f"{first}–{last} годы"
    return f"первые {STORY_BASE_MONTHS} {plural(STORY_BASE_MONTHS, 'месяц', 'месяца', 'месяцев')} панели"


def build_story(wide: pd.DataFrame) -> dict:
    """`story` данных главной: 30 рядов выборки и медиана по ВСЕМ рядам матрицы."""
    if wide.shape[1] < LANDING_SAMPLE_SIZE:
        raise ValueError(
            f"в матрице {wide.shape[1]} рядов — меньше выборки обложки ({LANDING_SAMPLE_SIZE})"
        )
    norm, median = _story_matrix(wide)
    picked = np.random.default_rng(LANDING_SAMPLE_SEED).choice(
        wide.shape[1], size=LANDING_SAMPLE_SIZE, replace=False
    )
    ids = [str(wide.columns[i]) for i in picked]
    return {
        "months": [pd.Timestamp(m).strftime("%Y-%m") for m in wide.index],
        "base_months": STORY_BASE_MONTHS,
        "ids": ids,
        "series": [[round(float(v), 3) for v in norm[series_id].to_numpy()] for series_id in ids],
        "median": [round(float(v), 3) for v in median.to_numpy()],
        "n_total": int(wide.shape[1]),
        "n_sample": LANDING_SAMPLE_SIZE,
    }


def horizon_label(horizon: int) -> str:
    """Подпись горизонта на переключателе главной: «Наукаст», «3 мес.», «Год»."""
    if horizon == NOWCAST_HORIZON:
        return "Наукаст"
    if horizon == YEAR_HORIZON:
        return "Год"
    return f"{horizon}\u00a0мес."


def _label_parts(model_id: str) -> tuple[str, str]:
    """Название и пояснение модели из `MODEL_LABELS`: «наивная: оставить как в прошлом
    месяце» → («Наивная», «оставить как в прошлом месяце»); «эталон конкурса (Prophet по
    умолчанию)» → («Эталон конкурса», «Prophet по умолчанию»). Без «:» и скобок — всё название,
    пояснение пусто."""
    text = MODEL_LABELS.get(model_id, model_id)
    if ": " in text:
        head, tail = text.split(": ", 1)
    elif text.endswith(")") and " (" in text:
        head, tail = text[:-1].split(" (", 1)
    else:
        head, tail = text, ""
    return head[:1].upper() + head[1:], tail


def horizon_folds(horizons_cfg: dict, horizons_summary: pd.DataFrame) -> dict[int, int]:
    """Число фолдов протокола на каждом горизонте: колонка «фолдов» `horizons_summary.csv`,
    сверенная с `n_folds` из `configs/horizons.yaml`.

    Число стоит в пояснении каждого горизонта: средняя по двум фолдам и по девяти — разные
    по весу утверждения. Файл результатов и конфиг, разошедшиеся в этом числе, — признак
    прогона не с тем конфигом, и сборка падает, а не выбирает одно из двух."""
    counts = horizons_summary.groupby("horizon")["фолдов"].agg(["min", "max"])
    mixed = [int(h) for h, row in counts.iterrows() if row["min"] != row["max"]]
    if mixed:
        raise ValueError(f"в results/horizons_summary.csv у горизонтов {mixed} разное число фолдов у моделей")
    wanted = {int(item["horizon"]): int(item["n_folds"]) for item in horizons_cfg["horizons"]}
    found = {h: int(counts.loc[h, "max"]) for h in wanted if h in counts.index}
    if found != wanted:
        raise ValueError(
            f"число фолдов по горизонтам в results/horizons_summary.csv ({found}) не совпадает "
            f"с configs/horizons.yaml ({wanted}) — результаты прогнаны с другим конфигом"
        )
    return wanted


def build_horizons(
    *, horizons_cfg: dict, horizons_summary: pd.DataFrame, summary: pd.DataFrame,
    full_cfg: dict, placeholders: dict[str, str],
) -> dict:
    """`horizons` данных главной: MAE четырёх моделей на горизонтах `configs/horizons.yaml`.

    Порядок — как в докстринге модуля; модель в двух ролях сразу входит одной строкой
    (роли через «+»), как в `_build_model_roles`. Название и пояснение берутся из
    `MODEL_LABELS`; у лучшей модели основного протокола название — «Лучшая панельная»:
    на годовом горизонте её MAE нет, и слово «панельная» объясняет почему — панельным
    моделям на год вперёд не хватает истории."""
    horizons = sorted({int(item["horizon"]) for item in horizons_cfg["horizons"]})
    best = summary["MAE"].idxmin()
    roles_by_model: dict[str, list[str]] = {}
    for model_id, role in [("prophet", "reference"), ("naive_last", "naive"),
                           (best, "best_mean"), ("two_stage", "two_stage")]:
        roles_by_model.setdefault(model_id, []).append(role)

    mae_table = horizons_summary.set_index(["horizon", "model"])["MAE"]

    def mae_at(horizon: int, model_id: str) -> int | None:
        try:
            return _ruble(float(mae_table.loc[(horizon, model_id)]))
        except KeyError:
            return None

    models = []
    for model_id, roles in roles_by_model.items():
        if roles == ["best_mean"]:
            label = "Лучшая панельная" if model_id.startswith("global_") else "Лучшая по MAE"
            note = MODEL_LABELS.get(model_id, model_id)
        else:
            label, note = _label_parts(model_id)
        models.append({
            "id": model_id, "role": "+".join(roles), "label": label, "note": note,
            "mae": [mae_at(h, model_id) for h in horizons],
        })

    best_label = next(m["label"] for m in models if "best_mean" in m["role"].split("+"))
    horizon_main = int(full_cfg["split"]["horizon"])
    folds = horizon_folds(horizons_cfg, horizons_summary)
    notes = []
    for i, horizon in enumerate(horizons):
        sentences = []
        if horizon == NOWCAST_HORIZON:
            sentences.append("Наукаст — прогноз текущего месяца, пока его данных ещё нет.")
        elif any(m["mae"][i] is None for m in models):
            sentences.append("Где полосы нет, модель не удалось обучить: на этом горизонте истории не хватает.")
        elif horizon == horizon_main:
            sentences.append("Средняя абсолютная ошибка по фолдам, ₽ на человека в месяц.")
        else:
            sentences.append("Средняя абсолютная ошибка по фолдам скользящего origin, ₽ на человека в месяц.")
        # Основной горизонт открыт по умолчанию: его подпись читатель видит первой, и в ней — слова, которых
        # дальше не объясняют: фолд (проверочное окно) и скользящий origin.
        if horizon == horizon_main and horizon != NOWCAST_HORIZON:
            sentences.append(
                f"Фолд — проверочное окно в {horizon} мес.: модель учится на месяцах до него и прогнозирует его. "
                "Origin — месяц, от которого строится прогноз; от фолда к фолду он сдвигается вперёд "
                "(скользящий origin)."
            )
        # Число фолдов — сразу за описанием горизонта: полоса «лучшая» по среднему, а на горизонтах
        # с двумя фолдами и одним порядок моделей по фолдам может быть другим.
        sentences.append(f"Фолдов: {folds[horizon]}.")
        if horizon == NOWCAST_HORIZON and placeholders["h1_gain"] != "—":
            sentences.append(f"{best_label} точнее эталона на {placeholders['h1_gain']}%.")
        if horizon == horizon_main and horizon != NOWCAST_HORIZON:
            sentences.append(f"{best_label} точнее эталона на {placeholders['best_gain']}%.")
            sentences.append(placeholders["folds_caveat"])
        if horizon == YEAR_HORIZON and placeholders["h12_model"] != "—":
            sentences.append(f"Лучшая — {placeholders['h12_model']}{placeholders['h12_folds_caveat']}.")
        notes.append(" ".join(sentences))

    return {
        "list": horizons,
        "main": horizon_main,
        "labels": [horizon_label(h) for h in horizons],
        "unit": "руб. на человека в месяц",
        "models": models,
        "notes": notes,
    }


def build_teaser(demo: DemoBuild, wide: pd.DataFrame, forward_cfg: dict) -> dict:
    """`teaser` данных главной: три МО из `TEASER_MO` теми же значениями, что в
    `demo/data/mo/*.json` (берутся из собранных данных стенда, а не считаются заново)."""
    missing = [series_id for series_id, _ in TEASER_MO if series_id not in demo.entries]
    if missing:
        raise ValueError(f"МО блока «Стенд» нет среди рядов панели: {', '.join(missing)}")
    # Мини-график блока не умеет рисовать дыры: линия факта, прогноза и пунктира целиком.
    gaps = [series_id for series_id, _ in TEASER_MO
            if any(value is None for key in ("fact", "forecast", "known") for value in demo.entries[series_id][key])]
    if gaps:
        raise ValueError(f"в данных МО блока «Стенд» есть пропуски: {', '.join(gaps)}")
    origin = pd.Period(forward_cfg["origin"], "M")
    forecast_months = [str(origin + step) for step in range(1, max(forward_cfg["recommended"]) + 1)]
    return {
        "months": [pd.Timestamp(m).strftime("%Y-%m") for m in wide.index] + forecast_months,
        "items": [
            {
                "id": series_id, "short": short, "region": demo.regions[series_id],
                **{key: demo.entries[series_id][key] for key in ("fact", "forecast", "known", "breaks")},
            }
            for series_id, short in TEASER_MO
        ],
    }


def build_landing_data(
    *, wide: pd.DataFrame, summary: pd.DataFrame, full_cfg: dict, forward_cfg: dict,
    horizons_cfg: dict, horizons_summary: pd.DataFrame, aggregate: dict, demo: DemoBuild,
    placeholders: dict[str, str], breaks: BreaksBuild,
) -> dict:
    """Все данные главной одним словарём: `story`, `horizons`, `fact`, `teaser`, `breaks`."""
    return {
        "story": build_story(wide),
        "horizons": build_horizons(
            horizons_cfg=horizons_cfg, horizons_summary=horizons_summary, summary=summary,
            full_cfg=full_cfg, placeholders=placeholders,
        ),
        "fact": aggregate,
        "teaser": build_teaser(demo, wide, forward_cfg),
        "breaks": breaks.data,
    }


def landing_json(payload: dict) -> str:
    """Данные главной текстом для `<script type="application/json">`.

    `<` пишется как `\\u003c`: иначе «</script» в любой строке данных закрыл бы тег и сломал
    страницу. JSON.parse в браузере разбирает такую запись в обычный «<». Размер
    проверяется здесь же — до записи страницы."""
    text = _serialize_json(payload).replace("<", "\\u003c")
    size = len(text.encode("utf-8"))
    if size > MAX_LANDING_BYTES:
        raise ValueError(
            f"данные главной {size} байт больше порога {MAX_LANDING_BYTES} — "
            "сократите состав или точность полей, не поднимайте порог не глядя"
        )
    return text


def _serialize_json(payload) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сборка главной страницы (index.html) и данных демонстрационного стенда"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT,
        help="куда писать index.html и demo/data/ (по умолчанию — корень репозитория)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out
    today = dt.date.today()

    full_cfg = _load_yaml(ROOT / "configs" / "full.yaml")
    forward_cfg = _load_yaml(ROOT / "configs" / "forecast_forward.yaml")
    horizons_cfg = _load_yaml(ROOT / "configs" / "horizons.yaml")
    cp_cfg = _load_yaml(ROOT / "configs" / "changepoints.yaml")
    penalty = _load_penalty()

    panel = load_panel(ROOT / forward_cfg["data"]["path"])
    wide, _ = build_matrix(panel, forward_cfg["data"]["category"], max_gap=forward_cfg["data"]["max_gap"])

    per_series = read_results(ROOT / "results" / "per_series.csv", dtype={"error": object})
    ok = per_series[~refused(per_series)]
    summary = read_results(ROOT / "results" / "summary.csv", index_col=0)
    horizons_summary = read_results(ROOT / "results" / "horizons_summary.csv")
    horizons_folds = read_results(ROOT / "results" / "horizons_folds.csv")
    agg_check = read_results(ROOT / "results" / "forecast_2025_aggregate_check.csv")
    forecast = read_results(ROOT / "results" / "forecast_2025.csv")

    _check_series_match(forecast, wide)

    placeholders = compute_placeholders(
        wide=wide, summary=summary, ok=ok, horizons_summary=horizons_summary,
        horizons_folds=horizons_folds, agg_check=agg_check,
        full_cfg=full_cfg, forward_cfg=forward_cfg, today=today,
    )

    # Структура агрегата одна на стенд и на главную: стенд пишет её в aggregate.json,
    # главная кладёт в свои данные как есть.
    aggregate = build_aggregate_json(agg_check, forward_cfg)

    data_dir = out / "demo" / "data"
    demo = build_demo_data(
        data_dir, wide=wide, forecast=forecast, ok=ok, summary=summary,
        full_cfg=full_cfg, forward_cfg=forward_cfg, cp_cfg=cp_cfg, penalty=penalty,
        aggregate=aggregate, built=today.isoformat(),
    )

    # Раздел «Изломы»: фраза резюме отчёта и доли территорий с изломом — из файлов стенда разладок.
    breaks = build_breaks(load_cp_tables(cp_cfg), cp_cfg, penalty)
    placeholders.update({
        "cp_claim": breaks.claim_html, "cp_method": breaks.method, "cp_penalty": breaks.penalty,
    })

    landing = build_landing_data(
        wide=wide, summary=summary, full_cfg=full_cfg, forward_cfg=forward_cfg,
        horizons_cfg=horizons_cfg, horizons_summary=horizons_summary, aggregate=aggregate,
        demo=demo, placeholders=placeholders, breaks=breaks,
    )
    placeholders["landing_json"] = landing_json(landing)

    template_path = ROOT / "site" / "index.template.html"
    page_html = Template(template_path.read_text(encoding="utf-8")).substitute(placeholders)

    # Пороги и подстановки проверены выше, в памяти: на диск идёт только собранное целиком.
    write_demo_data(demo, data_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.html"
    index_path.write_text(page_html, encoding="utf-8")

    print(f"написано: {index_path}")
    print(f"данные стенда: {data_dir} — {demo.total_bytes} байт (порог {MAX_DEMO_BYTES})")
    print(f"данные главной: {len(placeholders['landing_json'].encode('utf-8'))} байт (порог {MAX_LANDING_BYTES})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
