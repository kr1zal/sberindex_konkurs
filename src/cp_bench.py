"""Стенд сравнения детекторов точек структурных изменений.

Главная трудность — не выбрать алгоритм, а понять, чем его мерить: размеченных
разладок в муниципальных расходах не существует. Стенд решает это так.

**Потоковый протокол.** Задача конкурса требует выявлять шоки «как можно раньше».
«Раньше» — свойство онлайнового режима: офлайновый метод, видящий ряд целиком,
на вопрос о своевременности не отвечает вовсе. Поэтому все методы, включая офлайновые,
гоняются в расширяющемся окне: в момент t детектор получает только y[:t+1] и должен
решить, произошла ли разладка вблизи текущего конца. Тогда запаздывание становится
сопоставимым числом для всех, а не привилегией онлайновых методов.

**Разметка из синтетики поверх настоящего шума.** Сдвиг известной величины вносится
в реальный ряд из панели. Разметка тогда верна по построению, а форма шума, сезонность
и масштаб — настоящие, а не выдуманные генератором.

**Ряды без структурных изменений.** Без них таблица показывает только полноту и молчит о точности:
детектор, кричащий на каждом шаге, получил бы стопроцентное обнаружение. Каждый ряд
выборки проходит стенд и нетронутым, где любое срабатывание есть ложная тревога:
на каждую настройку детектора и режим подготовки приходится одна чистая строка
и по одной испорченной на каждое сочетание возмущения, величины и позиции врезки.
Обнаружение и ложные тревоги считаются раздельно, по своим строкам, поэтому
это соотношение на сводку не влияет.

**Неизвестное число изломов.** Все методы, кроме CUSUM, работают в штрафном режиме
и сами решают, есть ли излом; штраф пробрасывается из стенда. У CUSUM свой порог.

Протокол — `configs/changepoints.yaml`, прогон — `scripts/changepoints.py`.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from functools import partial

import numpy as np
import pandas as pd

from src.changepoints import DETECTORS, Detection

MIN_HISTORY = 8      # раньше восьми точек ни один метод не имеет шансов
RECENT_WINDOW = 3    # разладка засчитывается, если найдена вблизи текущего конца

# n_failed — шаги, на которых детектор упал: протокол считает их шагами без флага,
# и без счётчика такой детектор выглядел бы молчаливым, а не сломанным.
BENCH_COLUMNS = [
    "detector", "mode", "penalty", "penalty_effective", "kind", "magnitude", "position",
    "series_id", "detected", "delay", "false_alarm", "signal", "n_failed",
]
SUMMARY_KEY = ["detector", "mode", "penalty"]
SUMMARY_COLUMNS = [
    "обнаружено, %", "задержка, медиана", "задержка, среднее", "доля с задержкой 0, %",
    "ложных на чистых, %", "J Юдена", "n испорченных", "n чистых", "n_failed",
]
CALIBRATION_COLUMNS = ["penalty", "kernel_pen", "fa_pelt", "fa_kernel", "n_failed_pelt", "n_failed_kernel"]


def inject(
    y: np.ndarray, kind: str, position: int, magnitude: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Вносит возмущение известного типа и величины. Величина — в долях σ ряда.

    Дисперсионному возмущению нужен генератор: стенд сеет его своим сидом, номером
    ряда в выборке и позицией врезки. Прежде генератор сеялся одной позицией,
    и на всех рядах стенда оказывался один и тот же шум.
    """
    out = np.array(y, dtype=float)
    sigma = float(np.std(np.diff(out))) or 1.0
    shift = magnitude * sigma

    if kind == "level":
        out[position:] += shift
    elif kind == "trend":
        steps = np.arange(len(out) - position, dtype=float)
        out[position:] += shift * steps / max(1, len(steps) - 1) * 3.0
    elif kind == "variance":
        if rng is None:
            raise ValueError("дисперсионному возмущению нужен генератор случайных чисел (rng)")
        out[position:] += rng.normal(0.0, abs(shift), size=len(out) - position)
    else:
        raise ValueError(f"неизвестный тип возмущения: {kind}")
    return out


def preprocess(y: np.ndarray, mode: str) -> np.ndarray:
    """Подготовка ряда перед детекцией.

    На сырых рядах все детекторы бесполезны: декабрьский скачок расходов на 17-22%
    выглядит как сдвиг уровня, и на нетронутых рядах срабатывание происходит в 77-100%
    случаев. То есть находится сезонность, а не структурные изменения.

    Все преобразования causal - используют только прошлое, иначе потоковый протокол
    потерял бы смысл.
    """
    y = np.asarray(y, dtype=float)
    if mode == "raw":
        return y
    if mode == "ratio":
        # темп роста месяц к месяцу: убирает мультипликативный тренд и разницу масштабов
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.diff(y) / y[:-1]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "deseason":
        # вычитаем средний профиль месяца, посчитанный по уже виденной истории
        out = np.empty(len(y))
        for t in range(len(y)):
            month = t % 12
            past = y[[i for i in range(t) if i % 12 == month]]
            out[t] = y[t] - (past.mean() if len(past) else y[: t + 1].mean())
        return out
    raise ValueError(f"неизвестный режим подготовки: {mode}")


def detector_with_penalty(detector: str, penalty: float | None) -> Callable[[np.ndarray], Detection]:
    """Детектор со штрафом стенда; None — штраф детектора по умолчанию.

    CUSUM штраф игнорирует: у него свой порог, и стенд его не трогает.
    """
    fn = DETECTORS[detector]
    if penalty is None or detector == "cusum":
        return fn
    return partial(fn, penalty=penalty)


def month_offset(mode: str) -> int:
    """Сдвиг индекса излома к месяцу ряда: темп роста короче исходного ряда на единицу.

    Излом b в ряду темпов роста — месяц b + 1, с которого уровень ряда стал другим.
    """
    return 1 if mode == "ratio" else 0


def _flag_months(
    y: np.ndarray, detector: str, mode: str, penalty: float | None, failed: list[int] | None = None,
) -> Iterator[int]:
    """Месяцы флага при последовательном просмотре ряда — единственная копия правила протокола.

    Флаг в месяц t: детектор на y[:t+1] после `preprocess` нашёл излом в последних
    RECENT_WINDOW точках. Шаг, на котором детектор упал, — шаг без флага, но его месяц
    дописывается в `failed`: сбой не должен выглядеть молчанием. Генератор ленивый:
    `streaming_signal` берёт первый флаг, и дальше детектор не зовётся.
    """
    fn = detector_with_penalty(detector, penalty)
    offset = month_offset(mode)
    for t in range(MIN_HISTORY, len(y)):
        window = preprocess(y[: t + 1], mode)
        try:
            found = fn(window).breakpoints
        except Exception:
            if failed is not None:
                failed.append(t)
            continue
        if any(b + offset >= t - RECENT_WINDOW for b in found):
            yield t


def streaming_signal(
    y: np.ndarray, detector: str, mode: str = "raw", penalty: float | None = None,
    failed: list[int] | None = None,
) -> int | None:
    """Момент первого сигнала при последовательном просмотре ряда.

    Возвращает индекс t, на котором детектор впервые сообщил о разладке вблизи
    конца доступной истории, либо None, если не сообщил ни разу.

    `penalty` — штраф стенда (`detector_with_penalty`). Без него перебор штрафа
    до потокового прогона не доходил: детектор звался со штрафом по умолчанию
    при любом значении сетки. `failed` — куда дописать месяцы сбоев детектора.
    """
    return next(_flag_months(y, detector, mode, penalty, failed), None)


def bench_sample(wide: pd.DataFrame, n_series: int, seed: int) -> pd.DataFrame:
    """Случайная выборка рядов стенда. Порядковый номер ряда в ней сеет его возмущение."""
    if n_series < 1:
        raise ValueError(f"в выборке стенда должен быть хотя бы один ряд, n_series = {n_series}")
    rng = np.random.default_rng(seed)
    columns = rng.choice(wide.columns, size=min(n_series, wide.shape[1]), replace=False)
    return wide.loc[:, list(columns)]


def calibrate_kernel_penalty(
    sample: pd.DataFrame, penalties: Sequence[float], mode: str, grid: Sequence[float],
) -> pd.DataFrame:
    """Штраф ядра, при котором доля ложных тревог как у PELT при номинальном штрафе.

    Номинальный штраф ядра несопоставим с методами l2 (`detect_kernel`): при одном
    числе сравнивалась бы строгость, а не метод. На нетронутых рядах `sample` в режиме
    `mode` по потоковому протоколу для каждого номинального штрафа берётся значение
    из `grid`, при котором доля ложных тревог ядра ближе всего к доле PELT; из точек,
    одинаково близких к ней, — самый чувствительный штраф, то есть меньший. Близость
    меряется числом рядов с тревогой, а не процентом: у процентов равные расстояния вверх
    и вниз расходятся в последнем разряде, и ничью решал бы шум округления. Доли в таблице —
    в процентах, как «ложных на чистых, %» в сводке. `n_failed_pelt`, `n_failed_kernel` —
    шаги, на которых упал PELT при номинальном штрафе и ядро при выбранном: протокол
    считает их шагами без тревоги, и доля с ними занижена.
    """
    if sample.shape[1] == 0:
        raise ValueError("калибровать штраф ядра не на чем: в выборке стенда нет рядов")
    grid = np.asarray(grid, dtype=float)
    if grid.size == 0:
        raise ValueError("сетка калибровки штрафа ядра пуста")
    series = [sample[col].to_numpy(dtype=float) for col in sample.columns]

    def alarms(detector: str, penalty: float) -> tuple[int, int]:
        failed: list[int] = []
        count = sum(streaming_signal(y, detector, mode, penalty=penalty, failed=failed) is not None
                    for y in series)
        return count, len(failed)

    scanned = [alarms("kernel_rbf", pen) for pen in grid]
    kernel = np.array([count for count, _ in scanned])
    rows = []
    for penalty in penalties:
        pelt, pelt_failed = alarms("pelt", penalty)
        distance = np.abs(kernel - pelt)
        tied = np.flatnonzero(distance == distance.min())
        best = tied[np.argmin(grid[tied])]
        rows.append({
            "penalty": penalty, "kernel_pen": float(grid[best]),
            "fa_pelt": pelt / len(series) * 100, "fa_kernel": kernel[best] / len(series) * 100,
            "n_failed_pelt": pelt_failed, "n_failed_kernel": scanned[best][1],
        })
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def effective_penalty(
    detector: str, penalty: float, kernel_penalties: dict[float, float] | None,
) -> float | None:
    """Штраф вызова детектора при номинальном штрафе: у ядра — калиброванный, у CUSUM
    штрафа нет — свой порог. Одно место и для стенда, и для реальных данных."""
    if detector == "cusum":
        return None
    if detector != "kernel_rbf":
        return penalty
    if kernel_penalties is None:
        raise ValueError(
            "штраф ядра без калибровки несопоставим с методами l2: "
            "передайте kernel_penalties (calibrate_kernel_penalty)"
        )
    return kernel_penalties[penalty]


def _settings(
    detectors: Sequence[str], penalties: Sequence[float], kernel_penalties: dict[float, float] | None,
) -> list[tuple[str, float, float | None]]:
    """Настройки стенда: (детектор, номинальный штраф, штраф вызова)."""
    settings = []
    for detector in detectors:
        if detector == "cusum":
            settings.append((detector, np.nan, None))  # свой порог: один прогон, штраф пуст
            continue
        for penalty in penalties:
            settings.append((detector, penalty, effective_penalty(detector, penalty, kernel_penalties)))
    return settings


def run_bench(
    sample: pd.DataFrame,
    *,
    positions: Sequence[int],
    magnitudes: Sequence[float],
    kinds: Sequence[str],
    modes: Sequence[str],
    detectors: Sequence[str],
    penalties: Sequence[float],
    seed: int,
    kernel_penalties: dict[float, float] | None = None,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Полный прогон: обнаружение, запаздывание и ложные тревоги по каждому методу.

    `sample` — ряды стенда (`bench_sample`). Протокол приходит из конфига, умолчаний
    у него нет намеренно: второй экземпляр протокола в коде разошёлся бы с конфигом
    молча. Методы со штрафом гоняются при каждом штрафе из `penalties`, CUSUM — один
    раз на режим, и штраф в его строках пуст. Ядро вызывается с калиброванным штрафом
    `kernel_penalties[номинальный]`: в строке номинальный — в `penalty`, сопоставимый
    с остальными методами, а вызванный — в `penalty_effective`.

    Сигнал до врезки на испорченном ряде — ложная тревога, не раньше неё — обнаружение
    с задержкой `signal − position`. Испорченные ряды общие для всех детекторов:
    методы сравниваются на одних и тех же случайных числах.
    """
    settings = _settings(detectors, penalties, kernel_penalties)
    rows = []
    started = time.perf_counter()
    for i, col in enumerate(sample.columns):
        base = sample[col].to_numpy(dtype=float)
        spoiled = [
            (kind, magnitude, position,
             inject(base, kind, position, magnitude, rng=np.random.default_rng([seed, i, position])))
            for position in positions for kind in kinds for magnitude in magnitudes
        ]
        for detector, penalty, effective in settings:
            for mode in modes:
                common = {
                    "detector": detector, "mode": mode, "penalty": penalty,
                    "penalty_effective": np.nan if effective is None else effective, "series_id": col,
                }
                # нетронутый ряд: обнаруживать нечего, любой сигнал здесь — ложная тревога
                failed: list[int] = []
                signal = streaming_signal(base, detector, mode, penalty=effective, failed=failed)
                rows.append({
                    **common, "kind": "none", "magnitude": 0.0, "position": np.nan,
                    "detected": False, "delay": np.nan, "false_alarm": signal is not None,
                    "signal": np.nan if signal is None else signal, "n_failed": len(failed),
                })
                for kind, magnitude, position, y in spoiled:
                    failed = []
                    signal = streaming_signal(y, detector, mode, penalty=effective, failed=failed)
                    hit = signal is not None and signal >= position
                    rows.append({
                        **common, "kind": kind, "magnitude": magnitude, "position": position,
                        "detected": hit, "delay": (signal - position) if hit else np.nan,
                        "false_alarm": signal is not None and signal < position,
                        "signal": np.nan if signal is None else signal, "n_failed": len(failed),
                    })
        if log is not None:
            log(f"  ряд {i + 1}/{sample.shape[1]} ({col}): {time.perf_counter() - started:.0f} с")
    return pd.DataFrame(rows, columns=BENCH_COLUMNS)


def summarise(bench: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Сводка детектор × режим × штраф: полнота, задержка и ложные тревоги на чистых рядах.

    Задержка (медиана, среднее) и доля нулевой задержки — только среди обнаруженных:
    у необнаруженного задержки нет. `by` — дополнительный разрез испорченных строк,
    например по величине возмущения; ложные тревоги от него не зависят и берутся
    по чистым рядам той же тройки детектор × режим × штраф.

    J Юдена — обнаружено минус ложных, то есть TPR − FPR: детектор, кричащий всегда,
    получает стопроцентное обнаружение и должен быть за это наказан. Штраф CUSUM пуст,
    и группировка не должна его терять (`dropna=False`). `n_failed` — шаги, на которых
    детектор упал, по испорченным строкам группы и чистым строкам её тройки.
    """
    key = SUMMARY_KEY + list(by)
    spoiled = bench[bench["kind"] != "none"]
    clean = bench[bench["kind"] == "none"]

    by_spoiled = spoiled.groupby(key, dropna=False)
    by_hit = spoiled[spoiled["detected"]].groupby(key, dropna=False)["delay"]
    by_clean = clean.groupby(SUMMARY_KEY, dropna=False)
    detection = pd.DataFrame({
        "обнаружено, %": by_spoiled["detected"].mean() * 100,
        "n испорченных": by_spoiled.size(),
        "failed_spoiled": by_spoiled["n_failed"].sum(),
    })
    delay = pd.DataFrame({
        "задержка, медиана": by_hit.median(),
        "задержка, среднее": by_hit.mean(),
        "доля с задержкой 0, %": by_hit.agg(lambda d: (d == 0).mean() * 100),
    })
    alarms = pd.DataFrame({
        "ложных на чистых, %": by_clean["false_alarm"].mean() * 100,
        "n чистых": by_clean.size(),
        "failed_clean": by_clean["n_failed"].sum(),
    })
    # Слияние по колонкам, а не по индексу: pandas сопоставляет пустой штраф CUSUM
    # с пустым только при слиянии колонок.
    out = (
        detection.reset_index()
        .merge(delay.reset_index(), on=key, how="left")
        .merge(alarms.reset_index(), on=SUMMARY_KEY, how="left")
    )
    out["J Юдена"] = out["обнаружено, %"] - out["ложных на чистых, %"]
    out["n чистых"] = out["n чистых"].fillna(0).astype(int)
    out["n_failed"] = (out["failed_spoiled"] + out["failed_clean"].fillna(0)).astype(int)
    order = [*by, "J Юдена"]
    ascending = [True] * len(by) + [False]
    return out[key + SUMMARY_COLUMNS].sort_values(order, ascending=ascending, kind="stable").reset_index(drop=True)


def realtime_signals(
    y: np.ndarray, detector: str, mode: str, penalty: float | None, failed: list[int] | None = None,
) -> list[int]:
    """Начала эпизодов тревоги ряда при последовательном просмотре — для панели на реальных данных.

    Флаги — те же, что у `streaming_signal` (`_flag_months`). Сигнал — начало эпизода:
    флаг в t и ни одного флага в t − 1, …, t − RECENT_WINDOW; месяцы до MIN_HISTORY
    считаются месяцами без флага, так что в первый проверяемый месяц сигнал возможен —
    как на стенде. Первый сигнал совпадает со `streaming_signal`.

    Эпизод кончается только после RECENT_WINDOW месяцев подряд без флага. Излом считается
    недавним RECENT_WINDOW месяцев, и тревога живёт столько же после последнего флага —
    новой константы нет. Разовый сдвиг уровня в темпах роста — одиночный выброс: PELT
    сначала ставит излом у конца окна, при строгом штрафе на шаг-два теряет его и затем
    отсекает выброс вторым изломом. С одним лишь переходом «нет флага → флаг» такой
    перерыв давал второй эпизод через несколько месяцев после сдвига; с гистерезисом
    второй остаётся, только если флагов не было RECENT_WINDOW месяцев подряд.

    Ограничение: события, чьи серии флагов у ряда подходят друг к другу ближе чем на
    RECENT_WINDOW месяцев без флага, сливаются в один эпизод — второе у ряда не видно.
    Сливается и с шумом: ложный флаг в последние RECENT_WINDOW месяцев перед событием
    поглощает начало его эпизода, и при мягком штрафе, когда ложных флагов много, так
    теряется заметная часть начал. Перерыв ровно в RECENT_WINDOW месяцев и больше — уже
    два эпизода.
    """
    signals: list[int] = []
    last_flag: int | None = None
    for t in _flag_months(y, detector, mode, penalty, failed):
        if last_flag is None or t - last_flag > RECENT_WINDOW:
            signals.append(t)
        last_flag = t
    return signals


def panel_realtime(
    wide: pd.DataFrame, detector: str, mode: str, penalty: float | None,
    threshold_share: float, events: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Потоковый сигнал на панели: начала эпизодов тревоги по месяцам и по событиям.

    Каждый ряд просматривается в расширяющемся окне (`realtime_signals`, там же правило
    эпизода и его ограничение). По месяцам: `month` (ГГГГ-ММ), `n_signals` — рядов,
    у которых эпизод начался в этом месяце, `share` — их процент от рядов панели,
    `n_failed` — рядов, у которых в этом месяце упал детектор: такой ряд остаётся
    в знаменателе доли без шанса попасть в числитель. Первые MIN_HISTORY месяцев
    детектор не запускается, и доля там нулевая по построению.

    События — месяцы в терминах панели: месяц, с которого уровень ряда стал другим
    (излом b в ряду темпов роста — месяц b + 1). Окно события — от его месяца до месяца
    следующего события, не включая его, у последнего — до конца панели: каждый месяц
    принадлежит одному событию, и переход порога следующего события не приписывается
    предыдущему. В окне: `crossed_month` — первый месяц с долей не ниже `threshold_share`,
    иначе пусто; `delay` — месяцев от события до него; `max_share` — наибольшая месячная
    доля; `share_in_window` — процент МО, у которых эпизод начался где-либо в окне:
    потоковый аналог доли изломов по полному ряду, накопленный за окно.
    """
    months = pd.DatetimeIndex(wide.index).strftime("%Y-%m")
    failed: list[int] = []
    per_series = [
        realtime_signals(wide[col].to_numpy(dtype=float), detector, mode, penalty, failed)
        for col in wide.columns
    ]
    counts = np.zeros(len(months), dtype=int)
    for signals in per_series:
        for t in signals:
            counts[t] += 1
    share = counts / wide.shape[1] * 100
    n_failed = np.bincount(np.asarray(failed, dtype=int), minlength=len(months))
    monthly = pd.DataFrame({"month": months, "share": share, "n_signals": counts, "n_failed": n_failed})

    index = {month: i for i, month in enumerate(months)}
    unknown = [event for event in events if event not in index]
    if unknown:
        raise ValueError(f"месяцы событий вне панели: {unknown}")
    event_months = sorted(index[event] for event in events)
    rows = []
    for event in events:
        start = index[event]
        end = next((m for m in event_months if m > start), len(months))
        crossed = np.flatnonzero(share[start:end] >= threshold_share)
        at = start + int(crossed[0]) if crossed.size else None
        inside = sum(any(start <= t < end for t in signals) for signals in per_series)
        rows.append({
            "event": event,
            "crossed_month": None if at is None else months[at],
            "delay": np.nan if at is None else at - start,
            "max_share": float(share[start:end].max()),
            "share_in_window": inside / wide.shape[1] * 100,
        })
    columns = ["event", "crossed_month", "delay", "max_share", "share_in_window"]
    return monthly, pd.DataFrame(rows, columns=columns)
