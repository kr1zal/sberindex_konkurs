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
детектор, кричащий на каждом шаге, получил бы стопроцентное обнаружение. Половина
стенда — нетронутые ряды, на которых любое срабатывание есть ложная тревога.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.changepoints import DETECTORS

MIN_HISTORY = 8      # раньше восьми точек ни один метод не имеет шансов
RECENT_WINDOW = 3    # разладка засчитывается, если найдена вблизи текущего конца


@dataclass
class Injection:
    kind: str
    position: int
    magnitude: float


def inject(y: np.ndarray, kind: str, position: int, magnitude: float) -> np.ndarray:
    """Вносит возмущение известного типа и величины. Величина — в долях σ ряда."""
    out = np.array(y, dtype=float)
    sigma = float(np.std(np.diff(out))) or 1.0
    shift = magnitude * sigma

    if kind == "level":
        out[position:] += shift
    elif kind == "trend":
        steps = np.arange(len(out) - position, dtype=float)
        out[position:] += shift * steps / max(1, len(steps) - 1) * 3.0
    elif kind == "variance":
        rng = np.random.default_rng(position)
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


def streaming_signal(y: np.ndarray, detector: str, mode: str = "raw") -> int | None:
    """Момент первого сигнала при последовательном просмотре ряда.

    Возвращает индекс t, на котором детектор впервые сообщил о разладке вблизи
    конца доступной истории, либо None, если не сообщил ни разу.
    """
    fn = DETECTORS[detector]
    offset = 1 if mode == "ratio" else 0   # темп роста короче исходного ряда на единицу
    for t in range(MIN_HISTORY, len(y)):
        window = preprocess(y[: t + 1], mode)
        try:
            found = fn(window).breakpoints
        except Exception:
            continue
        if any(b + offset >= t - RECENT_WINDOW for b in found):
            return t
    return None


def run_bench(
    wide: pd.DataFrame,
    n_series: int = 60,
    position: int = 14,
    magnitudes: tuple[float, ...] = (1.0, 2.0, 4.0),
    kinds: tuple[str, ...] = ("level", "trend", "variance"),
    modes: tuple[str, ...] = ("raw", "ratio", "deseason"),
    seed: int = 20260920,
) -> pd.DataFrame:
    """Полный прогон: обнаружение, запаздывание и ложные тревоги по каждому методу."""
    rng = np.random.default_rng(seed)
    columns = rng.choice(wide.columns, size=min(n_series, wide.shape[1]), replace=False)
    rows = []

    for col in columns:
        base = wide[col].to_numpy(dtype=float)

        for detector in DETECTORS:
            for mode in modes:
                # нетронутый ряд: любой сигнал здесь — ложная тревога
                signal = streaming_signal(base, detector, mode)
                rows.append(
                    {"detector": detector, "mode": mode, "kind": "none", "magnitude": 0.0,
                     "detected": signal is not None, "delay": np.nan,
                     "false_alarm": signal is not None}
                )

                for kind in kinds:
                    for magnitude in magnitudes:
                        spoiled = inject(base, kind, position, magnitude)
                        signal = streaming_signal(spoiled, detector, mode)
                        early = signal is not None and signal < position
                        hit = signal is not None and signal >= position
                        rows.append(
                            {"detector": detector, "mode": mode, "kind": kind,
                             "magnitude": magnitude, "detected": hit,
                             "delay": (signal - position) if hit else np.nan,
                             "false_alarm": early}
                        )
    return pd.DataFrame(rows)


def summarise(bench: pd.DataFrame) -> pd.DataFrame:
    """Сводка: полнота, запаздывание и доля ложных тревог на чистых рядах."""
    spoiled = bench[bench.kind != "none"]
    clean = bench[bench.kind == "none"]

    key = ["detector", "mode"]
    summary = pd.DataFrame(
        {
            "обнаружено": spoiled.groupby(key)["detected"].mean() * 100,
            "запаздывание": spoiled.groupby(key)["delay"].mean(),
            "ложных на чистых": clean.groupby(key)["false_alarm"].mean() * 100,
        }
    )
    # качество = обнаружение минус ложные тревоги: детектор, кричащий всегда,
    # получает стопроцентное обнаружение и должен быть за это наказан
    summary["баланс"] = summary["обнаружено"] - summary["ложных на чистых"]
    return summary.sort_values("баланс", ascending=False)
