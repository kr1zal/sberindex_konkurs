"""Обнаружение точек структурных изменений и честное сравнение методов.

Главная методологическая трудность здесь — не выбор алгоритма, а то, ЧЕМ его мерить.
Размеченных разладок в муниципальных расходах не существует, «правильного ответа»
взять неоткуда. Поэтому сравнение строится на двух независимых опорах:

1. Синтетический стенд. В реальный ряд вносится сдвиг известной величины в известный
   момент, и метод оценивается по тому, нашёл ли он его, где именно и с каким
   запаздыванием. Разметка тут по построению верна, а форма шума — настоящая,
   взятая из самих данных.

2. Панельное согласие. Разладка, найденная одновременно во множестве муниципалитетов,
   скорее отражает общероссийский шок, чем случайность в отдельном ряду. На 24 точках
   одиночный ряд почти не несёт сигнала, а панель из двух тысяч — несёт.

Задача конкурса требует выявлять шоки «как можно раньше и точнее», поэтому
запаздывание обнаружения считается наравне с точностью, а не после неё.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import ruptures as rpt


@dataclass
class Detection:
    method: str
    breakpoints: list[int] = field(default_factory=list)


def _run_ruptures(y: np.ndarray, algo, penalty: float) -> list[int]:
    """Разладки при штрафе за каждую: число изломов метод выбирает сам.

    Режим с заданным заранее числом изломов здесь не поддерживается намеренно:
    при одном заданном изломе он находится всегда, даже на чистом ряде, — это
    сегментация при известном числе изломов, а не обнаружение. Стенд в таком режиме
    мерил постановку, а не метод: «ложные тревоги» на чистых рядах были её артефактом.

    ruptures возвращает последним индексом длину ряда — это не разладка, отрезаем.
    """
    found = algo.fit(y.reshape(-1, 1)).predict(pen=penalty)
    return [int(b) for b in found if b < len(y)]


# У методов со стоимостью l2 выигрыш от излома измеряется в квадратах единиц ряда,
# поэтому штраф умножается на дисперсию ряда: тогда одно число штрафа значит одно
# и то же у всех четырёх методов и на рядах любого масштаба.


def detect_pelt(y: np.ndarray, penalty: float = 3.0, model: str = "l2") -> Detection:
    """Точный поиск при штрафе за число разладок. Без задания их количества заранее."""
    bkps = _run_ruptures(y, rpt.Pelt(model=model, min_size=3, jump=1), penalty * np.var(y))
    return Detection("pelt", bkps)


def detect_binseg(y: np.ndarray, penalty: float = 3.0, model: str = "l2") -> Detection:
    """Жадное бинарное сегментирование. Быстрое, но может промахиваться на коротких рядах."""
    algo = rpt.Binseg(model=model, min_size=3, jump=1)
    return Detection("binseg", _run_ruptures(y, algo, penalty * np.var(y)))


def detect_window(y: np.ndarray, penalty: float = 3.0, width: int = 6, model: str = "l2") -> Detection:
    """Скользящее окно: сравнивает статистики слева и справа от центра окна."""
    algo = rpt.Window(width=width, model=model, jump=1)
    return Detection("window", _run_ruptures(y, algo, penalty * np.var(y)))


def detect_bottomup(y: np.ndarray, penalty: float = 3.0, model: str = "l2") -> Detection:
    """Восходящее слияние сегментов — зеркало бинарного сегментирования."""
    algo = rpt.BottomUp(model=model, min_size=3, jump=1)
    return Detection("bottomup", _run_ruptures(y, algo, penalty * np.var(y)))


def detect_kernel(y: np.ndarray, penalty: float = 3.0) -> Detection:
    """Ядровой метод: ловит изменения в распределении, а не только в среднем.

    Штраф берётся как есть, без умножения на дисперсию: стоимость считается
    в пространстве ядра, а ширина ядра подбирается по медиане расстояний внутри
    ряда, поэтому от масштаба ряда стоимость не зависит. Зато номинальное число здесь
    несопоставимо с методами l2 — стенд калибрует его по доле ложных тревог
    (`src.cp_bench.calibrate_kernel_penalty`).
    """
    algo = rpt.KernelCPD(kernel="rbf", min_size=3)
    return Detection("kernel_rbf", _run_ruptures(y, algo, penalty))


def detect_cusum(y: np.ndarray, threshold: float = 5.0, baseline: int = 6) -> Detection:
    """CUSUM — онлайн-ориентир: накопленная сумма отклонений от базового уровня.

    Нужен именно как ориентир: если сложный метод не бьёт накопленную сумму,
    его присутствие в работе ничем не оправдано.

    Разброс оценивается через медианное абсолютное отклонение, а не через
    выборочное стандартное: на базе из шести точек обычная оценка занижается,
    порог становится слишком узким и детектор даёт ложные тревоги ещё до
    настоящей разладки. Это не теоретическое соображение — с базой в четыре
    точки и обычным std он срабатывал на позиции 7 при разладке в 12.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < baseline + 2:
        return Detection("cusum", [])

    base = y[:baseline]
    mu = float(np.median(base))
    mad = float(np.median(np.abs(base - mu)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(base)) or 1.0

    pos = neg = 0.0
    drift = 0.5 * sigma
    for i in range(baseline, len(y)):
        z = y[i] - mu
        pos = max(0.0, pos + z - drift)
        neg = min(0.0, neg + z + drift)
        if pos > threshold * sigma or -neg > threshold * sigma:
            return Detection("cusum", [i])
    return Detection("cusum", [])


DETECTORS = {
    "pelt": detect_pelt,
    "binseg": detect_binseg,
    "window": detect_window,
    "bottomup": detect_bottomup,
    "kernel_rbf": detect_kernel,
    "cusum": detect_cusum,
}
