"""Длинные ряды СберИндекса и перенос с них на короткую муниципальную панель.

## Зачем

У муниципальной панели двадцать четыре точки, и удлинить её нечем — собственный
набор СберИндекса по муниципальным образованиям ровно такой же. Но в том же
каталоге лежат федеральные ряды за семь-девять лет, собранные той же методикой
по тем же транзакциям. Годовая сезонность в них определена, а в панели — нет.

Насколько именно нет, видно на первом фолде. Обучение кончается мартом 2024,
прогнозируем апрель-июнь. Признаки требуют шести точек истории, поэтому целевыми
месяцами в обучении оказываются только июль-декабрь 2023 и январь-март 2024.
**Апреля, мая и июня в обучении нет вообще ни разу.** Панельная модель физически
не может знать, что происходит с расходами в апреле. Федеральный ряд к этому
моменту видел пять апрелей.

## Что переносится

Не уровень и не форма ряда, а **общий фактор** — множитель, на который
меняются расходы у всех муниципалитетов сразу. Связь измерена: логарифм
фактора панели объясняется логарифмом федерального фактора с R² = 0,74,
эластичность 1,23. Панель колеблется сильнее федерального агрегата, что
естественно: подушевые расходы отдельного района подвижнее страны в целом.

Конструкция переноса:

1. Выбрать модель федерального ряда **бэктестом на его собственной истории
   до origin** — это честно, вся история до origin доступна законно.
2. Спрогнозировать федеральный фактор на горизонт.
3. Перевести его в фактор панели эластичностью, оценённой на обучающих
   месяцах панели, с поправкой на то, что прогноз фактора шумный.

## Дисциплина утечки

Федеральные ряды идут до августа 2026, то есть на полтора года дальше панели.
Любое обращение к ним обрезается по origin: `series.loc[:origin]`. Это главная
ловушка всего модуля — ряд, который знает будущее, даёт прекрасные метрики
и бессмысленный результат.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path("data/reference/sberindex")

# Совокупные потребительские расходы России, декабрь 2018 — август 2026.
# Из шести длинных федеральных рядов именно этот ближе всего к нашему показателю:
# корреляция темпов роста с медианой по панели 0,91 против 0,86 у оборотов бизнеса
# и 0,62 у медианной зарплаты. Сезонно сглаженный вариант (consumper-spending-index-sa)
# не годится принципиально: сглаживание убирает ровно ту сезонность, ради которой
# длинная история и нужна, и корреляция у него уходит в минус.
AGGREGATE = ("consumer-spending", {"type": "Всего"})

# Отраслевые ряды для предобучения: каждая отрасль — отдельный длинный ряд
# той же природы (деньги в месяц), на которых можно учить форму динамики.
INDUSTRY = (
    ("oboroty-biznesa", "activity"),        # 20 отраслей × 116 месяцев
    ("izmenenie-obema-fot", "activity"),    # 20 отраслей × 113 месяцев
    ("median-wages", "activity"),           # 19 отраслей ×  79 месяцев
)


def _monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Приводит period к месячному PeriodIndex. У median-wages период — конец месяца."""
    out = frame.copy()
    out["p"] = pd.to_datetime(out["period"]).dt.to_period("M")
    return out


def load_aggregate(directory: str | Path = DIR) -> pd.Series:
    """Совокупные потребительские расходы России помесячно."""
    slug, where = AGGREGATE
    frame = _monthly(pd.read_parquet(Path(directory) / f"{slug}.parquet"))
    for column, value in where.items():
        frame = frame.loc[frame[column] == value]
    return frame.groupby("p")["value"].mean().sort_index()


def load_industry(directory: str | Path = DIR) -> dict[str, pd.Series]:
    """Отраслевые длинные ряды: ключ «набор/отрасль», значение — месячный ряд.

    Агрегат «Все отрасли» выбрасывается: он повторяет сумму остальных и в наборе
    для предобучения был бы дублем, взвешенным как обычный ряд.
    """
    out: dict[str, pd.Series] = {}
    for slug, dimension in INDUSTRY:
        path = Path(directory) / f"{slug}.parquet"
        if not path.exists():
            continue
        frame = _monthly(pd.read_parquet(path))
        for name, part in frame.groupby(dimension):
            if str(name).strip().lower() == "все отрасли":
                continue
            series = part.groupby("p")["value"].mean().sort_index()
            if len(series) >= 48 and (series > 0).all():
                out[f"{slug}/{name}"] = series
    return out


# ---------------------------------------------------------------------------
# Прогноз федерального ряда
# ---------------------------------------------------------------------------
# Кандидаты работают с логарифмом ряда: расходы меняются мультипликативно,
# и в логарифмах сезонность аддитивна, а прогноз отношения получается вычитанием.


def _seasonal_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Прошлогодний прирост того же календарного месяца, приложенный к текущему уровню."""
    n = len(history)
    if n < 13:
        return np.repeat(history[-1], horizon)
    return np.array([history[-1] + (history[n - 12 + k] - history[n - 13]) for k in range(horizon)])


def _seasonal_median(history: np.ndarray, horizon: int) -> np.ndarray:
    """Медианный по годам прирост того же месяца.

    Устойчивее предыдущего кандидата там, где один год выпадает из ряда вон:
    у апреля разброс приростов 12,1% против 1-3% у прочих месяцев — это апрель
    2020 года с закрытой торговлей. Среднее такой год утаскивает, медиана нет.
    """
    n = len(history)
    if n < 25:
        return _seasonal_naive(history, horizon)
    diffs = np.diff(history)
    months = np.arange(1, n) % 12
    out, level = [], history[-1]
    for k in range(horizon):
        same = diffs[months == (n + k) % 12]
        level = level + (np.median(same) if len(same) else np.median(diffs))
        out.append(level)
    return np.array(out)


def _ets(history: np.ndarray, horizon: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(
        history, trend="add", damped_trend=True, seasonal="add",
        seasonal_periods=12, initialization_method="estimated",
    ).fit()
    return np.asarray(model.forecast(horizon))


def _sarima(history: np.ndarray, horizon: int) -> np.ndarray:
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # SARIMAX ругается на каждую подгонку, а подгонок при отборе модели сотни.
    # Предупреждения относятся к стартовым значениям оптимизатора, не к результату,
    # и в логе прогона забивают собой всё остальное.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(history, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend="n")
        return np.asarray(model.fit(disp=False).forecast(horizon))


def _random_walk(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[-1], horizon)


CANDIDATES = {
    "seasonal_naive": _seasonal_naive,
    "seasonal_median": _seasonal_median,
    "ets": _ets,
    "sarima": _sarima,
    "random_walk": _random_walk,
}


def choose_model(history: np.ndarray, horizon: int, n_origins: int = 24) -> tuple[str, float]:
    """Отбирает модель федерального ряда бэктестом на его собственной истории.

    Отбор идёт только по данным до origin, поэтому подгонкой под тест не является.

    Возвращает имя победителя и долю дисперсии фактора, которую он объясняет:
    единица минус отношение его ошибки к ошибке ``random_walk``, а random_walk
    здесь — это в точности «фактора нет, множитель единица». Величина нужна
    дальше как мера доверия к прогнозу: если объяснено немного, поправку надо
    гасить, а не применять целиком.
    """
    scores: dict[str, list[float]] = {name: [] for name in CANDIDATES}
    n = len(history)
    for origin in range(max(36 + horizon, n - n_origins), n - horizon + 1):
        train = history[:origin]
        actual = history[origin : origin + horizon] - train[-1]
        for name, fn in CANDIDATES.items():
            try:
                predicted = fn(train, horizon) - train[-1]
            except Exception:  # кандидат, который не сошёлся, просто выбывает
                continue
            scores[name].extend((predicted - actual) ** 2)

    usable = {name: float(np.mean(v)) for name, v in scores.items() if v}
    if not usable or "random_walk" not in usable or usable["random_walk"] <= 0:
        return "random_walk", 0.0
    best = min(usable, key=usable.get)
    explained = 1.0 - usable[best] / usable["random_walk"]
    return best, float(np.clip(explained, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Общий фактор панели
# ---------------------------------------------------------------------------


@dataclass
class FactorFit:
    """Что именно получилось при подгонке — для протокола прогона, а не для отладки."""

    model: str
    multipliers: np.ndarray
    elasticity: np.ndarray
    shrinkage: float
    n_pairs: int
    n_regions: int = 0
    pooling: float = 0.0

    def as_text(self) -> str:
        mult = ", ".join(f"{m:.4f}" for m in self.multipliers)
        elas = ", ".join(f"{b:.2f}" for b in self.elasticity)
        tail = (
            f" | регионов {self.n_regions}, вес собственной оценки {self.pooling:.2f}"
            if self.n_regions
            else ""
        )
        return (
            f"федеральный ряд: {self.model} | множители [{mult}] | "
            f"эластичность [{elas}] | сжатие {self.shrinkage:.2f} | пар {self.n_pairs}{tail}"
        )


class CommonFactor:
    """Множитель общего движения панели, спрогнозированный по длинной истории.

    Эластичность оценивается на обучающих месяцах панели регрессией логарифма
    фактора панели на логарифм федерального фактора. Но в обучении федеральный
    фактор известен точно, а при прогнозе — только приблизительно, и подстановка
    шумной оценки в коэффициент, оценённый по точной, систематически переусиливает
    поправку. Поэтому эластичность сжимается в отношении дисперсии истинного
    фактора к дисперсии его прогноза; обе величины берутся из истории до origin.
    """

    def __init__(
        self, aggregate: pd.Series | None = None, min_pairs: int = 8,
        regions: dict[str, str] | None = None,
    ) -> None:
        self.aggregate = load_aggregate() if aggregate is None else aggregate
        self.min_pairs = min_pairs
        self.regions = regions or {}
        self.region_ratio: dict[str, float] = {}
        self.fit_report: FactorFit | None = None

    def fit(self, wide: pd.DataFrame, train_end: int, horizon: int) -> "CommonFactor":
        self.index = pd.to_datetime(wide.index).to_period("M")
        origin = self.index[train_end - 1]

        history = np.log(self.aggregate.loc[:origin].to_numpy(dtype=float))
        model_name, shrinkage = choose_model(history, horizon)
        forecast = CANDIDATES[model_name](history, horizon) - history[-1]

        values = wide.to_numpy(dtype=float).T
        elasticity, pairs = self._elasticity(values, self.index, train_end, horizon)

        # Логарифм федерального фактора для каждой пары (t, шаг) заранее: он один
        # и тот же для всех рядов, а спрашивают его на каждой обучающей строке.
        self._log_federal = np.full((len(self.index), horizon + 1), np.nan)
        for t in range(len(self.index)):
            start = self.aggregate.get(self.index[t], np.nan)
            for step in range(1, horizon + 1):
                if t + step >= len(self.index):
                    continue
                end = self.aggregate.get(self.index[t + step], np.nan)
                if np.isfinite(start) and np.isfinite(end) and start > 0 and end > 0:
                    self._log_federal[t, step] = np.log(end / start)

        # Коэффициент перевода один и тот же и при обучении, и при прогнозе —
        # иначе из цели вычиталось бы одно, а прибавлялось обратно другое.
        self.scale = elasticity * shrinkage
        self.log_forecast = forecast

        pooling = 0.0
        if self.regions:
            pooling = self._regional_ratios(wide.columns, values, train_end, horizon)

        self.multipliers = np.exp(self.scale * forecast)
        self.fit_report = FactorFit(
            model_name, self.multipliers, elasticity, shrinkage, pairs,
            n_regions=len(self.region_ratio), pooling=pooling,
        )
        return self

    def _regional_ratios(
        self, columns: pd.Index, values: np.ndarray, train_end: int, horizon: int
    ) -> float:
        """Насколько регион чувствительнее или спокойнее страны в целом.

        Эластичность оценивается по региону целиком (все шаги горизонта вместе —
        на пятнадцати месяцах обучения по каждому шагу отдельно вышло бы по дюжине
        точек) и выражается отношением к общей: Якутия примерно вдвое подвижнее
        Татарстана, и это свойство устойчивое, а не шум — оценки по первой и второй
        половине периода коррелируют на 0,77.

        Собственная оценка региона смешивается с общей по Джеймсу-Стайну: вес
        считается из того, насколько разброс эластичностей между регионами больше
        ошибки их оценки. Если регионы на самом деле одинаковы, вес уходит в ноль
        и поправка исчезает сама.
        """
        by_region: dict[str, list[int]] = {}
        for position, column in enumerate(columns):
            region = self.regions.get(column)
            if isinstance(region, str) and region:
                by_region.setdefault(region, []).append(position)

        pooled_slope, _ = self._slope(values, np.arange(values.shape[0]), train_end, horizon)
        if not np.isfinite(pooled_slope) or pooled_slope == 0:
            return 0.0

        slopes, errors, names = [], [], []
        for region, members in by_region.items():
            if len(members) < 8:
                continue
            slope, variance = self._slope(values, np.asarray(members), train_end, horizon)
            if np.isfinite(slope) and np.isfinite(variance) and variance > 0:
                slopes.append(slope)
                errors.append(variance)
                names.append(region)
        if len(slopes) < 10:
            return 0.0

        slopes, errors = np.asarray(slopes), np.asarray(errors)
        between = float(np.var(slopes, ddof=1) - np.mean(errors))
        weight = float(np.clip(between / (between + np.mean(errors)), 0.0, 1.0)) if between > 0 else 0.0
        shrunk = weight * slopes + (1.0 - weight) * pooled_slope
        self.region_ratio = dict(zip(names, shrunk / pooled_slope))
        return weight

    def _slope(
        self, values: np.ndarray, members: np.ndarray, train_end: int, horizon: int
    ) -> tuple[float, float]:
        """Наклон регрессии логарифма фактора группы на логарифм федерального.

        Возвращает наклон и дисперсию его оценки — вторая нужна для смешивания.
        """
        x, y = [], []
        for step in range(1, horizon + 1):
            for t in range(train_end - step):
                federal = self._log_federal[t, step]
                if not np.isfinite(federal):
                    continue
                ratio = np.nanmedian(values[members, t + step] / values[members, t])
                if ratio > 0:
                    x.append(federal)
                    y.append(np.log(ratio))
        if len(x) < self.min_pairs:
            return np.nan, np.nan
        x_arr, y_arr = np.asarray(x), np.asarray(y)
        centred = x_arr - x_arr.mean()
        denominator = float((centred**2).sum())
        if denominator <= 0:
            return np.nan, np.nan
        slope = float((centred * (y_arr - y_arr.mean())).sum() / denominator)
        residual = y_arr - y_arr.mean() - slope * centred
        dof = max(len(x_arr) - 2, 1)
        return slope, float((residual**2).sum() / dof / denominator)

    def multipliers_for(self, columns: pd.Index) -> np.ndarray:
        """Множители на каждый ряд: матрица (рядов × горизонт)."""
        if not self.region_ratio:
            return np.tile(self.multipliers, (len(columns), 1))
        ratios = np.array(
            [self.region_ratio.get(self.regions.get(column), 1.0) for column in columns]
        )
        return np.exp(ratios[:, None] * (self.scale * self.log_forecast)[None, :])

    def realised(self, t: int, step: int, column: str | None = None) -> float:
        """Множитель для обучающей пары (t, t+step) по наблюдённому федеральному ряду.

        Вызывается только на парах внутри обучающей части, поэтому обращение
        к федеральному ряду в момент t+step законно: этот месяц уже прошёл.
        """
        federal = self._log_federal[t, step] if step < self._log_federal.shape[1] else np.nan
        if not np.isfinite(federal):
            return 1.0
        ratio = self.region_ratio.get(self.regions.get(column), 1.0) if self.region_ratio else 1.0
        return float(np.exp(ratio * self.scale[step - 1] * federal))

    def _elasticity(
        self, values: np.ndarray, index: pd.PeriodIndex, train_end: int, horizon: int
    ) -> tuple[np.ndarray, int]:
        """Коэффициент перевода федерального фактора в панельный, по шагам горизонта.

        Оценивается только на парах, целиком лежащих в обучающей части. При нехватке
        пар возвращает единицу — то есть федеральный фактор применяется как есть.
        """
        out, total = [], 0
        for step in range(1, horizon + 1):
            x, y = [], []
            for t in range(train_end - step):  # t + step <= train_end - 1, то есть внутри обучения
                base, target = values[:, t], values[:, t + step]
                ratio = np.nanmedian(target / base)
                federal = self.aggregate.get(index[t + step], np.nan) / self.aggregate.get(
                    index[t], np.nan
                )
                if ratio > 0 and np.isfinite(federal) and federal > 0:
                    x.append(np.log(federal))
                    y.append(np.log(ratio))
            total += len(x)
            if len(x) < self.min_pairs:
                out.append(1.0)
                continue
            x_arr, y_arr = np.asarray(x), np.asarray(y)
            centred = x_arr - x_arr.mean()
            denominator = float((centred**2).sum())
            out.append(float((centred * (y_arr - y_arr.mean())).sum() / denominator)
                       if denominator > 0 else 1.0)
        return np.asarray(out), total


# ---------------------------------------------------------------------------
# Признаки из длинных рядов (подход «внешние регрессоры»)
# ---------------------------------------------------------------------------


class ExternalFeatures:
    """Значения длинных рядов на момент t — как дополнительные признаки модели.

    Все эти ряды федеральные, то есть в месяце t одинаковы для всех 2028
    муниципалитетов. Признак, постоянный в сечении, различает не ряды, а моменты
    времени, и при семи различных моментах в обучающей части работает как метка
    месяца. Проверить всё равно надо — но результат следует читать именно так.
    """

    SOURCES = {
        "consumer-spending": {"type": "Всего"},
        "median-wages": {"activity": "Все отрасли"},
        "oboroty-biznesa": {"activity": "Все отрасли"},
        "izmenenie-obema-fot": {"activity": "Все отрасли"},
    }

    def __init__(self, directory: str | Path = DIR, key_rate: str | Path | None = None) -> None:
        self.series: dict[str, pd.Series] = {}
        for slug, where in self.SOURCES.items():
            path = Path(directory) / f"{slug}.parquet"
            if not path.exists():
                continue
            frame = _monthly(pd.read_parquet(path))
            for column, value in where.items():
                frame = frame.loc[frame[column] == value]
            self.series[slug] = frame.groupby("p")["value"].mean().sort_index()

        rate_path = Path(key_rate) if key_rate else Path("data/reference/key_rate.parquet")
        if rate_path.exists():
            rate = _monthly(pd.read_parquet(rate_path))
            self.series["key_rate_real"] = rate.groupby("p")["value"].mean().sort_index()

    def row(self, index: pd.PeriodIndex, t: int) -> dict[str, float]:
        """Темпы роста длинных рядов к моменту t. Уровни намеренно не берутся:
        в них тренд, который дерево запомнит как номер месяца."""
        out: dict[str, float] = {}
        period = index[t]
        for name, series in self.series.items():
            current = series.get(period, np.nan)
            if not np.isfinite(current):
                continue
            for lag in (1, 3, 12):
                previous = series.get(period - lag, np.nan)
                if np.isfinite(previous) and previous != 0:
                    # Ставка уже в процентных пунктах: её сравнивают разностью, не отношением.
                    out[f"{name}_d{lag}"] = (
                        current - previous if name == "key_rate_real" else current / previous - 1.0
                    )
        return out
