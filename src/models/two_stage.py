"""Двухэтапный прогноз: одно федеральное число и разнос по муниципалитетам.

Центральный тезис работы — «задача только выглядит как прогноз двух тысяч
муниципалитетов, на деле это прогноз одного числа и разнос». Он подтверждён
измерением: межрядовая составляющая ошибки лежит в 695–849 рублях по всем
пятнадцати сочетаниям модели и фолда при стандартном отклонении 45, тогда как
сама MAE гуляет от 805 до 3 535. Всё, что двигается, — это попадание в общее
движение страны.

Но измерение — не конструкция. В `GlobalGBM` общий фактор заходит нормировкой
цели, и бустинг волен им не пользоваться. Здесь тезис проверяется в лоб: модель
**вообще не смотрит на отдельные ряды при прогнозе динамики**. Она прогнозирует
один федеральный ряд по его собственной длинной истории и разносит результат
долями, оценёнными на обучающей части.

## Два этапа

1. **Федеральный агрегат.** Модель выбирается бэктестом на его собственной
   истории до origin — 64–70 месяцев против наших 15–21. Ровно та длина,
   нехватка которой, по измеренному в этой работе закону, всё и определяет.
2. **Разнос.** Доля муниципалитета `y_i(t) / F(t)` оценивается по обучающей
   части, прогноз — произведение доли на прогноз агрегата.

## Чем этот подход ограничен по построению

Постоянная доля означает эластичность ровно единицу: панель обязана двигаться
с той же силой, что и страна. На деле эластичность 1,23 — подушевые расходы
района подвижнее национального агрегата. Значит модель систематически
недокорректирует, и это не дефект реализации, а свойство конструкции.
Сравнение с `global_gbm_factor`, где эластичность оценивается, показывает
цену этого упрощения.

## Какой агрегат брать

Сезонно сглаженный индекс (`consumper-spending-index-sa`) для разноса долями
не годится, и это видно до всякого прогона: сглаживание вычищает именно ту
сезонность, которую разнос делит между муниципалитетами. Разделив панель на ряд
без сезонности, мы оставляем всю её сезонность внутри доли, и доля перестаёт
быть устойчивой. Обе версии зарегистрированы отдельными моделями, чтобы
число подтвердило рассуждение (`two_stage` против `two_stage_sa` в сводке).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Оценщики доли. Все смотрят только назад от origin.
SHARE_RULES = {
    "last": 1,      # доля последнего месяца
    "median3": 3,   # медиана долей за три месяца
    "median6": 6,   # медиана за шесть — устойчивее к выбросу в одном месяце
}


class TwoStage:
    """Прогноз федерального агрегата по длинной истории, разнос по долям."""

    name = "two_stage"

    def __init__(self, slug: str | None = None, where: dict[str, str] | None = None) -> None:
        self.slug = slug
        self.where = where
        self.notes: list[str] = []
        self.context = None

    def set_context(self, context) -> None:
        self.context = context

    # -- служебное ---------------------------------------------------------

    def _aggregate(self) -> pd.Series:
        from src.external import DIR, load_aggregate

        directory = DIR
        if self.slug is None and self.context is not None and self.context.aggregate is not None:
            return self.context.aggregate
        return load_aggregate(directory, self.slug, self.where)

    def _forecast_aggregate(self, index: pd.PeriodIndex, origin_pos: int, horizon: int):
        """Прогноз агрегата на horizon шагов вперёд от origin. Только прошлое."""
        from src.external import CANDIDATES, choose_model

        history_series = self.aggregate.loc[: index[origin_pos]]
        history = np.log(history_series.to_numpy(dtype=float))
        name, _explained = choose_model(history, horizon)
        forecast = CANDIDATES[name](history, horizon)
        return name, np.exp(forecast), float(history_series.iloc[-1])

    @staticmethod
    def _shares(values: np.ndarray, aggregate_window: np.ndarray, window: int) -> np.ndarray:
        """Доля каждого ряда в агрегате, усреднённая по последним `window` месяцам."""
        ratio = values[:, -window:] / aggregate_window[None, -window:]
        return np.nanmedian(ratio, axis=1)

    def _predict_at(
        self, values: np.ndarray, index: pd.PeriodIndex, origin_pos: int,
        horizon: int, window: int,
    ) -> np.ndarray:
        _name, forecast, _last = self._forecast_aggregate(index, origin_pos, horizon)
        aggregate_window = self.aggregate.reindex(index[: origin_pos + 1]).to_numpy(dtype=float)
        shares = self._shares(values[:, : origin_pos + 1], aggregate_window, window)
        return shares[:, None] * forecast[None, :]

    # -- обучение и прогноз ------------------------------------------------

    def fit(self, wide: pd.DataFrame, train_end: int, horizon: int) -> "TwoStage":
        self.aggregate = self._aggregate()
        index = pd.to_datetime(wide.index).to_period("M")
        values = wide.to_numpy(dtype=float).T
        self.notes = []

        # Правило оценки доли выбирается на последних месяцах обучающей части:
        # origin сдвигается на горизонт назад, и модель прогнозирует месяцы,
        # которых при этом не видит. По тесту не подбирается ничего.
        inner_origin = train_end - horizon - 1
        scores: dict[str, float] = {}
        if inner_origin >= max(SHARE_RULES.values()) - 1:
            actual = values[:, inner_origin + 1 : inner_origin + 1 + horizon]
            for rule, window in SHARE_RULES.items():
                try:
                    predicted = self._predict_at(values, index, inner_origin, horizon, window)
                except Exception as exc:
                    self.notes.append(f"правило {rule}: не сработало ({str(exc)[:60]})")
                    continue
                scores[rule] = float(np.nanmean(np.abs(predicted - actual)))

        self.rule = min(scores, key=scores.get) if scores else "last"
        self.window = SHARE_RULES[self.rule]

        model_name, forecast, last_level = self._forecast_aggregate(index, train_end - 1, horizon)
        self.forecast = forecast
        self.notes.append(
            f"агрегат: {self.slug or 'consumer-spending'} | модель {model_name} | "
            f"множители [{', '.join(f'{v / last_level:.4f}' for v in forecast)}] | "
            f"доля по правилу {self.rule} из "
            + (", ".join(f"{k}: {v:.0f}" for k, v in sorted(scores.items())) or "нет кандидатов")
        )
        return self

    def predict(self, wide: pd.DataFrame, train_end: int, horizon: int) -> np.ndarray:
        index = pd.to_datetime(wide.index).to_period("M")
        values = wide.to_numpy(dtype=float).T
        aggregate_window = self.aggregate.reindex(index[:train_end]).to_numpy(dtype=float)
        shares = self._shares(values[:, :train_end], aggregate_window, self.window)
        return shares[:, None] * self.forecast[None, :]


# ---------------------------------------------------------------------------
# Новостная поправка к первому этапу
# ---------------------------------------------------------------------------

NEWS_PATH = "data/news/national.parquet"


class TwoStageNews(TwoStage):
    """Двухэтапная модель с новостной поправкой к прогнозу федерального агрегата.

    Региональные новости проверялись против муниципальных рядов и дали ноль
    трижды. Но разложение ошибки показало, что межрядовая составляющая не
    двигается ничем, а весь выигрыш сидит в попадании в общероссийское движение.
    Значит и проверять новости надо там же — на национальном уровне, против
    того самого федерального ряда, который прогнозирует первый этап.

    **Конструкция.** Сезонный прогноз агрегата остаётся как есть, новости лишь
    правят его остаток. Для каждого шага горизонта остаток
    `log F(t+h) − log F(t) − прогноз, сделанный в момент t` регрессируется
    на один новостной признак по месяцам, где есть и агрегат, и корпус.

    **Признак выбирается перекрёстной проверкой с выбрасыванием по одному**
    и только среди месяцев до origin. Условие жёсткое: поправка применяется,
    лишь если выбранный признак на этой проверке бьёт отсутствие поправки.
    Иначе прогноз остаётся прежним, и модель совпадает с `two_stage`.

    **Главное ограничение известно заранее и числом.** Агрегат идёт с декабря
    2018-го, корпус — с января 2023-го. На первом фолде сезонный прогноз опирается
    на 64 месяца, а регрессия остатков — на 12 пар. Это тот же закон о длине
    истории, только теперь он ограничивает саму проверку новостей.
    """

    name = "two_stage_news"

    def __init__(
        self, slug: str | None = None, where: dict[str, str] | None = None,
        news_path: str = NEWS_PATH,
    ) -> None:
        super().__init__(slug, where)
        self.news_path = news_path

    def _news(self) -> pd.DataFrame:
        from pathlib import Path

        path = Path(self.news_path)
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        frame["month"] = pd.PeriodIndex(frame["month"], freq="M")
        return frame.set_index("month")

    def _base_step(self, history: np.ndarray, horizon: int, model_name: str) -> np.ndarray:
        from src.external import CANDIDATES

        return CANDIDATES[model_name](history, horizon) - history[-1]

    def _residuals(self, origin: pd.Period, horizon: int, model_name: str):
        """Остатки сезонного прогноза по месяцам, где есть новости. Только прошлое."""
        log_series = np.log(self.aggregate.loc[:origin])
        periods = list(log_series.index)
        rows = {step: {} for step in range(1, horizon + 1)}
        for position, period in enumerate(periods):
            if period not in self.news.index or position < 25:
                continue  # сезонной оценке нужно не меньше двух лет истории
            history = log_series.to_numpy(dtype=float)[: position + 1]
            predicted = self._base_step(history, horizon, model_name)
            for step in range(1, horizon + 1):
                if position + step >= len(periods) or periods[position + step] > origin:
                    continue
                actual = float(log_series.iloc[position + step] - log_series.iloc[position])
                rows[step][period] = actual - float(predicted[step - 1])
        return rows

    @staticmethod
    def _candidates(columns) -> list[str]:
        """Кандидаты поправки — доли тем. Интенсивности среди них нет: она нормирована
        средним и разбросом числа публикаций издания за весь период, то есть заглядывает
        вперёд, и ни разу не выбиралась."""
        return [c for c in columns if c.startswith("t_")]

    @staticmethod
    def _loo_gain(x: np.ndarray, y: np.ndarray) -> float:
        """Насколько регрессия лучше нуля при проверке с выбрасыванием по одному.

        Сравнение именно с нулём, а не со средним: базовый прогноз уже несмещён
        по построению, и «поправка на константу» была бы подгонкой уровня.
        """
        errors, null_errors = [], []
        for i in range(len(x)):
            mask = np.arange(len(x)) != i
            xi, yi = x[mask], y[mask]
            centred = xi - xi.mean()
            denominator = float((centred**2).sum())
            if denominator <= 0:
                return -np.inf
            slope = float((centred * (yi - yi.mean())).sum() / denominator)
            intercept = float(yi.mean() - slope * xi.mean())
            errors.append((y[i] - (intercept + slope * x[i])) ** 2)
            null_errors.append(y[i] ** 2)
        return float(np.mean(null_errors) - np.mean(errors))

    def fit(self, wide: pd.DataFrame, train_end: int, horizon: int) -> "TwoStageNews":
        super().fit(wide, train_end, horizon)
        self.news = self._news()
        self.correction = np.zeros(horizon)
        if self.news.empty:
            self.notes.append("новостей нет, поправка не применяется")
            return self

        index = pd.to_datetime(wide.index).to_period("M")
        origin = index[train_end - 1]
        model_name = self.notes[0].split("модель ")[1].split(" |")[0]
        residuals = self._residuals(origin, horizon, model_name)
        features = self._candidates(self.news.columns)

        chosen, gains = {}, {}
        for step in range(1, horizon + 1):
            months = sorted(residuals[step])
            if len(months) < 8:
                continue
            y = np.array([residuals[step][m] for m in months])
            best_name, best_gain, best_pair = None, 0.0, None
            for name in features:
                x = self.news.loc[months, name].to_numpy(dtype=float)
                if not np.isfinite(x).all() or x.std() == 0:
                    continue
                gain = self._loo_gain(x, y)
                if gain > best_gain:
                    centred = x - x.mean()
                    slope = float((centred * (y - y.mean())).sum() / float((centred**2).sum()))
                    best_name, best_gain = name, gain
                    best_pair = (float(y.mean() - slope * x.mean()), slope)
            if best_name is not None and origin in self.news.index:
                intercept, slope = best_pair
                self.correction[step - 1] = intercept + slope * float(
                    self.news.loc[origin, best_name]
                )
                chosen[step] = best_name
                gains[step] = best_gain

        self.forecast = self.forecast * np.exp(self.correction)
        if chosen:
            self.notes.append(
                "новостная поправка: "
                + ", ".join(
                    f"шаг {s}: {chosen[s]} ({np.exp(self.correction[s - 1]):.4f}×)"
                    for s in sorted(chosen)
                )
                + f" | месяцев с новостями в обучении {len(residuals[1])}"
            )
        else:
            self.notes.append(
                "новостная поправка не применена: ни один признак не прошёл "
                f"проверку с выбрасыванием по одному (месяцев {len(residuals[1])})"
            )
        return self


# ---------------------------------------------------------------------------
# Разнос известного агрегата: наукаст в узком смысле и оракул общего движения
# ---------------------------------------------------------------------------


class TwoStageKnownAggregate(TwoStage):
    """Первый этап заменён фактом: агрегат за месяцы горизонта берётся из ряда как есть.

    Федеральный индекс потребительских расходов СберИндекс публикует регулярно
    и с коротким лагом, а муниципальный разрез выложен разово, к конкурсу. Значит
    в настоящей задаче наукаста агрегат за текущий месяц уже известен, когда
    муниципальных значений ещё нет, и прогнозировать надо только разнос. Здесь
    так и сделано: доли оцениваются по прошлому, как у `two_stage`, а множится
    на них не прогноз агрегата, а его опубликованное значение.

    На горизонте 1 это наукаст в узком смысле слова. На горизонтах длиннее —
    уже не прогноз, а оракул: общее движение страны на весь горизонт считается
    известным, и в ошибке остаётся одна межрядовая составляющая. То есть та
    самая нижняя граница из разложения ошибки, только построенная заранее,
    а не вычтенная задним числом по факту.

    Ряд агрегата читается **после** origin намеренно, и это единственная модель
    в проекте, которой это позволено. В основной реестр она не входит и с
    прогнозными моделями за одно место не соревнуется; живёт в прогоне по
    горизонтам (`scripts/horizons.py`) и в прогнозе вперёд (`scripts/forecast_forward.py`),
    где ей одной передаётся полный агрегат — разнос опубликованного агрегата
    за 2025 год.
    """

    name = "two_stage_known"

    def _forecast_aggregate(self, index: pd.PeriodIndex, origin_pos: int, horizon: int):
        future = index[origin_pos + 1 : origin_pos + 1 + horizon]
        values = self.aggregate.reindex(future).to_numpy(dtype=float)
        if len(values) < horizon or not np.isfinite(values).all():
            raise ValueError(f"агрегат за месяцы {future.min()}..{future.max()} неизвестен")
        return "известный агрегат", values, float(self.aggregate.loc[index[origin_pos]])
