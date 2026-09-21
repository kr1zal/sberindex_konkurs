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
не годится, и это видно до всякого прогона. Сезонный размах сырого агрегата —
33,7 процентных пункта между январём и декабрём, у сглаженного 1,6, у самой
панели 44. Разделив панель на ряд без сезонности, мы оставляем всю её сезонность
внутри доли, и доля перестаёт быть устойчивой: коэффициент вариации 0,087
против 0,051 у сырого. Обе версии зарегистрированы отдельными моделями, чтобы
число подтвердило рассуждение.
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
