"""Фундаментальные модели временных рядов — предобученные, работают без дообучения.

Почему они здесь уместны именно на наших данных. Классические модели упёрлись в потолок
потому, что пятнадцати точек мало для оценки параметров. Фундаментальная модель параметры
не оценивает: она уже обучена на миллионах чужих рядов и переносит эту статистику на наш.
Короткая история перестаёт быть препятствием — она становится просто контекстом.

Взяты только открытые веса. TimeGPT и прочие платные API исключены пунктом 10.3
Положения, запрещающим проприетарные технологии, требующие возмездного приобретения прав.
"""
from __future__ import annotations

import numpy as np

_PIPELINES: dict[str, object] = {}


def _pipeline(model_id: str):
    """Модель грузится один раз на процесс — веса весят сотни мегабайт."""
    if model_id not in _PIPELINES:
        import torch
        from chronos import BaseChronosPipeline

        _PIPELINES[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32
        )
    return _PIPELINES[model_id]


class Chronos:
    """Chronos-Bolt от Amazon: zero-shot прогноз, дообучения не требует.

    Берётся медиана предсказательного распределения, а не среднее: распределение
    у Chronos асимметрично, а оптимальной точечной оценкой под MAE является именно
    медиана. Под MAE брать среднее — систематически терять в метрике, по которой
    нас и оценивают.
    """

    def __init__(self, size: str = "small", transform: str = "none") -> None:
        self.model_id = f"amazon/chronos-bolt-{size}"
        self.transform = transform
        self.name = f"chronos_{size}" + ("" if transform == "none" else f"_{transform}")

    def fit(self, y: np.ndarray) -> "Chronos":
        y = np.asarray(y, dtype=float)
        # Из коробки Chronos проигрывает наивной модели: он инвариантен к масштабу и
        # не воспроизводит инфляционный рост в 15-17% годовых - на Орле предсказал
        # плавное снижение там, где факт рос. Логарифм превращает мультипликативный
        # тренд в аддитивный, который модель переносит куда увереннее.
        self._logged = self.transform == "log" and np.all(y > 0)
        self._context = np.log(y) if self._logged else y
        return self

    def predict(self, horizon: int) -> np.ndarray:
        import torch

        pipeline = _pipeline(self.model_id)
        context = torch.tensor(self._context, dtype=torch.float32).unsqueeze(0)
        quantiles, _mean = pipeline.predict_quantiles(
            context, prediction_length=horizon, quantile_levels=[0.5]
        )
        out = quantiles[0, :, 0].numpy().astype(float)
        return np.exp(out) if self._logged else out
