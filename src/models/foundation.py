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


# ---------------------------------------------------------------------------
# Панельный Chronos: zero-shot и дообучение
# ---------------------------------------------------------------------------


class ChronosPanel:
    """Chronos-Bolt, применённый ко всей панели сразу, с необязательным дообучением.

    Zero-shot Chronos проигрывает наивной модели (2 462 против 1 852). Очевидный
    следующий шаг — дообучить его на наших рядах, и именно его сделала бы сильная
    команда, поэтому отрицательный вывод без этой проверки был бы преждевременным.

    Почему отдельный класс, а не флаг у `Chronos`. Дообученная модель одна на весь
    фолд, а не своя на каждый ряд: 2 028 отдельных дообучений на пятнадцати точках
    каждое — это не дообучение, а переобучение. Значит модель панельная и живёт
    в `GLOBAL_MODELS`, как `GlobalGBM`.

    **Дисциплина.** Дообучение видит только обучающую часть фолда: контекст
    обрывается на `train_end`, целевые окна целиком лежат внутри него. По тестовым
    месяцам не подбирается ничего.

    **Валидация только по времени.** Сначала ранняя остановка была сделана по
    отложенным рядам: муниципалитетов две тысячи, десятой части не жалко, а каждый
    из пятнадцати месяцев на счету. Это оказалось неверно. Потеря на отложенных
    рядах уверенно падала, а MAE на тестовых месяцах росла — потому что сдвиг
    здесь во времени, а не в сечении. Отложенные ряды живут в тех же месяцах,
    что и обучающие, и распределение у них то же самое; увидеть, что модель
    заучивает декабрьско-январские переходы и потащит их в апрель, они не могут
    в принципе. Валидация отрезается по времени: последние `horizon` месяцев
    обучающей части, то есть ровно та задача, которую предстоит решать.
    """

    name = "chronos_panel"

    def __init__(
        self, size: str = "small", finetune: bool = False,
        learning_rates: tuple[float, ...] = (3e-5, 1e-4), max_steps: int = 600,
        batch_size: int = 64, patience: int = 3, seed: int = 20260920,
    ) -> None:
        self.model_id = f"amazon/chronos-bolt-{size}"
        self.finetune = finetune
        self.learning_rates = learning_rates
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.notes: list[str] = []
        self._pipeline = None

    # -- служебное ---------------------------------------------------------

    @staticmethod
    def _device():
        import torch

        # MPS на Apple Silicon даёт ускорение в разы, но не везде есть; молча
        # падать на CPU нельзя — время прогона отличается на порядок, и это надо
        # видеть в логе.
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _samples(self, wide, train_end: int, horizon: int):
        """Пары (контекст, цель) из обучающей части и момент origin для каждой.

        Контекст растущий, как при прогнозе: модель должна привыкнуть к тому,
        что истории мало, а не к тому, что её всегда пятнадцать месяцев.
        """
        import torch

        values = wide.to_numpy(dtype=float).T
        contexts, targets, origins = [], [], []
        for series in values:
            if not np.isfinite(series[:train_end]).all():
                continue
            for t in range(horizon, train_end - horizon):
                contexts.append(series[: t + 1])
                targets.append(series[t + 1 : t + 1 + horizon])
                origins.append(t)
        if not contexts:
            # Окно дообучения — горизонт контекста плюс горизонт цели, и обоим
            # надо уместиться в обучающую часть. При обучении 12 и горизонте 6
            # не остаётся ни одного примера; пусть отказ будет назван, а не
            # выглядеть как ошибка max() на пустом списке.
            raise ValueError(
                f"нет обучающих окон: обучение {train_end} мес короче двух горизонтов по {horizon}"
            )
        width = max(len(c) for c in contexts)
        padded = np.full((len(contexts), width), np.nan)
        for i, c in enumerate(contexts):
            padded[i, width - len(c) :] = c  # Chronos ждёт выравнивание вправо
        return (
            torch.tensor(padded, dtype=torch.float32),
            torch.tensor(np.asarray(targets), dtype=torch.float32),
            np.asarray(origins),
        )

    # -- обучение и прогноз ------------------------------------------------

    def fit(self, wide, train_end: int, horizon: int) -> "ChronosPanel":
        self.notes = []
        self._pipeline = _pipeline(self.model_id)
        if not self.finetune:
            return self

        import numpy as _np

        context, target, origins = self._samples(wide, train_end, horizon)
        # Валидация — последние horizon месяцев обучающей части. Ровно та задача,
        # что и на тесте: спрогнозировать месяцы, которых модель ещё не видела.
        cutoff = origins.max() - horizon + 1
        train_idx = _np.flatnonzero(origins < cutoff)
        valid_idx = _np.flatnonzero(origins >= cutoff)
        if len(train_idx) == 0 or len(valid_idx) == 0:
            self.notes.append("дообучение пропущено: не хватает месяцев на валидацию по времени")
            return self

        best = None
        for learning_rate in self.learning_rates:
            state, loss, steps = self._train_one(context, target, train_idx, valid_idx, learning_rate)
            self.notes.append(
                f"дообучение chronos, скорость {learning_rate:g}: {steps} шагов, "
                f"потеря на отложенных месяцах {loss:.4f}"
            )
            if best is None or loss < best[1]:
                best = (state, loss, learning_rate)

        import copy

        pipeline = copy.deepcopy(self._pipeline)
        if best[0] is not None:
            pipeline.model.load_state_dict(best[0])
        pipeline.model.eval()
        self._pipeline = pipeline
        self.notes.append(
            f"выбрана скорость {best[2]:g}, {len(train_idx)} обучающих примеров, "
            f"{len(valid_idx)} проверочных, устройство {self._device()}"
        )
        return self

    def _train_one(self, context, target, train_idx, valid_idx, learning_rate):
        """Одна скорость обучения. Возвращает лучшее состояние, его потерю и число шагов."""
        import copy

        import torch

        device = self._device()
        # Сид нужен не только выборке батчей. В режиме обучения у T5 работает
        # dropout, и он берёт глобальный генератор torch — без этой строки два
        # прогона на одних данных давали 1 606 и 1 772 на трёх фолдах протокола,
        # а на третьем фолде расходились на 555 рублей. Сидируется и MPS.
        torch.manual_seed(self.seed)
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(self.seed)
        model = copy.deepcopy(self._pipeline).model.to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(self.seed)

        def validation_loss() -> float:
            model.eval()
            losses = []
            with torch.no_grad():
                for start in range(0, len(valid_idx), 256):
                    chunk = valid_idx[start : start + 256]
                    losses.append(float(model(
                        context=context[chunk].to(device),
                        mask=(~torch.isnan(context[chunk])).to(device),
                        target=target[chunk].to(device),
                    ).loss))
            model.train()
            return float(np.mean(losses))

        # Нулевой шаг — это zero-shot. Если дообучение ничего не даёт, ранняя
        # остановка обязана вернуть исходную модель, а не худшую из обученных.
        best_loss = validation_loss()
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad_rounds, step, check_every = 0, 0, 25

        model.train()
        while step < self.max_steps:
            batch = train_idx[
                torch.randint(len(train_idx), (self.batch_size,), generator=generator).numpy()
            ]
            model(
                context=context[batch].to(device),
                mask=(~torch.isnan(context[batch])).to(device),
                target=target[batch].to(device),
            ).loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            step += 1
            if step % check_every:
                continue
            current = validation_loss()
            if current < best_loss - 1e-5:
                best_loss, bad_rounds = current, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_rounds += 1
                if bad_rounds >= self.patience:
                    break
        return best_state, best_loss, step

    def predict(self, wide, train_end: int, horizon: int) -> np.ndarray:
        import torch

        values = wide.to_numpy(dtype=float).T
        contexts = [torch.tensor(s[:train_end], dtype=torch.float32) for s in values]
        out = np.full((len(contexts), horizon), np.nan)
        device = self._pipeline.model.device

        for start in range(0, len(contexts), 256):
            chunk = contexts[start : start + 256]
            with torch.no_grad():
                quantiles, _mean = self._pipeline.predict_quantiles(
                    [c.to(device) for c in chunk],
                    prediction_length=horizon,
                    quantile_levels=[0.5],
                )
            out[start : start + len(chunk)] = quantiles[:, :, 0].cpu().numpy()
        return out


class TimesFM:
    """TimesFM 2.5 от Google — второе семейство фундаментальных моделей.

    Второе семейство нужно не для полноты списка. Chronos zero-shot проиграл
    наивной модели, и по одному семейству нельзя отличить «фундаментальные модели
    не годятся для таких рядов» от «не годится именно Chronos». Архитектуры разные:
    Chronos-Bolt — энкодер-декодер на патчах поверх T5, TimesFM — декодер на патчах
    с собственным квантильным выходом. Совпадение выводов у двух разных архитектур
    гораздо убедительнее, чем у одной.

    Берётся медиана, а не точечный прогноз модели: последний соответствует среднему,
    а под MAE оптимальна медиана. Выход `forecast` отдаёт десять колонок — среднее
    и девять децилей, медиана среди них пятая.

    Дообучения нет намеренно: публичный интерфейс TimesFM 2.5 его не предоставляет,
    и самодельный цикл поверх приватных внутренностей сравнивался бы не с той
    моделью, которую может воспроизвести проверяющий.
    """

    name = "timesfm"
    REPO = "google/timesfm-2.5-200m-pytorch"
    _MODEL = None
    _COMPILED_FOR: tuple[int, int] | None = None

    def __init__(self, max_context: int = 64) -> None:
        self.max_context = max_context

    @classmethod
    def _model(cls, max_context: int, horizon: int):
        """Модель грузится и компилируется один раз на процесс: веса 200M параметров."""
        import timesfm as _timesfm

        if cls._MODEL is None:
            cls._MODEL = _timesfm.TimesFM_2p5_200M_torch.from_pretrained(cls.REPO)
        if cls._COMPILED_FOR != (max_context, horizon):
            cls._MODEL.compile(
                _timesfm.ForecastConfig(
                    max_context=max_context, max_horizon=max(horizon, 8), normalize_inputs=True
                )
            )
            cls._COMPILED_FOR = (max_context, horizon)
        return cls._MODEL

    def fit(self, wide, train_end: int, horizon: int) -> "TimesFM":
        self._model(self.max_context, horizon)
        self.notes = [f"timesfm {self.REPO}, контекст до {self.max_context}, без дообучения"]
        return self

    def predict(self, wide, train_end: int, horizon: int) -> np.ndarray:
        model = self._model(self.max_context, horizon)
        values = wide.to_numpy(dtype=float).T
        contexts = [np.asarray(s[:train_end], dtype=float) for s in values]
        out = np.full((len(contexts), horizon), np.nan)

        for start in range(0, len(contexts), 256):
            chunk = contexts[start : start + 256]
            _point, quantiles = model.forecast(horizon=horizon, inputs=chunk)
            # Колонка 0 — среднее, дальше девять децилей; медиана пятая по счёту.
            out[start : start + len(chunk)] = np.asarray(quantiles)[:, :horizon, 5]
        return out


class Moirai:
    """Moirai 1.1-R от Salesforce — третье семейство фундаментальных моделей.

    Два семейства уже показали одно и то же: без дообучения фундаментальная
    модель проигрывает наивной. Третье нужно ровно для того, чтобы этот вывод
    перестал зависеть от выбора двух конкретных архитектур. Moirai отличается
    от обоих: это маскированный энкодер, обученный сразу на многих разрешениях,
    с вероятностным выходом в виде выборки, а не квантилей.

    Точечная оценка — медиана выборки: под MAE оптимальна именно она.

    Размер патча выбирается по **отложенным месяцам обучающей части**: origin
    сдвигается на горизонт назад, кандидаты сравниваются на последних месяцах,
    которые модель при этом не видит. На контексте в пятнадцать точек выбор
    между патчами в 8 и в 32 точки решает многое, а подбирать его по тесту
    нельзя. Встроенный режим `patch_size="auto"` на наших формах падает
    на несовпадении размерностей внутри модуля.

    Установка отдельная и неприятная: `uni2ts` тянет старые numpy и scipy,
    которые на Python 3.14 собираются из исходников и падают на cython. Ставится
    через `--no-deps` плюс gluonts, einops, jaxtyping, hydra-core, lightning и jax
    по отдельности; jax нужен не сам по себе, а из-за `PyTree` в jaxtyping.
    """

    name = "moirai"
    REPO = "Salesforce/moirai-1.1-R-small"
    _MODULE = None

    PATCH_SIZES = (8, 16, 32)

    def __init__(self, num_samples: int = 100, batch_size: int = 256, probe: int = 300) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.probe = probe  # на скольких рядах сравниваются размеры патча
        self.patch_size = self.PATCH_SIZES[0]
        self.notes: list[str] = []

    @classmethod
    def _module(cls):
        if cls._MODULE is None:
            from uni2ts.model.moirai import MoiraiModule

            cls._MODULE = MoiraiModule.from_pretrained(cls.REPO)
        return cls._MODULE

    def fit(self, wide, train_end: int, horizon: int) -> "Moirai":
        self._module()
        self.notes = []

        inner_end = train_end - horizon
        values = wide.to_numpy(dtype=float).T
        probe = values[: self.probe]
        scores = {}
        if inner_end > horizon:
            actual = probe[:, inner_end : inner_end + horizon]
            for patch in self.PATCH_SIZES:
                try:
                    predicted = self._forecast(probe, inner_end, horizon, patch)
                except Exception as exc:  # неподходящий патч просто выбывает
                    self.notes.append(f"патч {patch}: не сработал ({str(exc)[:60]})")
                    continue
                scores[patch] = float(np.nanmean(np.abs(predicted - actual)))
        if scores:
            self.patch_size = min(scores, key=scores.get)
        self.notes.append(
            f"moirai {self.REPO}, контекст {train_end}, без дообучения | "
            f"патч {self.patch_size} из "
            + ", ".join(f"{k}: {v:.0f}" for k, v in sorted(scores.items()))
            + " (по отложенным месяцам обучения)"
        )
        return self

    def _forecast(self, values: np.ndarray, context_length: int, horizon: int, patch: int):
        """Прогноз для матрицы рядов при заданном размере патча."""
        import torch
        from uni2ts.model.moirai import MoiraiForecast

        forecaster = MoiraiForecast(
            module=self._module(),
            prediction_length=horizon,
            context_length=context_length,
            patch_size=patch,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        window = values[:, :context_length]
        out = np.full((window.shape[0], horizon), np.nan)
        for start in range(0, len(window), self.batch_size):
            chunk = window[start : start + self.batch_size]
            context = torch.tensor(chunk[:, :, None], dtype=torch.float32)
            with torch.no_grad():
                samples = forecaster(
                    past_target=context,
                    past_observed_target=torch.ones(context.shape, dtype=torch.bool),
                    past_is_pad=torch.zeros(context.shape[:2], dtype=torch.bool),
                )
            # (ряды, выборки, горизонт) -> медиана по выборкам
            out[start : start + len(chunk)] = np.median(np.asarray(samples), axis=1)[:, :horizon]
        return out

    def predict(self, wide, train_end: int, horizon: int) -> np.ndarray:
        values = wide.to_numpy(dtype=float).T
        return self._forecast(values, train_end, horizon, self.patch_size)
