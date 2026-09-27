"""Прогноз вперёд от конца панели: scripts/forecast_forward.py.

Синтетика — как в tests/test_horizons.py, настоящие данные не читаются. Агрегат
до origin короче 37 месяцев: бэктест внутри `choose_model`
(`for origin in range(max(36 + h, n - 24), n - h + 1)`) не набирает ни одной точки
ни на одном горизонте протокола (1, 3, 6, 12) — при n=30 нижняя граница диапазона
(не меньше 36 + h) уже больше верхней (n − h + 1) для любого из них, и кандидат
побеждает random_walk без единого вызова ETS/SARIMA. Дорогая часть, что остаётся, —
три подгонки global_gbm_cat (h=1, 3, 6); весь файл укладывается в ту же дюжину секунд.
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib.util
import io
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.global_model import PanelContext  # noqa: E402

# scripts/ — не пакет: скрипт грузится по пути, без правки sys.path под все тесты.
_spec = importlib.util.spec_from_file_location(
    "forecast_forward", ROOT / "scripts" / "forecast_forward.py"
)
forecast_forward_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(forecast_forward_module)

N_SERIES = 20  # меньше рядов — короче подгонка global_gbm_cat
N_PRE_ORIGIN = 30    # < 37: бэктест choose_model пуст на всех горизонтах протокола
N_POST_ORIGIN = 12   # хватает на самый длинный горизонт (12 месяцев 2025 года)

SERIES = [f"мо_{i}" for i in range(N_SERIES)]
DROPPED_REGION_SERIES = SERIES[-1]  # «омоним» без региона — проверяем, что merge не роняет строку

CFG = {
    "origin": "2024-12",
    "horizons": [1, 3, 6, 12],
    "recommended": {1: "global_gbm_cat", 3: "global_gbm_cat", 6: "global_gbm_cat", 12: "two_stage"},
    "comparison": "two_stage",
    "known_aggregate": "two_stage_known",
}
ORIGIN = pd.Period(CFG["origin"], freq="M")


def make_wide(series: list[str] = SERIES, seed: int = 1) -> pd.DataFrame:
    """24 месяца (2023-01..2024-12), тренд + сезонность + шум; индекс — начало месяца,
    как у настоящей панели после build_matrix."""
    rng = np.random.default_rng(seed)
    t = np.arange(24, dtype=float)
    index = pd.date_range("2023-01-01", periods=24, freq="MS")
    data = {
        s: 1_000 + 100 * i + 5 * (i + 1) * t + 50 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 24)
        for i, s in enumerate(series)
    }
    return pd.DataFrame(data, index=index)


def make_categories(wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Две категории — постоянные доли итога плюс небольшой шум: то, что нужно
    share-признакам global_gbm_cat."""
    rng = np.random.default_rng(11)
    shares = {"продовольствие": 0.30, "маркетплейсы": 0.15}
    return {name: wide * share * (1 + rng.normal(0, 0.01, wide.shape)) for name, share in shares.items()}


def make_aggregate(n_pre: int = N_PRE_ORIGIN, n_post: int = N_POST_ORIGIN, seed: int = 2) -> pd.Series:
    """Федеральный агрегат с сезонностью, заходящий за origin; до origin короче 37 месяцев
    (см. докстринг модуля — почему это обходит дорогой бэктест)."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    t = np.arange(n, dtype=float)
    pre = pd.period_range(end=CFG["origin"], periods=n_pre, freq="M")
    post = pd.period_range(start=pre[-1] + 1, periods=n_post, freq="M")
    values = 1_000_000 + 3_000 * t + 40_000 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 500, n)
    return pd.Series(values, index=pre.append(post))


def corrupt_after_origin(aggregate: pd.Series, origin: pd.Period, horizon_reach: int) -> pd.Series:
    """Мусор после origin: ×1000 на месяцах, которые реально читает какой-то горизонт
    протокола (1..horizon_reach), и NaN дальше — за пределами того, что кто-либо читает.

    NaN внутри рабочего окна `two_stage_known` уронил бы её же исключением
    (`_forecast_aggregate` требует все месяцы горизонта конечными) — тогда «прогноз
    меняется» было бы неотличимо от «прогон падает». ×1000 в рабочем окне и так меняет
    прогноз и факт, а NaN относится дальше, где до него никто не дотягивается.
    """
    out = aggregate.copy()
    within_reach = (out.index > origin) & (out.index <= origin + horizon_reach)
    beyond_reach = out.index > origin + horizon_reach
    out.loc[within_reach] *= 1000.0
    out.loc[beyond_reach] = np.nan
    return out


def make_long_series_with_future_garbage(
    origin: pd.Period, n_pre: int = 40, n_post: int = 6,
) -> pd.Series:
    """Длинный ряд, заходящий за origin, с мусором на этих месяцах.

    Для проверки обрезки контекста (см. `ContextTruncationTest`): если бы скрипт
    не обрезал `long_series` по origin перед тем, как отдать его прогнозным
    моделям, этот мусор («не тот» порядок величины) дошёл бы до них, и его
    легко отличить от настоящих значений.
    """
    rng = np.random.default_rng(7)
    n = n_pre + n_post
    pre = pd.period_range(end=origin, periods=n_pre, freq="M")
    post = pd.period_range(start=origin + 1, periods=n_post, freq="M")
    values = 500_000 + 1_000 * np.arange(n) + rng.normal(0, 100, n)
    series = pd.Series(values, index=pre.append(post))
    series.loc[series.index > origin] *= 1000.0  # мусор — обрезанный контекст не должен его увидеть
    return series


def make_context(aggregate: pd.Series, wide: pd.DataFrame) -> PanelContext:
    index = pd.to_datetime(wide.index).to_period("M")
    return PanelContext(
        # Не None: иначе проверка «прогнозным моделям внешние признаки не передаются»
        # проходила бы и без external=None в скрипте.
        index=index, categories=make_categories(wide), external=object(),
        aggregate=aggregate, long_series={}, regions={},
    )


def make_regions(columns: pd.Index) -> pd.DataFrame:
    """series_id/region/oktmo для колонок вывода. DROPPED_REGION_SERIES в справочнике
    нет намеренно — проверяем, что merge оставляет его пустым, а не роняет строку."""
    rows = [
        {"series_id": c, "region": f"регион_{i % 3}", "oktmo": f"00-{i:03d}-000"}
        for i, c in enumerate(columns) if c != DROPPED_REGION_SERIES
    ]
    return pd.DataFrame(rows, columns=["series_id", "region", "oktmo"])


def run(aggregate: pd.Series, wide: pd.DataFrame, cfg: dict = CFG, context: PanelContext | None = None):
    """forecast_forward на синтетике, вывод и предупреждения подавлены.

    `context` — готовый контекст вместо построенного из `aggregate`/`wide`: нужен
    тестам, которым требуется положить что-то своё в контекст (например, длинный
    ряд с мусором после origin), не переопределяя `make_context`.
    """
    if context is None:
        context = make_context(aggregate, wide)
    regions = make_regions(wide.columns)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return forecast_forward_module.forecast_forward(wide, context, regions, cfg)


@functools.lru_cache(maxsize=1)
def _clean_run():
    """Единственный дорогой прогон на чистых данных — переиспользуется всеми тестами,
    которым не нужен собственный мусор в агрегате. `lru_cache` считает его один раз
    на весь файл, независимо от того, какой класс тестов запросил его первым."""
    wide = make_wide()
    aggregate = make_aggregate()
    forecast, check, notes = run(aggregate, wide)
    return wide, aggregate, forecast, check, notes


def sort_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["model", "series_id", "horizon", "month"]).reset_index(drop=True)


def sort_check(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["method", "horizon", "month"]).reset_index(drop=True)


class ShapeAndKeysTest(unittest.TestCase):
    """Форма и ключи выхода — на едином прогоне чистых данных."""

    @classmethod
    def setUpClass(cls):
        cls.wide, cls.aggregate, cls.forecast, cls.check, cls.notes = _clean_run()

    def test_forecast_columns_and_order(self):
        self.assertEqual(
            list(self.forecast.columns),
            ["series_id", "region", "oktmo", "model", "horizon", "month",
             "forecast", "recommended", "uses_published_aggregate"],
        )

    def test_check_columns_and_order(self):
        self.assertEqual(
            list(self.check.columns),
            ["horizon", "method", "aggregate_model", "month", "forecast", "actual", "error_pct"],
        )

    def test_row_count_is_series_times_horizon_per_pair(self):
        counts = self.forecast.groupby(["model", "horizon"]).size().to_dict()
        expected = {}
        for h in CFG["horizons"]:
            for name in dict.fromkeys([CFG["recommended"][h], CFG["comparison"]]):
                expected[(name, h)] = N_SERIES * h
            expected[(CFG["known_aggregate"], h)] = N_SERIES * h
        self.assertEqual(counts, expected)

    def test_months_are_first_h_months_after_origin(self):
        for (model, horizon), rows in self.forecast.groupby(["model", "horizon"]):
            expected_months = [str(ORIGIN + k) for k in range(1, horizon + 1)]
            with self.subTest(model=model, horizon=horizon):
                self.assertEqual(sorted(rows["month"].unique()), expected_months)

    def test_two_stage_known_ratio_to_month_aggregate_is_constant_per_series_and_horizon(self):
        """`forecast / aggregate[месяц строки]` обязано быть одним числом на пару
        (ряд, горизонт) — это и есть доля ряда, на которую `two_stage_known`
        домножает ОПУБЛИКОВАННЫЙ агрегат месяца из колонки `month`.

        В отличие от `test_months_are_first_h_months_after_origin` (который сверяет
        подписи месяцев с тем, что сама же `_long_rows` в них и кладёт — тавтология),
        здесь подпись месяца сверяется с фактическим значением агрегата на этот месяц
        и с тем, что реально попало в `forecast`. Если бы индекс дополнения
        (`_future_index`) был сдвинут на месяц, `two_stage_known` умножала бы долю
        не на тот агрегат, а подпись всё равно осталась бы «правильной» (её ставит
        `_long_rows` независимо от того, что почитала модель) — отношение
        перестало бы быть постоянным по шагам горизонта, потому что синтетический
        агрегат — тренд и сезонность с шумом, а не чистая экспонента, и отношение
        соседних месяцев у него само не постоянно.
        """
        known = self.forecast.loc[self.forecast["model"] == "two_stage_known"].copy()
        months = pd.PeriodIndex(known["month"], freq="M")
        known["aggregate_at_month"] = self.aggregate.reindex(months).to_numpy()
        known["ratio"] = known["forecast"] / known["aggregate_at_month"]
        for (series, horizon), group in known.groupby(["series_id", "horizon"]):
            with self.subTest(series=series, horizon=horizon):
                ratios = group["ratio"].to_numpy()
                self.assertTrue(
                    np.allclose(ratios, ratios[0], rtol=1e-9, atol=0),
                    f"отношение forecast/aggregate[month] непостоянно по горизонту: {ratios}",
                )

    def test_key_is_unique(self):
        key = list(zip(
            self.forecast["series_id"], self.forecast["model"],
            self.forecast["horizon"], self.forecast["month"],
        ))
        self.assertEqual(len(key), len(set(key)))

    def test_exactly_one_recommended_model_per_horizon(self):
        for horizon, rows in self.forecast.groupby("horizon"):
            with self.subTest(horizon=horizon):
                recommended_models = sorted(rows.loc[rows["recommended"], "model"].unique())
                self.assertEqual(recommended_models, [CFG["recommended"][horizon]])
                self.assertEqual(int(rows["recommended"].sum()), N_SERIES * horizon)

    def test_uses_published_aggregate_only_for_known_model(self):
        flagged = set(self.forecast.loc[self.forecast["uses_published_aggregate"], "model"])
        not_flagged = set(self.forecast.loc[~self.forecast["uses_published_aggregate"], "model"])
        self.assertEqual(flagged, {"two_stage_known"})
        self.assertEqual(not_flagged, {"global_gbm_cat", "two_stage"})

    def test_region_and_oktmo_attach_and_homonym_stays_empty(self):
        matched = self.forecast.loc[self.forecast["series_id"] == SERIES[0]]
        self.assertTrue((matched["region"] == "регион_0").all())
        self.assertTrue((matched["oktmo"] == "00-000-000").all())
        dropped = self.forecast.loc[self.forecast["series_id"] == DROPPED_REGION_SERIES]
        self.assertFalse(dropped.empty)
        self.assertTrue(dropped["region"].isna().all())
        self.assertTrue(dropped["oktmo"].isna().all())

    def test_notes_are_about_aggregate_model_and_share_rule(self):
        self.assertTrue(self.notes)
        two_stage_family = {CFG["comparison"], CFG["known_aggregate"]}
        for entry in self.notes:
            with self.subTest(entry=entry):
                self.assertIn(entry["model"], two_stage_family)
                self.assertIn("модель", entry["note"])
                self.assertIn("доля по правилу", entry["note"])


class AggregateCheckFormulaTest(unittest.TestCase):
    """Формулы ориентиров проверки — на том же прогоне: наивная, сезонно-наивная, ошибка,%."""

    @classmethod
    def setUpClass(cls):
        cls.wide, cls.aggregate, cls.forecast, cls.check, cls.notes = _clean_run()

    def test_naive_is_december_level_repeated(self):
        december = float(self.aggregate.loc[ORIGIN])
        naive = self.check.loc[self.check["method"] == "naive"]
        self.assertTrue(np.array_equal(naive["forecast"].to_numpy(), np.full(len(naive), december)))
        self.assertTrue((naive["aggregate_model"] == "как в декабре").all())

    def test_seasonal_naive_is_same_calendar_month_last_year(self):
        seasonal = self.check.loc[self.check["method"] == "seasonal_naive"]
        expected = [float(self.aggregate.loc[pd.Period(m, freq="M") - 12]) for m in seasonal["month"]]
        self.assertTrue(np.array_equal(seasonal["forecast"].to_numpy(), np.asarray(expected)))
        self.assertTrue((seasonal["aggregate_model"] == "как год назад").all())

    def test_error_pct_formula(self):
        expected = (self.check["forecast"] - self.check["actual"]) / self.check["actual"] * 100.0
        self.assertTrue(np.allclose(self.check["error_pct"].to_numpy(), expected.to_numpy()))

    def test_actual_is_the_uncut_aggregate(self):
        for month, actual in zip(self.check["month"], self.check["actual"]):
            self.assertEqual(actual, float(self.aggregate.loc[pd.Period(month, freq="M")]))

    def test_two_stage_forecast_matches_the_fitted_model(self):
        # Проверка обязана описывать именно тот первый этап, который дал прогноз, а не
        # похожий. forecast_2025.csv хранит ряд × доля, aggregate_check — сам агрегат:
        # для одного ряда их отношение месяц к месяцу обязано совпасть, доля — общий
        # множитель. Основная гарантия — внутри самого forecast_forward (np.allclose
        # против model.forecast, иначе ValueError и весь прогон, включая _clean_run, падает).
        one_series = SERIES[0]
        rows = self.forecast.loc[(self.forecast["model"] == "two_stage") & (self.forecast["series_id"] == one_series)]
        for horizon, part in rows.groupby("horizon"):
            months = sorted(part["month"].unique())
            from_forecast = part.set_index("month").loc[months, "forecast"].to_numpy()
            check_rows = self.check[(self.check["horizon"] == horizon) & (self.check["method"] == "two_stage")]
            from_check = check_rows.set_index("month").loc[months, "forecast"].to_numpy()
            share = from_forecast[0] / from_check[0]
            with self.subTest(horizon=horizon):
                self.assertTrue(np.allclose(from_forecast, share * from_check))


class OriginAndAggregateValidationTest(unittest.TestCase):
    """Проверки, которые обязаны падать раньше любой подгонки модели — дешёвые, без GBM."""

    def test_origin_not_last_month_of_matrix_raises(self):
        wide = make_wide()
        context = make_context(make_aggregate(), wide)
        regions = make_regions(wide.columns)
        bad_cfg = {**CFG, "origin": "2024-11"}  # последний месяц матрицы — 2024-12
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                forecast_forward_module.forecast_forward(wide, context, regions, bad_cfg)

    def test_missing_aggregate_raises(self):
        wide = make_wide()
        context = dataclasses.replace(make_context(make_aggregate(), wide), aggregate=None)
        regions = make_regions(wide.columns)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                forecast_forward_module.forecast_forward(wide, context, regions, CFG)


class NoLeakageTest(unittest.TestCase):
    """Мусор в агрегате после origin не должен менять то, что видит только прошлое."""

    @classmethod
    def setUpClass(cls):
        cls.wide, base_aggregate, cls.clean_forecast, cls.clean_check, _ = _clean_run()
        garbage_aggregate = corrupt_after_origin(base_aggregate, ORIGIN, max(CFG["horizons"]))
        cls.garbage_forecast, cls.garbage_check, _ = run(garbage_aggregate, cls.wide)

    def test_global_gbm_cat_and_two_stage_forecasts_are_bit_identical_on_all_horizons(self):
        for model in ("global_gbm_cat", "two_stage"):
            with self.subTest(model=model):
                clean = sort_forecast(self.clean_forecast.loc[self.clean_forecast["model"] == model])
                garbage = sort_forecast(self.garbage_forecast.loc[self.garbage_forecast["model"] == model])
                self.assertGreater(len(clean), 0)  # иначе сравнение пустых массивов ничего не проверяет
                self.assertEqual(len(clean), len(garbage))
                self.assertTrue(np.array_equal(clean["forecast"].to_numpy(), garbage["forecast"].to_numpy()))

    def test_aggregate_check_forecast_column_is_bit_identical(self):
        clean = sort_check(self.clean_check)
        garbage = sort_check(self.garbage_check)
        self.assertTrue(np.array_equal(clean["forecast"].to_numpy(), garbage["forecast"].to_numpy()))

    def test_two_stage_known_forecast_changes(self):
        clean = sort_forecast(self.clean_forecast.loc[self.clean_forecast["model"] == "two_stage_known"])
        garbage = sort_forecast(self.garbage_forecast.loc[self.garbage_forecast["model"] == "two_stage_known"])
        self.assertFalse(np.array_equal(clean["forecast"].to_numpy(), garbage["forecast"].to_numpy()))
        self.assertFalse(np.isnan(garbage["forecast"].to_numpy()).any())  # мусор в легальном окне — не NaN

    def test_aggregate_check_actual_column_changes(self):
        clean = sort_check(self.clean_check)
        garbage = sort_check(self.garbage_check)
        self.assertFalse(clean["actual"].reset_index(drop=True).equals(garbage["actual"].reset_index(drop=True)))


class _ContextSpy:
    """Оборачивает настоящую модель: `fit`/`predict` идут в неё без изменений,
    а `set_context` попутно складывает полученный контекст в `sink`.

    Не подменяет логику скрипта ради теста — прогноз считает настоящая модель,
    спай только подсматривает, что ей передали, чтобы проверить, что скрипт
    обрезает контекст по origin для прогнозных моделей и не обрезает для
    `two_stage_known`, без пересборки этой проверки из значений самого прогноза.
    """

    def __init__(self, inner, sink: list) -> None:
        self._inner = inner
        self._sink = sink

    def set_context(self, context) -> None:
        self._sink.append(context)
        self._inner.set_context(context)

    def fit(self, wide, train_end, horizon):
        self._inner.fit(wide, train_end, horizon)
        return self

    def predict(self, wide, train_end, horizon):
        return self._inner.predict(wide, train_end, horizon)

    @property
    def notes(self):
        return self._inner.notes


def _spy_factory(real_factory, sink: list):
    return lambda: _ContextSpy(real_factory(), sink)


class ContextTruncationTest(unittest.TestCase):
    """Контекст, который получают прогнозные модели, обрезан по origin;
    `two_stage_known` — нет. Тестовый контекст раньше нёс `long_series={}`
    и `external=None` без единого длинного ряда — обрезать было попросту нечего,
    и этот путь скрипта тестами не проверялся."""

    def test_forecast_models_get_context_cut_at_origin_known_model_gets_full_one(self):
        wide = make_wide()
        aggregate = make_aggregate()
        long_series = make_long_series_with_future_garbage(ORIGIN)
        context = dataclasses.replace(
            make_context(aggregate, wide), long_series={"industry_x": long_series},
        )
        cfg = {**CFG, "horizons": [1]}  # одного горизонта достаточно: обрезка — не свойство горизонта

        cut_contexts: list[PanelContext] = []
        full_contexts: list[PanelContext] = []
        with mock.patch.dict(
            forecast_forward_module.GLOBAL_MODELS,
            {"global_gbm_cat": _spy_factory(
                forecast_forward_module.GLOBAL_MODELS["global_gbm_cat"], cut_contexts,
            )},
        ), mock.patch.dict(
            forecast_forward_module.EXTRA_GLOBAL,
            {"two_stage_known": _spy_factory(
                forecast_forward_module.EXTRA_GLOBAL["two_stage_known"], full_contexts,
            )},
        ):
            run(aggregate, wide, cfg=cfg, context=context)

        self.assertEqual(len(cut_contexts), 1)
        self.assertEqual(len(full_contexts), 1)

        cut = cut_contexts[0]
        pd.testing.assert_series_equal(cut.long_series["industry_x"], long_series.loc[:ORIGIN])
        pd.testing.assert_series_equal(cut.aggregate, aggregate.loc[:ORIGIN])
        self.assertIsNone(cut.external)

        known = full_contexts[0]
        pd.testing.assert_series_equal(known.aggregate, aggregate)
        pd.testing.assert_series_equal(known.long_series["industry_x"], long_series)


class RealConfigConsistencyTest(unittest.TestCase):
    """`configs/forecast_forward.yaml` согласован с тем, что скрипт от него ожидает.

    `CFG` во всех тестах выше вписан руками и с настоящим конфигом никак не
    сверяется — расхождение (модель без регистрации, горизонт без правила,
    неразбираемый origin) иначе осталось бы незамеченным до самого прогона.
    """

    def test_real_config_matches_registries_and_horizons(self):
        cfg = yaml.safe_load((ROOT / "configs" / "forecast_forward.yaml").read_text(encoding="utf-8"))
        prognostic = forecast_forward_module.GLOBAL_MODELS
        known_available = forecast_forward_module.GLOBAL_MODELS | forecast_forward_module.EXTRA_GLOBAL

        for horizon, name in cfg["recommended"].items():
            with self.subTest(role="recommended", horizon=horizon, model=name):
                self.assertIn(name, prognostic)
        with self.subTest(role="comparison", model=cfg["comparison"]):
            self.assertIn(cfg["comparison"], prognostic)
        with self.subTest(role="known_aggregate", model=cfg["known_aggregate"]):
            self.assertIn(cfg["known_aggregate"], known_available)

        with self.subTest(check="recommended_keys_match_horizons"):
            self.assertEqual(sorted(cfg["recommended"]), sorted(cfg["horizons"]))

        with self.subTest(check="origin_parses_as_month"):
            origin = pd.Period(cfg["origin"], freq="M")  # бросит, если не месяц — и есть проверка
            self.assertEqual(str(origin), cfg["origin"])

        with self.subTest(check="output_has_both_filenames"):
            self.assertIn("forecast", cfg["output"])
            self.assertIn("aggregate_check", cfg["output"])


if __name__ == "__main__":
    unittest.main()
