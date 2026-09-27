"""Глобальная модель: одна обученная модель на всю панель вместо 2190 отдельных.

Классические модели упёрлись в потолок около 1 700–2 000 MAE, и это не недостаток
конкретных алгоритмов: пятнадцать точек истории просто не содержат больше информации.
Единственный способ добавить информации — взять её у соседних рядов. Две тысячи
муниципалитетов переживают одни и те же общероссийские шоки и один и тот же
сезонный ритм, так что форма динамики у них общая, а различается масштаб.

Отсюда две конструктивные идеи:

1. **Целевая переменная — отношение, а не уровень.** Модель предсказывает
   y[t+h] / y[t], то есть во сколько раз изменятся расходы. Это ставит Магадан
   с 74 тысячами и сельский район с десятью в одинаковые условия: учимся на форме,
   а не на масштабе. Заодно снимается вопрос о завышенности абсолютного уровня
   показателя — в отношении постоянное смещение сокращается.

2. **Прямой прогноз на каждый горизонт.** Для h = 1, 2, 3 обучаются отдельные модели,
   а не одна с рекурсивной подстановкой собственных прогнозов. На горизонте в три шага
   прямой подход и точнее, и не накапливает ошибку.

## Источники признаков сверх собственной истории ряда

Шесть, и каждый включается отдельно, чтобы вклад каждого был виден в таблице.

| источник | что добавляет | различает ли муниципалитеты |
|---|---|---|
| `external` | темпы роста длинных федеральных рядов | нет, в месяце t одинаков для всех |
| `common_factor` | нормировка цели на общее движение панели | нет, но снимает то, чего модель знать не может |
| `categories` | доли пяти категорий трат и их динамика | **да** |
| `stack_categories` | шесть категорий как отдельные обучающие ряды | **да** |
| `pretrain` | предобучение на длинных отраслевых рядах | нет |
| `news_path` | доли новостных тем региона (`data/news/monthly.parquet`) | по региону |

Первые два и четвёртый переносят **историю**: длинные ряды знают апрельскую
сезонность, которой в пятнадцати месяцах панели нет ни одного раза. Третий
переносит **сечение**: структура трат отличает районы друг от друга, и этого
не умеет ни один федеральный ряд.

Шестой, `news_path`, — ни то ни другое: не история и не структура трат, а внешний
сигнал, различающий не районы, а их регионы — муниципалитеты одного региона получают
одну и ту же новостную долю тем месяца.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Лаг 12 намеренно отсутствует. Он требует одиннадцати точек истории, и при обучении
# в 15 точек на горизонт 3 не остаётся ни одного обучающего примера. Годовой цикл на
# этих рядах всё равно не идентифицируется — это уже показано на Prophet.
LAGS = (1, 2, 3, 4, 5, 6)

# Сдвиг, на котором считается изменение доли категории. Шесть месяцев, а не один:
# помесячная доля шумная, а нас интересует, куда структура трат едет, а не дрожит.
SHARE_SHIFT = 6


@dataclass
class PanelContext:
    """Всё, что глобальная модель знает сверх матрицы самих рядов.

    Собирается один раз в ``run.py`` и передаётся модели через ``set_context``:
    иначе каждая модель на каждом фолде заново читала бы одни и те же файлы.
    """

    index: pd.PeriodIndex
    categories: dict[str, pd.DataFrame] = field(default_factory=dict)
    external: object | None = None
    aggregate: pd.Series | None = None
    long_series: dict[str, pd.Series] = field(default_factory=dict)
    regions: dict[str, str] = field(default_factory=dict)


def _news_row(news: dict | None, region: str | None, month_index: int) -> dict:
    """Новостные признаки региона за месяц. Отсутствие новостей — не ноль, а NaN.

    Ноль означал бы «ничего не писали», а у нас нет издания по этому региону вовсе.
    Гистограммный бустинг sklearn работает с пропусками штатно и учится обходиться
    без признака там, где его нет, — подменять их нулями значило бы врать модели.
    """
    if not news or region is None:
        return {}
    return news.get((region, month_index), {})


# Шесть долей тем в data/news/monthly.parquet — темы ДКП там нет, она есть только
# в национальном файле (src/news.py). `intensity` не входит: нормирована средним
# и разбросом числа публикаций издания за весь период, то есть заглядывает вперёд
# (то же исключение, что у `TwoStageNews._candidates`). `n_articles`/`n_outlets` —
# объём, а не тема, и корпус растёт с 19,7 до 30 тыс. публикаций в месяц: такой
# признак работал бы меткой времени, а не темой.
NEWS_TOPICS = ("t_ceny", "t_dohody", "t_zanjatost", "t_proizvodstvo", "t_kredit", "t_torgovlja")


def _load_news(
    path: Path, regions: dict[str, str], index: pd.PeriodIndex
) -> dict[tuple[str, int], dict[str, float]]:
    """Новостные признаки по (регион, позиция месяца t в панели). Вызывается один раз
    из ``set_context`` — здесь только сопоставление месяца файла с его позицией в этой
    панели: причинность самой доли темы месяца t (публикации только этого месяца)
    обеспечивает ``scripts/build_news_features.py``, не эта функция.

    Регион берётся из ``PanelContext.regions``, а не из колонки файла напрямую: ряд,
    для которого регион не определён (омоним), новостных признаков не получит —
    ключа с его именем в словаре просто не будет.
    """
    frame = pd.read_parquet(path)
    months = pd.PeriodIndex(pd.to_datetime(frame["month"]), freq="M")
    position = {period: t for t, period in enumerate(index)}
    covered = set(regions.values())
    out: dict[tuple[str, int], dict[str, float]] = {}
    for row, period in zip(frame.itertuples(index=False), months):
        if row.region_name not in covered:
            continue
        t = position.get(period)
        if t is None:
            continue
        out[(row.region_name, t)] = {f"news_{topic}": getattr(row, topic) for topic in NEWS_TOPICS}
    return out


def _features(series: np.ndarray, t: int) -> dict | None:
    """Признаки, доступные в момент t. Всё считается только по прошлому."""
    if t < max(LAGS) - 1:
        return None
    last = series[t]
    if last <= 0:
        return None

    row = {f"ratio_lag_{lag}": series[t - lag + 1] / last for lag in LAGS if lag > 1}
    row["level_log"] = float(np.log(last))
    for window in (3, 6):
        if t + 1 >= window:
            window_slice = series[t - window + 1 : t + 1]
            row[f"mean_ratio_{window}"] = float(window_slice.mean() / last)
            row[f"std_ratio_{window}"] = float(window_slice.std() / last)
    row["month"] = (t % 12) + 1
    return row


class _CategoryShares:
    """Доли пяти категорий трат в общих расходах муниципалитета и их динамика.

    Доли, а не уровни, и это не стилистика. «Все категории» включает остальные пять
    по определению, поэтому уровень категории почти линейно связан с уровнем итога,
    и в отношениях он выродился бы в копию уже имеющегося признака. Доля же говорит
    о структуре трат: где половина расходов идёт на продовольствие, динамика другая,
    чем там, где столько же приходится на маркетплейсы. Пять категорий покрывают
    около 72% итога, остаток на долю прочего — он и различает районы дальше.
    """

    def __init__(self, categories: dict[str, pd.DataFrame], columns: pd.Index) -> None:
        self.names = sorted(categories)
        self.shares: dict[str, np.ndarray] = {}
        for name in self.names:
            frame = categories[name].reindex(columns=columns)
            self.shares[name] = frame.to_numpy(dtype=float).T  # ряды по строкам

    def row(self, series_pos: int, series: np.ndarray, t: int) -> dict[str, float]:
        total = series[t]
        if total <= 0:
            return {}
        out: dict[str, float] = {}
        for name in self.names:
            values = self.shares[name][series_pos]
            share = values[t] / total
            if not np.isfinite(share):
                continue
            out[f"share_{name}"] = share
            past_total = series[t - SHARE_SHIFT] if t >= SHARE_SHIFT else np.nan
            if t >= SHARE_SHIFT and past_total > 0:
                past = values[t - SHARE_SHIFT] / past_total
                if np.isfinite(past) and past > 0:
                    out[f"dshare_{name}"] = share / past - 1.0
        return out


class GlobalGBM:
    """Гистограммный градиентный бустинг на признаках отношений, обученный на всей панели.

    Взят sklearn, а не LightGBM, сознательно: LightGBM тянет системную библиотеку
    OpenMP, которой на чистой macOS нет, и запуск у проверяющего упирается в
    `brew install libomp`. Для воспроизводимости зависимость, требующая системного
    пакета, того не стоит. Алгоритм тот же —
    гистограммный бустинг на деревьях.
    """

    name = "global_gbm"

    def __init__(
        self, max_iter: int = 300, learning_rate: float = 0.05, max_leaf_nodes: int = 31,
        news_path: str | None = None, regions: dict | None = None,
        external: bool = False, common_factor: bool = False,
        categories: bool = False, pretrain: bool = False,
        regional_factor: bool = False, stack_categories: bool = False,
    ) -> None:
        # Путь до data/news/monthly.parquet — абсолютный (src/run.py собирает его от ROOT)
        # или относительно рабочего каталога, как у TwoStageNews.news_path. По умолчанию
        # выключен: остальные варианты GlobalGBM этот файл вообще не открывают. Сам словарь
        # `news` строится лениво в `set_context` (там есть и регионы, и ось времени панели) —
        # не здесь и не при каждом обращении к ряду.
        self.news_path = news_path
        self.news: dict | None = None
        self.regions = regions
        self.use_external = external
        self.use_factor = common_factor
        self.use_categories = categories
        self.use_pretrain = pretrain
        self.use_regional = regional_factor
        self.use_stack = stack_categories
        self.params = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=20260920,
        )
        self.context: PanelContext | None = None
        self.notes: list[str] = []
        self._models: dict[int, object] = {}
        # Колонки запоминаются на каждый шаг отдельно. На длинном обучении наборы
        # совпадают, а на коротком нет: при двенадцати месяцах и шаге 6 обучающая
        # пара всего одна, и в ней ещё нет сдвига долей категорий, — модель шага 6
        # видит 16 признаков, а модель шага 1 — 21. Один общий список колонок
        # здесь молча подсовывал бы одной из них чужую матрицу.
        self._columns: dict[int, list[str]] = {}
        self._factor = None
        self._shares: _CategoryShares | None = None

    def set_context(self, context: PanelContext) -> None:
        self.context = context
        if self.news_path is not None and self.news is None:
            self.news = _load_news(Path(self.news_path), context.regions, context.index)
            self.regions = context.regions

    # -- служебное ---------------------------------------------------------

    def _require_context(self) -> PanelContext:
        if self.context is None:
            raise RuntimeError(
                "модели нужен PanelContext: внешние признаки, категории и общий "
                "фактор берутся из него, а без него они молча отключились бы"
            )
        return self.context

    def _extra_row(self, series_pos: int, series: np.ndarray, col: str, t: int) -> dict:
        """Признаки сверх собственной истории ряда."""
        out = _news_row(self.news, (self.regions or {}).get(col), t)
        if self.use_external:
            context = self._require_context()
            if context.external is not None:
                out |= context.external.row(context.index, t)
        if self._shares is not None:
            out |= self._shares.row(series_pos, series, t)
        return out

    def _multiplier(self, t: int, step: int, col: str) -> float:
        """Общий фактор, которым нормируется цель. Единица, если фактор не используется."""
        if self._factor is None:
            return 1.0
        return self._factor.realised(t, step, col)

    # -- обучение и прогноз ------------------------------------------------

    def fit(self, wide: pd.DataFrame, train_end: int, horizon: int) -> "GlobalGBM":
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.notes = []
        if self.use_categories:
            context = self._require_context()
            self._shares = _CategoryShares(context.categories, wide.columns)
        if self.use_factor or self.use_regional:
            from src.external import CommonFactor

            context = self._require_context()
            self._factor = CommonFactor(
                aggregate=context.aggregate,
                regions=context.regions if self.use_regional else None,
            ).fit(wide, train_end, horizon)
            self.notes.append(self._factor.fit_report.as_text())

        pretrain_rows = (
            self._pretrain_set(wide, train_end, horizon) if self.use_pretrain else {}
        )

        for step in range(1, horizon + 1):
            features, target = self._training_set(wide, train_end, step)
            if features.empty:
                raise ValueError(f"нет обучающих примеров для шага {step}")

            model = HistGradientBoostingRegressor(**self.params)
            if step in pretrain_rows:
                # Предобучение: сначала длинные отраслевые ряды, потом панель.
                # warm_start продолжает тот же ансамбль на новых данных, то есть
                # деревья, выученные на длинной истории, остаются, а следующие
                # правят их под муниципальные ряды. Это и есть дообучение.
                long_features, long_target = pretrain_rows[step]
                long_features = long_features.reindex(columns=features.columns)
                model.set_params(warm_start=True, max_iter=self.params["max_iter"] // 2)
                model.fit(long_features.to_numpy(dtype=float), long_target)
                model.set_params(max_iter=self.params["max_iter"])
                self.notes.append(
                    f"шаг {step}: предобучение на {len(long_target)} примерах длинных рядов"
                )
            self._columns[step] = list(features.columns)
            model.fit(features.to_numpy(dtype=float), target)
            self._models[step] = model
        return self

    def _training_set(
        self, wide: pd.DataFrame, train_end: int, horizon_step: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Все пары (ряд, момент) из обучающей части, где известен и признак, и ответ."""
        rows, targets = [], []
        values = wide.to_numpy(dtype=float).T  # ряды по строкам
        for position, (col, series) in enumerate(zip(wide.columns, values)):
            for t in range(max(LAGS) - 1, train_end - horizon_step):
                row = _features(series, t)
                if row is None:
                    continue
                target = series[t + horizon_step]
                if not np.isfinite(target) or series[t] <= 0:
                    continue
                row |= self._extra_row(position, series, col, t)
                if self.use_stack:
                    row["is_total"] = 1.0
                rows.append(row)
                targets.append(target / series[t] / self._multiplier(t, horizon_step, col))

        if self.use_stack:
            extra_rows, extra_targets = self._category_rows(wide, train_end, horizon_step)
            rows.extend(extra_rows)
            targets.extend(extra_targets)
        return pd.DataFrame(rows), np.asarray(targets, dtype=float)

    def _category_rows(
        self, wide: pd.DataFrame, train_end: int, horizon_step: int
    ) -> tuple[list[dict], list[float]]:
        """Пять категорий трат как отдельные обучающие ряды, а не как признаки.

        Ход другой, чем у `categories`, и бьёт в другое место. Признаки описывают
        ряд, но число обучающих пар не меняют, а здесь пар становится вшестеро
        больше: 12 168 рядов вместо 2 028. Покрытие календарных месяцев то же —
        апреля в обучении фолда 0 по-прежнему нет, — но примеров на каждый
        переход в шесть раз больше, и главное, формы динамики у категорий разные.
        Общепит живёт летом, продовольствие — декабрём. Модель, которой показали
        только итог, заучивает конкретные переходы этого одного ряда; модель,
        которой показали шесть разных сезонных рисунков, вынуждена опираться
        на признаки, а не на то, каким был прошлый декабрь.

        Признак `is_total` отличает строки итога от строк категорий. Он безопасен:
        при прогнозе принимает значение 1, которое в обучении есть.
        """
        context = self._require_context()
        rows, targets = [], []
        for name in sorted(context.categories):
            frame = context.categories[name].reindex(columns=wide.columns)
            values = frame.to_numpy(dtype=float).T
            for position, (col, series) in enumerate(zip(wide.columns, values)):
                if not np.isfinite(series).all():
                    continue
                for t in range(max(LAGS) - 1, train_end - horizon_step):
                    row = _features(series, t)
                    if row is None:
                        continue
                    target = series[t + horizon_step]
                    if not np.isfinite(target) or series[t] <= 0:
                        continue
                    # Доли категорий и новости описывают муниципалитет целиком,
                    # а строка здесь — одна его категория. Подставлять их значило бы
                    # приписать части свойства целого, поэтому остаются пропуском.
                    row["is_total"] = 0.0
                    rows.append(row)
                    targets.append(
                        target / series[t] / self._multiplier(t, horizon_step, col)
                    )
        return rows, targets

    def _pretrain_set(
        self, wide: pd.DataFrame, train_end: int, horizon: int
    ) -> dict[int, tuple[pd.DataFrame, np.ndarray]]:
        """Обучающие примеры из длинных отраслевых рядов — та же конструкция признаков.

        Ряды обрезаются по origin. Это главная ловушка всего подхода: отраслевые
        данные идут до 2026 года, а прогнозируем мы 2024-й, и необрезанный ряд
        рассказал бы модели, чем кончилась та самая динамика, которую она учится
        предсказывать. Ошибка была бы тихой: метрики улучшились бы, а результат
        не значил бы ничего.

        Набор признаков ровно тот же, что у панели, и это условие сравнения:
        отличаться `global_gbm_pretrain` от `global_gbm` должен предобучением,
        а не другим набором колонок. Смысл переносится не у всех признаков —
        `level_log` у отраслевого ряда это логарифм миллиардов рублей, а у панели
        логарифм рублей на человека, — но деревья первой стадии просто разрежут
        его в своём диапазоне, и на панельных строках эти разрезы окажутся
        нерабочими. Это честнее, чем подменять колонку и сравнивать разное.

        Календарный месяц берётся из самого ряда: формула `(t % 12) + 1` верна
        только для панели, которая начинается с января, а отраслевые ряды
        начинаются кто с января 2017-го, кто с декабря 2018-го.
        """
        context = self._require_context()
        if not context.long_series:
            return {}

        origin = pd.to_datetime(wide.index).to_period("M")[train_end - 1]
        out: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
        for step in range(1, horizon + 1):
            rows, targets = [], []
            for full_series in context.long_series.values():
                series = full_series.loc[:origin]
                values = series.to_numpy(dtype=float)
                for t in range(max(LAGS) - 1, len(values) - step):
                    row = _features(values, t)
                    if row is None:
                        continue
                    row["month"] = series.index[t].month
                    rows.append(row)
                    targets.append(values[t + step] / values[t])
            if rows:
                out[step] = (pd.DataFrame(rows), np.asarray(targets, dtype=float))
        return out

    def predict(self, wide: pd.DataFrame, train_end: int, horizon: int) -> np.ndarray:
        """Прогноз для всех рядов панели. Возвращает матрицу (рядов × горизонт)."""
        values = wide.to_numpy(dtype=float).T
        t = train_end - 1
        out = np.full((values.shape[0], horizon), np.nan)

        if self._shares is not None:
            self._shares = _CategoryShares(self._require_context().categories, wide.columns)

        rows, index = [], []
        for position, (col, series) in enumerate(zip(wide.columns, values)):
            row = _features(series, t)
            if row is not None:
                row |= self._extra_row(position, series, col, t)
                rows.append(row)
                index.append(position)
        if not rows:
            return out

        frame = pd.DataFrame(rows)
        base = values[index, t]
        if self._factor is not None:
            multipliers = self._factor.multipliers_for(wide.columns)[index]
        else:
            multipliers = np.ones((len(index), horizon))
        for step in range(1, horizon + 1):
            features = frame.reindex(columns=self._columns[step])
            ratio = self._models[step].predict(features.to_numpy(dtype=float))
            out[index, step - 1] = base * ratio * multipliers[:, step - 1]
        return out
