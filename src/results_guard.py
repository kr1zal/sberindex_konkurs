"""Страховка от тихой перезаписи результатов прогонов.

Результаты копятся в файлах слиянием: модели гоняются частями, и свежая партия
заменяет в файле строки своих моделей, а остальное оставляет. Слияние молча
предполагает, что партия считана на тех же рядах и фолдах, что уже лежат в файле.
Когда это не так, файл становится смесью двух протоколов, а сводка по нему —
сравнением моделей на разных множествах рядов. 21.09 прогон `naive_last` на выборке
из 300 рядов (`configs/baseline.yaml`) затёр её полнопанельные строки
в `results/per_series.csv`, и сводка показала 1 908 вместо 1 852 — без единой
ошибки, заметили по числу.

Проверка одна на `src/run.py` и `scripts/horizons.py`: правило «один файл — одна
панель» не должно разъехаться между двумя копиями. Сверяются дважды: план прогона —
до расчёта (`check_plan`), чтобы не тратить часы счёта на партию, которую потом
не слить, и сама партия — перед записью (`check_same_panel`). Сравнение у обеих
общее (`_compare`).

Предел сверки: фолды сравниваются по номерам (в файлах горизонтов — по номерам внутри
горизонта), а не по границам обучения `train_end`. Смену `split.horizon` в `run.py`
или удлинение панели при том же числе фолдов она пропустит — штампа протокола
в файлах нет. Прогон на намеренно другом протоколе пишется в другой `output.dir`.

Здесь же правило отказа (`refused`, `failure_reason`), одно на прогоны и отчёт,
чтение файлов результатов (`read_results`) и снимки строк, которые слияние заменяет
(`save_realization`). Модуль лёгкий, только pandas и стандартная библиотека: отчёт
берёт правила отсюда, не поднимая `src.run` со всеми моделями.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Проверка останавливает и намеренную смену протокола. Подсказка — как сменить его,
# не смешав файлы: убрать в сторону один файл по рядам мало, шаги горизонтов
# своей сверки не имеют и слились бы со старыми.
INTENTIONAL = (
    "Если протокол сменён намеренно, задайте в конфиге другой `output.dir` "
    "(для horizons.py — другой `output.prefix`), а не убирайте из каталога один файл "
    "по рядам: у `horizons_steps.csv` своей сверки нет, и новые шаги слились бы со старыми"
)
HINT = (
    "партия считана на другом протоколе или выборке (например, `configs/baseline.yaml` "
    "с 300 рядами); в файл ничего не записано. " + INTENTIONAL
)
PLAN_HINT = (
    "прогон запущен с другим конфигом или выборкой (например, `configs/baseline.yaml` "
    "с 300 рядами), и его результаты с этим файлом не слить; модели не запускались. "
    + INTENTIONAL
)


def check_same_panel(previous: pd.DataFrame, fresh: pd.DataFrame, *, path: Path) -> None:
    """Падает с ValueError, если свежие строки считаны не на том же множестве рядов
    или не на тех же фолдах, что файл `path`. Ничего не возвращает и ничего не пишет.

    Ряды файла — все его строки: все модели, включая строки с отказами (`error`).
    На ряду, где модель только отказывала, она всё равно гонялась.

    Фолды сверяются по номерам, а не по `train_end` (предел — в докстринге модуля),
    и по горизонтам, если колонка `horizon` есть в обоих кадрах: число фолдов
    у горизонтов разное по построению. Горизонт, которого в файле ещё нет,
    не сверяется — сверять его не с чем.
    """
    by_horizon = "horizon" in previous.columns and "horizon" in fresh.columns
    if fresh.empty:
        # Партия без строк — аномалия сама по себе: слияние с ней ничего бы
        # не заменило, а прогон выглядел бы успешным.
        details = _details(_series(previous), set(), _folds(previous, by_horizon), {}, "в партии")
        raise ValueError(
            f"{path}: свежая партия пуста — прогон не дал ни одной строки.\n{details}\n"
            "Проверьте список моделей, выборку рядов и фолды в конфиге; в файл ничего не записано."
        )
    if previous.empty:
        return
    _compare(
        previous, _series(fresh), _folds(fresh, by_horizon), by_horizon,
        path=path, subject="свежая партия", where="в партии", hint=f"Вероятно, {HINT}.",
    )


def check_plan(
    previous: pd.DataFrame, series: Iterable[str], folds: list[int] | dict[int, list[int]],
    *, path: Path,
) -> None:
    """Падает с ValueError, если план прогона не сходится с файлом `path`, — до расчёта.
    Ничего не возвращает и ничего не пишет.

    `series` — ряды, которые пойдут в прогон (колонки матрицы после отбора выборки);
    `folds` — номера фолдов списком, а для прогона по горизонтам — словарём
    {горизонт: номера}. Правила те же, что в `check_same_panel`: ряды — всем файлом,
    фолды — по горизонтам, которые в файле уже есть. Перед записью партия сверяется
    ещё раз, но там несовпадение всплывает, когда часы счёта уже потрачены.
    """
    if previous.empty:
        return
    by_horizon = isinstance(folds, dict) and "horizon" in previous.columns
    if by_horizon:
        planned = {int(h): set(numbers) for h, numbers in folds.items()}
    elif isinstance(folds, dict):
        planned = {None: set().union(*folds.values())}
    else:
        planned = {None: set(folds)}
    _compare(
        previous, set(series), planned, by_horizon,
        path=path, subject="план прогона", where="в плане", hint=f"Вероятно, {PLAN_HINT}.",
    )


def uneven_series(per_series: pd.DataFrame) -> pd.Series | None:
    """Число уникальных `mo` по моделям, считая и строки с отказами (`error`), —
    это множество рядов, на котором модель гонялась, а не на котором она справилась.
    Возвращает Series (index=model), если хотя бы у двух моделей числа различаются, иначе None.

    Колонка `серий` из сводки для этого не годится: она считается по строкам
    без отказов и законно различается у моделей, отказавших на части рядов.
    """
    counts = per_series.groupby("model")["mo"].nunique()
    return counts if counts.nunique() > 1 else None


def resolve_results(path: Path | str) -> Path:
    """Путь, который действительно читать: обычный файл результатов, если он есть,
    иначе его сжатый снимок `path` + `.gz` (`per_series.csv` → `per_series.csv.gz`).

    Конвейер (`src/run.py`, `scripts/horizons.py`, `scripts/forecast_forward.py`,
    `scripts/changepoints.py`) всегда пишет обычные CSV вне git — начисто на каждый
    прогон. В git попадают только снимки пяти крупных файлов, сделанные отдельной
    командой (`scripts/export_results.py`) и потому не обязанные совпадать по времени
    с последним прогоном. Обычный файл проверяется первым не просто как более свежий:
    слияние (`src.run.merge_results`, `scripts.horizons.merge_into`) само читает
    прежние строки через эту функцию, и партия, даже частичная, на чистом клоне сначала
    подхватывает снимок целиком, а уже потом пишет обычный CSV — поэтому обычный файл,
    когда он появился, полный, а не часть панели одной партии. `.gz` находится только
    на чистом клоне или когда конвейер после клонирования ещё не запускался.

    `path` — `Path` или строка: отчёт зовёт `table("../results/per_series.csv")` строками,
    а `.exists()` есть только у `Path`, поэтому вход приводится к `Path` сразу.

    Падает `FileNotFoundError`, называющим оба пути, если нет ни одного, — это исключение
    ловит `table()` отчёта, чтобы явно сказать, каких данных не хватает, а не упасть
    на разборе несуществующего файла где-то внутри pandas."""
    path = Path(path)
    if path.exists():
        return path
    gz = Path(f"{path}.gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"нет ни обычного файла результатов, ни его снимка: {path}, {gz}")


def read_results(path: Path | str, **kwargs) -> pd.DataFrame:
    """Файл результатов — числами ровно такими, какими они записаны. Прочие аргументы —
    как у `pd.read_csv`.

    Разбор чисел pandas по умолчанию не обратим: примерно каждое восьмое число читается
    на единицу последнего разряда не тем, что записано. Слияние читает файл и пишет его
    заново, и каждая партия сдвигала бы сохранённые метрики чужих моделей. Все чтения
    файлов результатов в прогонах и бэкфилле идут через эту функцию — правило одно.

    Путь проходит через `resolve_results`: обычный CSV или, если его нет, снимок `.csv.gz` —
    pandas распознаёт gzip по расширению сам, без дополнительных аргументов."""
    return pd.read_csv(resolve_results(path), float_precision="round_trip", **kwargs)


def realization_stamp() -> str:
    """Штамп снимков: время UTC до секунды, `YYYYmmddTHHMMSS`. Берётся один раз на вызов
    `main`, чтобы снимки одной партии совпадали по имени."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def save_realization(rows: pd.DataFrame, directory: Path, label: str, stamp: str) -> Path:
    """Пишет строки, которые слияние сейчас заменит, в `directory/<label>__<stamp>.csv`,
    печатает путь и возвращает его.

    Прогон той же модели заменяет её строки, и прежняя реализация пропадала, а модель
    со случайностью показывается средним и размахом по реализациям: дообученный Chronos
    на одних данных дал 1 606 и 1 772. Слияния зовут это после сверки партии с файлом
    и до замены — партия, которую сверка остановила, снимка не оставляет. Снимок
    с тем же именем не перезаписывается: к имени добавляется номер."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}__{stamp}.csv"
    number = 2
    while path.exists():
        path = directory / f"{label}__{stamp}__{number}.csv"
        number += 1
    rows.to_csv(path, index=False)
    print(f"снимок заменяемых строк: {path}")
    return path


def failure_reason(exc: Exception) -> str:
    """Текст отказа с типом исключения. Голый `assert` в библиотеке падает без текста:
    пустое поле уходит в CSV, читается обратно как NaN — и отказ выглядит успехом."""
    return f"{type(exc).__name__}: {exc}"


def refused(per_series: pd.DataFrame) -> pd.Series:
    """Строки-отказы: с текстом отказа или без MAE. Строка без MAE — отказ, даже если
    текст пуст: по одной колонке `error` такой отказ после CSV неотличим от успеха."""
    return per_series["error"].notna() | per_series["mae"].isna()


def _folds(frame: pd.DataFrame, by_horizon: bool) -> dict:
    """Множества фолдов по горизонтам; без горизонта — одно множество под ключом None.

    Ключи одного словаря всегда однородны — либо один None, либо целые горизонты, —
    поэтому их можно сортировать."""
    if frame.empty:
        return {}
    if not by_horizon:
        return {None: set(frame["fold"].tolist())}
    return {int(h): set(folds.tolist()) for h, folds in frame.groupby("horizon")["fold"]}


def _compare(
    previous: pd.DataFrame, series: set, folds: dict, by_horizon: bool,
    *, path: Path, subject: str, where: str, hint: str,
) -> None:
    """Общая сверка файла с партией или с планом: ряды — всем файлом, фолды — там,
    где горизонт в файле уже есть. `where` — «в партии» или «в плане» для сообщения."""
    file_series, file_folds = _series(previous), _folds(previous, by_horizon)
    problems = []
    if file_series != series:
        problems.append("другое множество рядов")
    for horizon in sorted(folds):
        if horizon in file_folds and folds[horizon] != file_folds[horizon]:
            problems.append("другие фолды" + (f" на горизонте {horizon}" if by_horizon else ""))
    if problems:
        raise ValueError(
            f"{path}: {subject} не сходится с файлом — {'; '.join(problems)}.\n"
            f"{_details(file_series, series, file_folds, folds, where)}\n"
            f"{hint}"
        )


def _series(frame: pd.DataFrame) -> set:
    return set() if frame.empty else set(frame["mo"])


def _details(file_series: set, series: set, file_folds: dict, folds: dict, where: str) -> str:
    return (
        f"  рядов: в файле {len(file_series)}, {where} {len(series)}; "
        f"только в файле {len(file_series - series)}, только {where} {len(series - file_series)}\n"
        f"  фолды в файле: {_folds_text(file_folds)}\n"
        f"  фолды {where}: {_folds_text(folds)}"
    )


def _folds_text(folds: dict) -> str:
    if not folds:
        return "нет"
    parts = []
    for horizon in sorted(folds):
        values = "{" + ", ".join(map(str, sorted(folds[horizon]))) + "}"
        parts.append(values if horizon is None else f"h{horizon} {values}")
    return ", ".join(parts)
