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
панель» не должно разъехаться между двумя копиями.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HINT = (
    "партия считана на другом протоколе или выборке (например, `configs/baseline.yaml` "
    "с 300 рядами); в файл ничего не записано"
)


def check_same_panel(previous: pd.DataFrame, fresh: pd.DataFrame, *, path: Path) -> None:
    """Падает с ValueError, если свежие строки считаны не на том же множестве рядов
    или не на тех же фолдах, что файл `path`. Ничего не возвращает и ничего не пишет.

    Ряды файла — все его строки: все модели, включая строки с отказами (`error`).
    На ряду, где модель только отказывала, она всё равно гонялась.

    Фолды сверяются по горизонтам, если колонка `horizon` есть в обоих кадрах:
    число фолдов у горизонтов разное по построению. Горизонт, которого в файле
    ещё нет, не сверяется — сверять его не с чем.
    """
    by_horizon = "horizon" in previous.columns and "horizon" in fresh.columns
    if fresh.empty:
        # Партия без строк — аномалия сама по себе: слияние с ней ничего бы
        # не заменило, а прогон выглядел бы успешным.
        raise ValueError(
            f"{path}: свежая партия пуста — прогон не дал ни одной строки.\n"
            f"{_details(previous, fresh, by_horizon)}\n"
            "Проверьте список моделей, выборку рядов и фолды в конфиге; в файл ничего не записано."
        )
    if previous.empty:
        return

    problems = []
    if set(previous["mo"]) != set(fresh["mo"]):
        problems.append("другое множество рядов")
    in_file, in_batch = _folds(previous, by_horizon), _folds(fresh, by_horizon)
    for horizon in sorted(in_batch):
        if horizon in in_file and in_batch[horizon] != in_file[horizon]:
            problems.append("другие фолды" + (f" на горизонте {horizon}" if by_horizon else ""))
    if problems:
        raise ValueError(
            f"{path}: свежая партия не сходится с файлом — {'; '.join(problems)}.\n"
            f"{_details(previous, fresh, by_horizon)}\n"
            f"Вероятно, {HINT}."
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


def _folds(frame: pd.DataFrame, by_horizon: bool) -> dict:
    """Множества фолдов по горизонтам; без горизонта — одно множество под ключом None.

    Ключи одного словаря всегда однородны — либо один None, либо целые горизонты, —
    поэтому их можно сортировать."""
    if frame.empty:
        return {}
    if not by_horizon:
        return {None: set(frame["fold"].tolist())}
    return {int(h): set(folds.tolist()) for h, folds in frame.groupby("horizon")["fold"]}


def _details(previous: pd.DataFrame, fresh: pd.DataFrame, by_horizon: bool) -> str:
    in_file = set() if previous.empty else set(previous["mo"])
    in_batch = set() if fresh.empty else set(fresh["mo"])
    return (
        f"  рядов: в файле {len(in_file)}, в партии {len(in_batch)}; "
        f"только в файле {len(in_file - in_batch)}, только в партии {len(in_batch - in_file)}\n"
        f"  фолды в файле: {_folds_text(_folds(previous, by_horizon))}\n"
        f"  фолды в партии: {_folds_text(_folds(fresh, by_horizon))}"
    )


def _folds_text(folds: dict) -> str:
    if not folds:
        return "нет"
    parts = []
    for horizon in sorted(folds):
        values = "{" + ", ".join(map(str, sorted(folds[horizon]))) + "}"
        parts.append(values if horizon is None else f"h{horizon} {values}")
    return ", ".join(parts)
