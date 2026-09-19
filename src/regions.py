"""Привязка рядов к регионам и ОКТМО по справочнику муниципальных образований СберИндекса.

Справочник: «Данные о границах и изменениях муниципальных образований в России»,
https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities
Лицензия CC BY-SA 4.0.

Ключевая тонкость — время. Справочник версионный: у каждой записи есть year_from/year_to,
и муниципалитеты сливаются, переименовываются и меняют тип. Наши данные за 2023–2024,
поэтому сопоставлять надо с записями, действовавшими в этот период, а не с текущими.
Фильтр year_to == 9999 выбрасывает, например, городской округ Павловский Посад,
Электрогорск и город Сасово — они были объединены в 2024-м, но в нашем периоде жили.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DICT_PATH = Path("data/reference/t_dict_municipal_districts.xlsx")
PANEL_YEAR_FROM = 2023
PANEL_YEAR_TO = 2024


@dataclass
class JoinReport:
    n_series: int
    n_resolved: int
    n_ambiguous: int
    n_unmatched: int

    def as_text(self) -> str:
        pct = 100.0 * self.n_resolved / self.n_series if self.n_series else 0.0
        return (
            f"рядов: {self.n_series}\n"
            f"  регион определён: {self.n_resolved} ({pct:.1f}%)\n"
            f"  название неоднозначно: {self.n_ambiguous}\n"
            f"  нет в справочнике: {self.n_unmatched}"
        )


def _normalise(name: str) -> str:
    """Схлопывает повторные пробелы — в выгрузке встречаются двойные."""
    return re.sub(r"\s+", " ", str(name)).strip()


def load_dictionary(path: str | Path = DICT_PATH) -> pd.DataFrame:
    """Справочник, суженный до записей, действовавших в период панели."""
    d = pd.read_excel(path)
    d["year_from"] = pd.to_numeric(d["year_from"], errors="coerce")
    d["year_to"] = pd.to_numeric(d["year_to"], errors="coerce")
    alive = (d["year_from"] <= PANEL_YEAR_TO) & (d["year_to"] >= PANEL_YEAR_FROM)
    d = d.loc[alive].copy()
    d["_key"] = d["municipal_district_name"].map(_normalise)
    return d


def attach_regions(entities: pd.DataFrame, dictionary: pd.DataFrame) -> tuple[pd.DataFrame, JoinReport]:
    """Добавляет region_name, oktmo и territory_id к таблице рядов.

    Одноимённые муниципалитеты в разных регионах остаются неразрешёнными намеренно:
    в выгрузке нет ни ОКТМО, ни кода региона, и угадывать привязку по названию
    значило бы приписать ряду чужой регион. Такие ряды помечаются
    ``region_resolved = False`` и участвуют в моделировании, но исключаются
    из региональных агрегатов и с карты.
    """
    ent = entities.copy()
    ent["_key"] = ent["mo"].map(_normalise)

    # Неоднозначность считаем по числу РАЗНЫХ территорий, а не строк: у одного места
    # бывает несколько версионных записей (переименование, смена типа района на округ),
    # и они делят общий territory_id. Это одна территория, а не омонимы.
    counts = dictionary.groupby("_key")["territory_id"].nunique()
    unique_keys = counts[counts == 1].index
    unique_rows = (
        dictionary.loc[dictionary["_key"].isin(unique_keys)]
        .sort_values("year_to")
        .drop_duplicates("_key", keep="last")
    )
    lookup = unique_rows.set_index("_key")[
        ["region_name", "region_code", "oktmo", "territory_id", "municipal_district_type"]
    ]

    ent = ent.join(lookup, on="_key")

    # Запасное правило: справочник — снимок на октябрь 2024, а часть регионов
    # (Забайкальский край и другие) преобразовали районы в муниципальные округа
    # позже. Название меняется, территория та же, поэтому пробуем обратную замену
    # типа — но только если она даёт единственного кандидата.
    retyped = ent["region_name"].isna()
    if retyped.any():
        alt_keys = ent.loc[retyped, "_key"].str.replace(
            "муниципальный округ", "муниципальный район", regex=False
        )
        alt = alt_keys.map(lambda k: lookup.loc[k] if k in lookup.index else None)
        for column in lookup.columns:
            filled = alt.map(lambda row, c=column: None if row is None else row[c])
            ent.loc[retyped, column] = ent.loc[retyped, column].fillna(filled)

    ent["n_candidates"] = ent["_key"].map(counts).fillna(0).astype(int)
    ent["region_resolved"] = ent["region_name"].notna()

    report = JoinReport(
        n_series=len(ent),
        n_resolved=int(ent["region_resolved"].sum()),
        n_ambiguous=int((ent["n_candidates"] > 1).sum()),
        # не сопоставлены = не разрешены и при этом не омонимы (иначе запасное
        # правило, отработавшее выше, в отчёт бы не попало)
        n_unmatched=int((~ent["region_resolved"] & (ent["n_candidates"] <= 1)).sum()),
    )
    return ent.drop(columns="_key"), report
