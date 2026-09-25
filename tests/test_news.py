"""Словарь новостных тем: `src/news.py` и те, кто по нему ищет.

Паттерн проверяется на словах в той латинице, которую даёт `transliterate`,
признаки — на синтетических заголовках кириллицей: так они приходят из корпуса.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import news  # noqa: E402
from src.models.two_stage import TwoStageNews  # noqa: E402

# scripts/ — не пакет: скрипт грузится по пути.
_spec = importlib.util.spec_from_file_location("compare_headline_body", ROOT / "scripts" / "compare_headline_body.py")
compare_headline_body = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_headline_body)


def in_topic(topic: str, text: str) -> bool:
    return re.search(news.topic_pattern(news.NATIONAL_TOPICS[topic]), text) is not None


class TopicPatternTest(unittest.TestCase):
    """Корень темы должен начинать слово: подстрока в середине слова — чужое слово."""

    CASES = {
        "dkp": {"stavka": True, "czb snizil stavku": True, "vystavka": False, "postavka": False,
                "dostavka": False, "otstavka": False},
        "ceny": {"cena": True, "czena": True, "tseny": True, "centr": False, "czentr": False,
                 "tsentr": False},
        "kredit": {"dolg": True, "dolgi": True, "dolgo": False, "bank": True, "banki": True,
                   "banket": False, "stavka": True, "vystavka": False},
        "zanjatost": {"rabota": True, "bezrabotica": True, "zarabotok": False, "razrabotka": False,
                      "obrabotka": False},
    }

    def test_roots_match_only_from_the_start_of_a_word(self):
        for topic, cases in self.CASES.items():
            for text, expected in cases.items():
                with self.subTest(topic=topic, text=text):
                    self.assertIs(in_topic(topic, text), expected)

    def test_root_with_word_boundary_works_as_before(self):
        # `\bczb\b` не трогается: он и прежде ловил только отдельное слово «ЦБ».
        self.assertIn(r"\bczb\b", news.topic_pattern(news.NATIONAL_TOPICS["dkp"]).split("|"))
        before = re.compile(r"\bczb\b")
        for text in ("czb", "glava czb nabiullina", "czb.", "czbank", "aczb", "czb2"):
            with self.subTest(text=text):
                self.assertEqual(in_topic("dkp", text), before.search(text) is not None)

    def test_transliteration_gives_lowercase_latin(self):
        # Граница начала слова — «перед корнем не строчная латинская буква»: верно,
        # только пока transliterate сначала понижает регистр, потом меняет алфавит.
        self.assertEqual(news.transliterate("Ставка ЦБ, Выставка"), "stavka czb, vystavka")


def headlines() -> pd.DataFrame:
    """Четыре заголовка одного месяца, в каждой из трёх тем по одному настоящему."""
    titles = ["Выставка цветов в центре города", "ЦБ повысил ключевую ставку",
              "Цены на бензин выросли", "Банкет в честь юбилея"]
    return pd.DataFrame({
        "url": [f"https://x.ru/{i}" for i in range(len(titles))],
        "title": titles,
        "date": pd.to_datetime(["2024-01-05"] * len(titles)),
        "region_name": ["Область"] * len(titles),
        "domain": ["x.ru"] * len(titles),
    })


class FeaturesTest(unittest.TestCase):
    """Обе сборки признаков идут через один построитель паттерна. По старому словарю
    ДКП было бы 2/4 (с «выставкой»), цены 2/4 (с «центром»), кредит 3/4 (с «выставкой»
    и «банкетом»)."""

    def test_national_features(self):
        row = news.national_features(headlines()).iloc[0]
        self.assertEqual((row["t_dkp"], row["t_ceny"], row["t_kredit"]), (0.25, 0.25, 0.25))

    def test_monthly_features(self):
        frame = headlines()
        frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
        frame["norm"] = frame["title"].map(news.transliterate)
        row = news.monthly_features(frame).iloc[0]
        self.assertEqual((row["t_ceny"], row["t_kredit"]), (0.25, 0.25))

    def test_headline_against_body_comparison_uses_the_same_pattern(self):
        hits = compare_headline_body.topic_hits(headlines()["title"])
        self.assertEqual(hits["dkp"].tolist(), [False, True, False, False])
        self.assertEqual(hits["ceny"].tolist(), [False, False, True, False])


class NewsCandidatesTest(unittest.TestCase):
    def test_intensity_is_not_a_candidate(self):
        # Интенсивность нормирована средним и разбросом всего периода — заглядывание вперёд.
        columns = ["n_articles", "n_outlets", "t_ceny", "t_dkp", "intensity"]
        self.assertEqual(TwoStageNews._candidates(columns), ["t_ceny", "t_dkp"])


if __name__ == "__main__":
    unittest.main()
