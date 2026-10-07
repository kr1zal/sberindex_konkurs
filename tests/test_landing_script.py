"""Скрипт главной (`site/landing.js`) и его помощники (`site/landing-lib.js`): как они соединены."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LandingScriptWiringTest(unittest.TestCase):
    def test_page_script_takes_the_helpers_from_the_library(self) -> None:
        # Страница грузит помощники перед скриптом (порядок проверяет тест сборки), а сам скрипт не
        # держит их копий: правило живёт в одном месте, и тесты через node проверяют именно его.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        self.assertIn("} = LandingLib;", source)
        for duplicated in ("function niceScale", "function numberTokens", "function countedText",
                           "function groupDigits", "function pluralRu"):
            with self.subTest(copy=duplicated):
                self.assertNotIn(duplicated, source)

    def test_counting_covers_the_four_numbers_of_the_strip_and_nothing_else(self) -> None:
        # Считаются только числа полосы на обложке: слова рядом с числом на полпути счёта не пересчитываются
        # («946 рядов × 11 месяца»), и строк со словами при числе в полосе нет — строка панели удалена.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        self.assertIn('document.querySelectorAll(".stat .stat-number")', source)
        self.assertNotIn("stat-panel", source)

    def test_counting_is_short(self) -> None:
        # Числа полосы стоят на первом экране и считаются от нуля сразу при загрузке; на полпути число — не то, что
        # в итоге («9,7%» вместо «23,0%»), поэтому счёт короче 0,7 секунды.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        duration = int(re.search(r"const COUNT_MS = (\d+);", source).group(1))
        self.assertLessEqual(duration, 700)
        self.assertIn("animate(COUNT_MS,", source)

    def test_the_cover_chart_is_an_illustration_without_axis_labels(self) -> None:
        # График обложки — иллюстрация: линии и медиана, без подписей месяцев; скрипт не ищет несуществующих узлов.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        start = source.index("function initHero(story) {")
        hero = source[start:source.index("\n  }\n", start)]
        self.assertNotIn("hero-axis", hero)
        self.assertNotIn("setAxisLabels", hero)
        self.assertIn('byId("hero-lines")', hero)
        self.assertIn('byId("hero-median")', hero)

    def test_the_cover_search_is_gone_from_the_script(self) -> None:
        # Поиск остался только на странице прогноза: скрипт главной не знает ни поля, ни списка подсказок.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        for gone in ("initSearch", "mo-q", "mo-list", "mo-note", "demo/data/index.json", "fetch("):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, source)
        # Блоки запускаются под try: без поиска их восемь — шапка, обложка, числа, история, горизонты, факт,
        # изломы, прогноз по муниципалитету.
        launch = source[source.index("  [\n    [\"шапка\""):source.index("].forEach((block) => {")]
        self.assertEqual(len(re.findall(r'^\s*\["', launch, re.M)), 8)

    def test_axis_labels_follow_the_scale_step(self) -> None:
        # Подписи оси раздела 01 и графика проверки фактом считает помощник по шагу шкалы:
        # «98% / 100% / 103%» при шаге 2,5% — результат целых подписей.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        self.assertIn("percentTick(value, target.scale.step)", source)
        self.assertIn("formatTick(value, scale.step)", source)
        self.assertNotIn("Math.round(value * 100)", source)


if __name__ == "__main__":
    unittest.main()
