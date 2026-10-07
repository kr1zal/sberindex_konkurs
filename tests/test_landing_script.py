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
                           "function searchRows", "function groupDigits", "function pluralRu"):
            with self.subTest(copy=duplicated):
                self.assertNotIn(duplicated, source)

    def test_counting_skips_the_panel_line(self) -> None:
        # Строка панели («N рядов × M месяцев») на полпути счёта читалась как «946 рядов × 11 месяца»:
        # формы слов при счёте не пересчитываются, и считаются только числа карточек.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        self.assertIn('document.querySelectorAll(".stat:not(.stat-panel) .stat-number")', source)

    def test_counting_is_short(self) -> None:
        # На полпути счёта число карточки — не то, что в итоге («9,7%» вместо «23,0%»), и беглый взгляд или снимок
        # на первой секунде не должен его застать: счёт короче шестисот миллисекунд.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        duration = int(re.search(r"const COUNT_MS = (\d+);", source).group(1))
        self.assertLessEqual(duration, 600)
        self.assertIn("animate(COUNT_MS,", source)

    def test_axis_labels_follow_the_scale_step(self) -> None:
        # Подписи оси раздела 01 и графика проверки фактом считает помощник по шагу шкалы:
        # «98% / 100% / 103%» при шаге 2,5% — результат целых подписей.
        source = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        self.assertIn("percentTick(value, target.scale.step)", source)
        self.assertIn("formatTick(value, scale.step)", source)
        self.assertNotIn("Math.round(value * 100)", source)


if __name__ == "__main__":
    unittest.main()
