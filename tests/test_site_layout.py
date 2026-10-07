"""Вёрстка главной и стенда: то, что выяснилось только в браузере и держится на CSS и скриптах.

Сборка не нужна: читаются сами файлы `site/`, `demo/` и шаблон. Проверки — о формулах и
правилах, а не о пикселях: кегль заголовка обложки считается по формулам из стилей на
каждой ширине экрана; раскладка подписей отметок графика стенда — чистая функция
`LineChart.assignMarkerRows`, её запускает node (без него проверка пропускается).
"""
from __future__ import annotations

import ast
import json
import operator
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# «одного числа» в Unbounded 600 с межбуквенным -0,02em: 677,9 px при кегле 80 px
# (измерено в Chromium) — 8,47 кегля; берём с небольшим запасом вверх.
AMBER_PHRASE_EM = 8.5


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_FUNCTIONS = {"clamp": lambda low, value, high: max(low, min(value, high)), "min": min, "max": max}


def _evaluate(node: ast.AST) -> float:
    """Арифметика и вызовы clamp/min/max — больше ничего выражение из стилей не содержит."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ValueError(f"выражение не из того, что умеет проверка: {ast.dump(node)}")


def _css_length(expr: str, viewport: float) -> float:
    """Значение выражения из стилей (px, rem, vw, calc, clamp, min, max) в пикселях."""
    text = expr.strip()
    text = re.sub(r"(\d+(?:\.\d+)?)vw", lambda m: f"({m.group(1)}*{viewport}/100)", text)
    text = re.sub(r"(\d+(?:\.\d+)?)rem", lambda m: f"({m.group(1)}*16)", text)
    text = re.sub(r"(\d+(?:\.\d+)?)px", lambda m: f"({m.group(1)})", text)
    text = text.replace("calc(", "(")
    return _evaluate(ast.parse(text, mode="eval"))


def _hero_title_font_sizes(css: str) -> list[tuple[int, str]]:
    """Правила font-size у .hero-title: (min-width медиазапроса или 0, выражение)."""
    rules = [(0, re.search(r"\n\.hero-title \{[^}]*?font-size:\s*([^;]+);", css).group(1))]
    for width, expr in re.findall(
        r"@media \(min-width: (\d+)px\) \{\s*\.hero-title \{\s*font-size:\s*([^;]+);", css
    ):
        rules.append((int(width), expr))
    return sorted(rules)


class HeroTitleTest(unittest.TestCase):
    """Заголовок обложки: «одного числа» целиком в строке на любой ширине — две строки,
    а не три."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.landing_css = _read("site/landing.css")
        site_css = _read("site/site.css")
        cls.wrap_max = float(re.search(r"\.wrap \{[^}]*?max-width:\s*(\d+)px", site_css).group(1))
        cls.pad_wide = float(re.search(r"\.wrap \{[^}]*?padding-inline:\s*(\d+)px", site_css).group(1))
        cls.pad_phone = float(re.search(
            r"@media \(max-width: 639px\) \{\s*\.wrap \{\s*padding-inline:\s*(\d+)px", site_css).group(1))
        grid = re.search(r"\.hero-grid \{[^}]*?grid-template-columns:\s*minmax\(0,\s*([\d.]+)fr\)\s*"
                         r"minmax\(0,\s*([\d.]+)fr\);[^}]*?gap:\s*(\d+)px", cls.landing_css)
        cls.left_fr, cls.right_fr, cls.gap = float(grid.group(1)), float(grid.group(2)), float(grid.group(3))

    def _column(self, viewport: int) -> float:
        """Ширина колонки, в которой стоит заголовок."""
        if viewport < 640:
            return viewport - 2 * self.pad_phone
        if viewport < 960:  # одна колонка на всю ширину обёртки
            return viewport - 2 * self.pad_wide
        content = min(viewport, self.wrap_max) - 2 * self.pad_wide - self.gap
        return content * self.left_fr / (self.left_fr + self.right_fr)

    def _font(self, viewport: int) -> float:
        expr = [e for width, e in _hero_title_font_sizes(self.landing_css) if width <= viewport][-1]
        return _css_length(expr, viewport)

    def test_amber_phrase_fits_the_column_at_every_width(self) -> None:
        for viewport in list(range(320, 640, 10)) + list(range(640, 2561, 20)) + [959, 960, 1279, 1280, 1304]:
            with self.subTest(viewport=viewport):
                font, column = self._font(viewport), self._column(viewport)
                self.assertLessEqual(
                    font * AMBER_PHRASE_EM, column,
                    f"при {viewport} px «одного числа» ({font:.1f} px кегля) шире колонки {column:.0f} px: "
                    "заголовок снова рвётся на три строки",
                )

    def test_title_stays_large_on_desktop(self) -> None:
        # Подгонка под колонку не должна превратить заголовок в обычный: на широком экране
        # кегль заметно крупнее подзаголовка (32 px).
        self.assertGreaterEqual(self._font(1440), 60)

    def test_hero_columns_share_the_left_edge_below_desktop(self) -> None:
        # Без align-items: stretch сетка обложки центрирует блоки по ширине содержимого,
        # и на планшете у заголовка, графика и поиска разные левые края.
        block = re.search(r"@media \(max-width: 959px\) \{(.*?)\n\}", self.landing_css, re.S).group(1)
        grid_rule = re.search(r"\.hero-grid \{([^}]*)\}", block).group(1)
        self.assertRegex(grid_rule, r"align-items:\s*stretch")


class HeroMotionTest(unittest.TestCase):
    """Движение обложки: «дыхание» медианы со свечением не бесконечное."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.css = _read("site/landing.css")

    def test_median_breathing_repeats_a_few_times_and_stops(self) -> None:
        # Анимация прозрачности дочернего элемента SVG под фильтром (свечение) не композитится:
        # бесконечная заставляла браузер перерисовывать график каждый кадр, пока открыта страница.
        rule = re.search(r"\.hero-median \{([^}]*)\}", self.css).group(1)
        animation = re.search(r"animation:([^;]+);", rule).group(1)
        breathe = [part for part in animation.split(",") if "breathe" in part]
        self.assertEqual(len(breathe), 1, "у медианы пропало «дыхание»")
        self.assertNotIn("infinite", animation)
        self.assertRegex(breathe[0].strip(), r"\s\d+$", "число повторов должно стоять последним словом правила")
        self.assertNotIn("infinite", self.css)

    def test_breathing_starts_and_ends_at_full_opacity(self) -> None:
        # Конечная анимация возвращает элемент к обычному состоянию: кадры 0% и 100% — полная
        # непрозрачность, иначе после последнего повтора медиана «подпрыгнула» бы.
        frames = re.search(r"@keyframes breathe \{(.*?)\n\}", self.css, re.S).group(1)
        edge = re.search(r"0%, 100% \{\s*opacity:\s*([\d.]+);", frames).group(1)
        self.assertEqual(float(edge), 1.0)

    def test_reduced_motion_still_switches_the_animation_off(self) -> None:
        block = re.search(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n\}\n", self.css, re.S).group(1)
        self.assertRegex(block, r"\.hero-median\s*\{\s*animation:\s*none")


class StylesheetCommentsTest(unittest.TestCase):
    def test_no_page_numbers_in_stylesheets(self) -> None:
        # Число данных не пишется ни в стиле, ни в комментарии к нему: размер, подогнанный под
        # «3,8–4,5%», ломается на другом диапазоне, а комментарий с числом устаревает вместе с данными.
        page = _read("index.html")
        number = r"\d+(?:\u00a0\d{3})*(?:,\d+)?"
        shown = re.findall(r'<p class="stat-number">(.*?)</p>', page) + re.findall(
            r'<span class="stat-note stat-note-full">(.*?)</span>', page, re.S)
        tokens = {t for text in shown for t in re.findall(number, text)
                  if "," in t or "\u00a0" in t or len(re.sub(r"\D", "", t)) >= 4}
        self.assertGreater(len(tokens), 5, "проверка потеряла числа страницы")
        for name in ("site/site.css", "site/landing.css", "demo/demo.css"):
            css = _read(name)
            for token in sorted(tokens):
                for variant in {token, token.replace(",", "."), token.replace("\u00a0", " ")}:
                    with self.subTest(file=name, number=variant):
                        # «4,5:1» — отношение контраста из требований доступности, а не число страницы.
                        self.assertIsNone(re.search(r"(?<![\w.,-])" + re.escape(variant) + r"(?![\w.,]|:\d)", css),
                                          f"в {name} встречается число данных {variant!r}")

    def test_card_numbers_are_sized_by_the_card_width(self) -> None:
        # Правило размера описано словами: число занимает долю ширины карточки (cqw) в пределах
        # от читаемого минимума до размера на широкой карточке.
        css = _read("site/landing.css")
        block = re.search(r"@supports \(width: 1cqw\) \{(.*?)\n\}", css, re.S).group(1)
        self.assertRegex(block, r"font-size:\s*clamp\([^)]*\d+cqw[^)]*\)")
        comment = css[: css.index("@supports (width: 1cqw)")].rsplit("/*", 1)[1]
        self.assertIn("доля ширины карточки", comment)


class MaterialsTemplateTest(unittest.TestCase):
    def test_pdf_link_does_not_wrap_away_from_its_conjunction(self) -> None:
        # «HTML и» на конце строки и одинокий «PDF» на следующей. Неразрывный пробел не
        # помогает: ссылка «PDF» — inline-block со своими отступами, и браузер переносит
        # строку перед ней. «HTML и PDF» — в одном nowrap-блоке.
        template = _read("site/index.template.html")
        notes = re.findall(r'<p class="material-note">([^<]*<span class="material-pair">.*?)</p>', template)
        self.assertEqual(len(notes), 2, "ожидаются карточки отчёта и слайдов")
        for note in notes:
            with self.subTest(note=note):
                self.assertRegex(note, r'<span class="material-pair">HTML и <a [^>]*>PDF</a></span>$')
        rule = re.search(r"\.material-pair \{([^}]*)\}", _read("site/landing.css")).group(1)
        self.assertRegex(rule, r"white-space:\s*nowrap")


class TeaserBreakLabelsTest(unittest.TestCase):
    def test_labels_cover_neighbour_lines_and_stay_under_the_series(self) -> None:
        css = _read("site/landing.css")
        label = re.search(r"\.vline-break > span \{([^}]*)\}", css).group(1)
        self.assertRegex(label, r"background:\s*var\(--night-surface\)")
        label_z = int(re.search(r"z-index:\s*(\d+)", label).group(1))
        svg_z = int(re.search(r"\.stand-plot \.plot-svg \{[^}]*z-index:\s*(\d+)", css).group(1))
        self.assertGreater(label_z, 0, "подпись должна лежать над линиями изломов")
        self.assertGreater(svg_z, label_z, "линия ряда должна лежать над подписью")


class DemoChartMarkersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _read("demo/linechart.js")

    def test_all_marker_lines_are_drawn_before_any_label(self) -> None:
        # Линия соседней отметки, нарисованная после подписи, перечёркивала бы её.
        line = self.js.index("class: `chart-marker chart-marker-")
        label = self.js.index("class: `chart-marker-label chart-marker-label-")
        self.assertLess(line, label)

    def test_labels_have_a_halo_in_the_card_colour(self) -> None:
        css = _read("demo/demo.css")
        rule = re.search(r"\.chart-marker-label \{([^}]*)\}", css).group(1)
        self.assertRegex(rule, r"paint-order:\s*stroke")
        self.assertRegex(rule, r"stroke:\s*var\(--surface\)")

    @unittest.skipUnless(shutil.which("node"), "для раскладки подписей нужен node")
    def test_label_rows_do_not_share_a_row_when_labels_overlap(self) -> None:
        script = (
            "const vm = require('vm'), fs = require('fs');"
            "const lib = vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8') + '\\nLineChart;', {});"
            "const cases = JSON.parse(process.argv[2]);"
            "console.log(JSON.stringify(cases.map(c => lib.assignMarkerRows(c.boxes, c.rows, c.gap))));"
        )

        def box(left: float, right: float) -> dict:
            return {"left": left, "right": right}

        cases = [
            # далеко друг от друга — одна строка
            {"boxes": [box(0, 50), box(100, 150), box(200, 250)], "rows": 3, "gap": 4},
            # пересекаются — разные строки; третья свободна на первой строке (первая подпись кончилась)
            {"boxes": [box(0, 50), box(40, 90), box(60, 110)], "rows": 3, "gap": 4},
            # зазор меньше gap считается задевающим
            {"boxes": [box(0, 50), box(52, 100)], "rows": 3, "gap": 4},
            # зазор ровно gap — не задевает
            {"boxes": [box(0, 50), box(54, 100)], "rows": 3, "gap": 4},
            # строк не хватает — последняя подпись остаётся в последней строке
            {"boxes": [box(0, 50), box(10, 60), box(20, 70), box(30, 80)], "rows": 3, "gap": 4},
            # пусто
            {"boxes": [], "rows": 3, "gap": 4},
        ]
        result = subprocess.run(
            ["node", "-e", script, str(ROOT / "demo" / "linechart.js"), json.dumps(cases)],
            capture_output=True, text=True, check=True, timeout=60,
        )
        self.assertEqual(
            json.loads(result.stdout),
            [[0, 0, 0], [0, 1, 0], [0, 1], [0, 0], [0, 1, 2, 2], []],
        )


if __name__ == "__main__":
    unittest.main()
