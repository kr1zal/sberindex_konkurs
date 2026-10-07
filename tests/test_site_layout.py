"""Вёрстка главной и страницы прогноза: то, что выяснилось только в браузере и держится на CSS и скриптах.

Сборка не нужна: читаются сами файлы `site/`, `demo/` и шаблон. Проверки — о формулах и
правилах, а не о пикселях: кегль заголовка обложки и высоты первого экрана считаются по формулам из стилей
на каждой ширине и высоте окна; раскладка подписей отметок графика страницы прогноза — чистая функция
`LineChart.assignMarkerRows`, её запускает node (без него проверка пропускается). Что весь первый экран,
включая полосу чисел, видно без прокрутки, подтверждает браузер; здесь — что формулы этого не нарушают.
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


def _css_length(expr: str, viewport: float, height: float = 0.0) -> float:
    """Значение выражения из стилей (px, rem, vw, vh, calc, clamp, min, max) в пикселях; `height` — высота окна
    для vh, `viewport` — ширина."""
    text = expr.strip()
    text = re.sub(r"(\d+(?:\.\d+)?)vw", lambda m: f"({m.group(1)}*{viewport}/100)", text)
    text = re.sub(r"(\d+(?:\.\d+)?)vh", lambda m: f"({m.group(1)}*{height}/100)", text)
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


# Окна, в которых весь первый экран — шапка, заголовок с кнопками, график и полоса чисел — должен быть виден без
# прокрутки: ширина, высота.
FIRST_SCREENS = [(1366, 768), (1280, 720), (1440, 900), (1536, 864)]


class HeroTitleTest(unittest.TestCase):
    """Заголовок обложки: «одного числа» целиком в строке на любой ширине и высоте окна — две строки,
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

    def _font(self, viewport: int, height: float = 5000.0) -> float:
        """Кегль заголовка; без высоты окна — «обычный», когда доля высоты (vh) не ограничивает."""
        expr = [e for width, e in _hero_title_font_sizes(self.landing_css) if width <= viewport][-1]
        return _css_length(expr, viewport, height)

    def test_amber_phrase_fits_the_column_at_every_width_and_height(self) -> None:
        too_wide = []
        for viewport in list(range(320, 640, 10)) + list(range(640, 2561, 20)) + [959, 960, 1279, 1280, 1304]:
            for height in (480, 600, 720, 768, 864, 900, 1080, 1440):
                font, column = self._font(viewport, height), self._column(viewport)
                if font * AMBER_PHRASE_EM > column:
                    too_wide.append((viewport, height, round(font, 1), round(column)))
        self.assertEqual(too_wide, [], "«одного числа» шире колонки: заголовок снова рвётся на три строки")

    def test_title_stays_large_in_the_four_laptop_windows(self) -> None:
        # Подгонка под колонку и под высоту окна не должна превратить заголовок в обычный: в окнах ноутбуков кегль
        # заметно крупнее подзаголовка (24 px).
        for width, height in FIRST_SCREENS:
            with self.subTest(window=f"{width}x{height}"):
                self.assertGreaterEqual(self._font(width, height), 60)

    def test_height_only_shrinks_the_title(self) -> None:
        # На невысоких окнах (ноутбуки) заголовок мельче, чтобы заголовок, кнопки и полоса чисел помещались без
        # прокрутки; доля высоты окна только уменьшает кегль, поэтому «одного числа» в колонку по-прежнему влезает.
        found = re.search(r"@media \(min-width: 960px\) \{\s*\.hero-title \{\s*font-size:\s*([^;]+);", self.landing_css)
        self.assertIsNotNone(found, "у заголовка на широком окне нет своего правила")
        self.assertIn("vh", found.group(1), "кегль заголовка не зависит от высоты окна")
        for viewport in range(960, 2561, 20):
            usual = self._font(viewport)
            for height in (480, 600, 720, 768, 864, 900, 1080, 1440):
                with self.subTest(viewport=viewport, height=height):
                    self.assertLessEqual(self._font(viewport, height), usual)

    def test_hero_columns_share_the_left_edge_below_desktop(self) -> None:
        # Без align-items: stretch сетка обложки центрирует блоки по ширине содержимого,
        # и на планшете у заголовка и графика разные левые края.
        block = re.search(r"@media \(max-width: 959px\) \{(.*?)\n\}", self.landing_css, re.S).group(1)
        grid_rule = re.search(r"\.hero-grid \{([^}]*)\}", block).group(1)
        self.assertRegex(grid_rule, r"align-items:\s*stretch")


class FirstScreenTest(unittest.TestCase):
    """Весь первый экран — шапка, заголовок, две кнопки, график и полоса из четырёх чисел — помещается в окно без
    прокрутки в четырёх окнах ноутбуков (`FIRST_SCREENS`). Браузер подтверждает это измерением; здесь — что
    формулы стилей этого не нарушают: сумма высот по ним, с запасом на два перенесённых строки подписи и оговорки в
    каждой ячейке, не больше высоты окна."""

    SLACK = 40  # запас на разницу модели и браузера, px

    @classmethod
    def setUpClass(cls) -> None:
        cls.css = _read("site/landing.css")
        cls.site = _read("site/site.css")

    def _value(self, pattern: str, source: str | None = None) -> str:
        found = re.search(pattern, source or self.css, re.S)
        self.assertIsNotNone(found, f"в стилях не нашлось: {pattern}")
        return found.group(1).strip()

    def _header(self) -> float:
        """Высота шапки: поля сверху и снизу и самый высокий элемент строки (кнопка меню и ссылки — 44 px)."""
        padding = float(self._value(r"\.site-header-inner \{[^}]*?padding-block:\s*(\d+)px", self.site))
        row = float(self._value(r"\.nav-link \{[^}]*?min-height:\s*(\d+)px", self.site))
        return 2 * padding + row

    def _budget(self, width: int, height: int) -> float:
        """Сколько пикселей по формулам занимают шапка, обложка и полоса в окне `width`×`height` — с двумя
        строками в подписи и в оговорке каждого числа."""
        def length(expr: str, cq: float = 0.0) -> float:
            expr = re.sub(r"(\d+(?:\.\d+)?)cqw", lambda m: f"({m.group(1)}*{cq}/100)", expr)
            return _css_length(expr, width, height)

        wide = r"@media \(min-width: 960px\) \{\s*"
        grid_top, grid_bottom = re.match(
            r"(clamp\([^)]*\))\s+(clamp\([^)]*\))", self._value(r"\.hero-grid \{[^}]*?padding-block:\s*([^;]+);")).groups()
        title = _css_length(self._value(wide + r"\.hero-title \{\s*font-size:\s*([^;]+);"), width, height)
        sub_margin, sub_font = (
            length(self._value(wide + r"\.hero-sub \{\s*margin-top:\s*([^;]+);")),
            length(self._value(wide + r"\.hero-sub \{\s*margin-top:[^;]+;\s*font-size:\s*([^;]+);")))
        actions_margin = length(self._value(wide + r"\.hero-actions \{\s*margin-top:\s*([^;]+);"))
        button = length(self._value(wide + r"\.hero-actions \{[^}]*\}\s*\.hero-actions \.btn \{\s*min-height:\s*([^;]+);"))
        eyebrow = 13 * 1.4 + float(self._value(r"\.hero-eyebrow \{\s*margin:\s*0 0 (\d+)px;"))
        # Левая колонка: метка, заголовок в две строки, подзаголовок в две строки, кнопки.
        left = eyebrow + 2 * title * 1.02 + sub_margin + 2 * sub_font * 1.2 + actions_margin + button
        plot = length(self._value(r"\.hero-plot \{[^}]*?height:\s*([^;]+);"))
        grid = length(grid_top) + max(left, plot) + length(grid_bottom)

        top, side, bottom = re.match(
            r"(clamp\([^)]*\))\s+(\d+)px\s+(clamp\([^)]*\))", self._value(r"\.stat \{[^}]*?padding:\s*([^;]+);")).groups()
        # Ширина содержимого ячейки: четыре равные доли обёртки без боковых полей ячеек (у крайних — по одному).
        cell = (min(width, 1240) - 2 * 32) / 4 - 2 * float(side)
        number = length(self._value(r"@supports \(width: 1cqw\) \{\s*\.stat-number \{\s*font-size:\s*([^;]+);"), cell)
        label_margin = length(self._value(r"\.stat-label \{[^}]*?margin-top:\s*([^;]+);"))
        label_font = length(self._value(r"\.stat-label \{[^}]*?font-size:\s*([^;]+);"))
        note_font = length(self._value(r"\.stat-note \{[^}]*?font-size:\s*([^;]+);"))
        strip = (1 + length(top) + number * 1.0 + label_margin + 2 * label_font * 1.35 + 4 + 2 * note_font * 1.35
                 + length(bottom))
        return self._header() + grid + strip

    def test_header_height_is_the_one_the_cover_subtracts_from_the_window(self) -> None:
        # Обложка занимает окно за вычетом шапки: число в стилях обложки — высота шапки из общей темы.
        self.assertEqual(float(self._value(r"\.hero \{[^}]*?--header-h:\s*(\d+)px")), self._header())
        self.assertRegex(self.css, r"min-height:\s*calc\(100vh - var\(--header-h\)\);")
        self.assertRegex(self.css, r"min-height:\s*calc\(100svh - var\(--header-h\)\);")

    def test_cover_and_strip_fit_the_window_in_the_four_laptop_sizes(self) -> None:
        for width, height in FIRST_SCREENS:
            with self.subTest(window=f"{width}x{height}"):
                need = self._budget(width, height)
                self.assertLessEqual(need + self.SLACK, height,
                                     f"по формулам шапка, обложка и полоса занимают {need:.0f} px из {height}")

    def test_strip_is_pinned_to_the_bottom_of_the_cover(self) -> None:
        # Обложка — колонка во всю высоту окна, сетка с заголовком и графиком растёт (flex), полоса — после неё:
        # на высоком окне полоса стоит у нижнего края, а не сразу под кнопками.
        hero = self._value(r"\n\.hero \{([^}]*)\}")
        self.assertRegex(hero, r"display:\s*flex")
        self.assertRegex(hero, r"flex-direction:\s*column")
        self.assertRegex(self._value(r"\n\.hero-grid \{([^}]*)\}"), r"flex:\s*1 1 auto")

    def test_numbers_are_sized_by_the_window_height_and_never_wider_than_their_cell(self) -> None:
        # Размер числа — доля высоты окна, но не больше доли ширины ячейки (cqw): диапазон двух процентов через тире
        # длиннее одного значения и в узкой ячейке не должен вылезти за край или перенестись по тире.
        block = self._value(r"@supports \(width: 1cqw\) \{(.*?)\n\}")
        self.assertRegex(block, r"font-size:\s*clamp\([^;]*min\(\d+(?:\.\d+)?vh,\s*\d+(?:\.\d+)?cqw\)[^;]*\);")
        comment = self.css[: self.css.index("@supports (width: 1cqw)")].rsplit("/*", 1)[1]
        self.assertIn("доля высоты окна", comment)
        self.assertIn("ширины ячейки", comment)
        self.assertRegex(self._value(r"\.stat-number \{([^}]*)\}"), r"white-space:\s*nowrap")

    def test_first_number_is_amber_and_the_strip_uses_the_night_tokens(self) -> None:
        self.assertRegex(self._value(r"\.stat-lead \.stat-number \{([^}]*)\}"), r"color:\s*var\(--amber\)")
        # Цвета полосы — токены ночного блока (обложка — .night в обеих темах), своих цветов в правилах нет.
        for selector in (r"\.stats", r"\.stat", r"\.stat-label", r"\.stat-note"):
            with self.subTest(selector=selector):
                self.assertNotRegex(self._value(rf"\n{selector} \{{([^}}]*)\}}"), r"#[0-9a-fA-F]{3,6}\b|rgba?\(")

    def test_strip_goes_two_by_two_below_a_thousand_pixels_and_the_buttons_stack_on_a_phone(self) -> None:
        tablet = self._value(r"@media \(max-width: 999px\) \{(.*?)\n\}\n")
        self.assertRegex(self._value(r"\.stats \{([^}]*)\}", tablet), r"grid-template-columns:\s*repeat\(2,")
        self.assertRegex(self._value(r"\.stats \{[^}]*grid-template-columns:\s*repeat\((\d),"), r"4")
        phone = self._value(r"@media \(max-width: 639px\) \{\s*\.hero-grid(.*?)\n\}\n")
        self.assertRegex(phone, r"\.hero-actions \.btn \{\s*flex:\s*1 1 100%;")


class HeroMotionTest(unittest.TestCase):
    """Движение обложки: «дыхание» медианы со свечением не бесконечное; текст, кнопки и полоса проявляются быстро."""

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

    def test_cover_text_buttons_and_strip_appear_in_under_a_second(self) -> None:
        # Первая секунда страницы показывает заголовок, кнопки и числа, а не пустой тёмный экран: проявление текста
        # обложки, кнопок и полосы чисел — короче секунды. Пять блоков, пять задержек, и ни одной лишней.
        delays = {name: float(value)
                  for name, value in re.findall(r"\.(d\d) \{ animation-delay: ([\d.]+)s; \}", self.css)}
        self.assertEqual(sorted(delays), ["d1", "d2", "d3", "d4", "d5"])
        duration = float(re.search(r"\.rise \{\s*animation: rise ([\d.]+)s", self.css).group(1))
        self.assertLessEqual(max(delays.values()), 0.3)
        self.assertLess(max(delays.values()) + duration, 1.0)
        template = _read("site/index.template.html")
        used = re.findall(r'class="[^"]*\brise (d\d)\b', template)
        self.assertEqual(sorted(used), sorted(delays), "у блока обложки задержка, которой нет в стилях, или наоборот")

    def test_reduced_motion_still_switches_the_animation_off(self) -> None:
        block = re.search(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n\}\n", self.css, re.S).group(1)
        self.assertRegex(block, r"\.hero-median\s*\{\s*animation:\s*none")
        self.assertRegex(block, r"\.rise,")


class PhoneCoverTest(unittest.TestCase):
    """Обложка на телефоне: заголовок и кнопки — на первом экране, график — иллюстрация следом, числа — ниже."""

    @classmethod
    def setUpClass(cls) -> None:
        css = _read("site/landing.css")
        cls.block = re.search(r"@media \(max-width: 639px\) \{(.*?)\n\}", css, re.S).group(1)

    def test_cover_chart_is_short_so_that_the_buttons_are_not_pushed_below_the_first_screen(self) -> None:
        # Кнопки стоят до графика (порядок разметки), и график на телефоне невысокий: ни выше ста восьмидесяти
        # пикселей, ни ниже, чем он читается как линии. Он остаётся: иллюстрация обложки не прячется.
        height = re.search(r"\.hero-plot \{\s*height:\s*(\d+)px;", self.block)
        self.assertIsNotNone(height, "у обложки на телефоне нет своей высоты графика")
        self.assertLessEqual(int(height.group(1)), 200)
        self.assertGreaterEqual(int(height.group(1)), 120, "график остаётся читаемым")
        self.assertNotRegex(self.block, r"\.hero-plot\s*\{[^}]*display:\s*none")

    def test_buttons_stack_to_the_full_width(self) -> None:
        # В колонке 280 px (320 px окна) две кнопки рядом не помещаются: они друг под другом, каждая во всю ширину.
        self.assertRegex(self.block, r"\.hero-actions \.btn \{\s*flex:\s*1 1 100%;")


def _rgb(color: str) -> tuple[float, float, float]:
    """#rrggbb → компоненты 0…255."""
    value = color.strip().lstrip("#")
    return tuple(float(int(value[i:i + 2], 16)) for i in (0, 2, 4))


def _luminance(rgb: tuple[float, float, float]) -> float:
    """Относительная яркость по WCAG 2.1."""
    def channel(c: float) -> float:
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(foreground: str | tuple, background: str | tuple, alpha: float = 1.0) -> float:
    """Контраст цвета линии (с прозрачностью alpha) к фону, формула WCAG 2.1."""
    fg = _rgb(foreground) if isinstance(foreground, str) else foreground
    bg = _rgb(background) if isinstance(background, str) else background
    mixed = tuple(alpha * f + (1 - alpha) * b for f, b in zip(fg, bg))
    lighter, darker = sorted((_luminance(mixed), _luminance(bg)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _theme_tokens(css: str) -> dict[str, dict[str, str]]:
    """Токены трёх областей site.css: светлая тема (:root), тёмная (в prefers-color-scheme) и ночной блок (.night).
    Значения, ссылающиеся на другие токены через var(), раскрываются по :root."""
    blocks = {
        "light": re.search(r"\n:root \{(.*?)\n\}", css, re.S).group(1),
        "dark": re.search(r"@media \(prefers-color-scheme: dark\) \{\s*:root \{(.*?)\n  \}", css, re.S).group(1),
        "night": re.search(r"\n\.night \{(.*?)\n\}", css, re.S).group(1),
    }
    raw = {name: dict(re.findall(r"(--[\w-]+):\s*([^;]+?);", text)) for name, text in blocks.items()}

    def resolve(value: str, scope: str) -> str:
        found = re.match(r"var\((--[\w-]+)\)", value)
        if not found:
            return value.split("/*")[0].strip()
        name = found.group(1)
        return resolve(raw[scope].get(name) or raw["light"][name], scope)

    return {scope: {name: resolve(value, scope) for name, value in tokens.items()} for scope, tokens in raw.items()}


class ContrastTest(unittest.TestCase):
    """Цвета, которые несут смысл: рамка кнопки и линии графиков — не меньше 3:1 к фону (WCAG 1.4.11)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.site = _read("site/site.css")
        cls.landing = _read("site/landing.css")
        cls.tokens = _theme_tokens(cls.site)

    def surface(self, scope: str) -> dict[str, str]:
        t = self.tokens[scope]
        return {"page": t.get("--bg") or self.tokens["light"]["--bg"], "card": t["--surface"]}

    def test_step_buttons_are_visible_among_the_neighbouring_text(self) -> None:
        # Невыбранные шаги с рамкой 1,2:1 выглядели статичными карточками — читатель мог не узнать, что медиана и
        # остаток за кликом. Рамка — как у поля поиска: 3:1 к фону страницы в обеих темах.
        for scope in ("light", "dark"):
            with self.subTest(scope=scope):
                background = self.surface(scope)["page"]
                self.assertGreaterEqual(_contrast(self.tokens[scope]["--step-line"], background), 3.0)
        rule = re.search(r"\.step:hover \{([^}]*)\}", self.landing).group(1)
        self.assertIn("border-color: var(--ink)", rule)

    def test_cover_strip_and_the_second_button_are_readable_on_the_night_background(self) -> None:
        # Обложка тёмная в обеих темах: подписи и оговорки полосы чисел, янтарное главное число — не меньше 4,5:1,
        # рамка второй кнопки («Найти свой город») — граница элемента управления, не меньше 3:1.
        night = self.tokens["night"]
        background = night["--bg"]
        for name, token, minimum in (("подпись", "--ink", 4.5), ("оговорка", "--ink-soft", 4.5),
                                     ("рамка кнопки", "--field-line", 3.0)):
            with self.subTest(part=name):
                self.assertGreaterEqual(_contrast(night[token], background), minimum)
        self.assertGreaterEqual(_contrast(self.tokens["light"]["--amber"], background), 4.5)
        # Основная кнопка — янтарная с тёмным текстом ночного блока.
        self.assertGreaterEqual(_contrast(night["--btn-ink"], night["--btn-bg"]), 4.5)

    def test_step_buttons_say_what_they_do(self) -> None:
        template = _read("site/index.template.html")
        hints = re.findall(r'<span class="step-hint" aria-hidden="true">показать →</span>', template)
        self.assertEqual(len(hints), 3, "подсказка есть у каждого шага: выбранным может быть любой")
        self.assertRegex(self.landing, r'\.step\[aria-pressed="true"\] \.step-hint \{\s*display: none;')

    def test_series_lines_reach_three_to_one_at_their_opacity(self) -> None:
        opacity = float(re.search(r"\.story-lines \{\s*stroke-opacity: ([\d.]+);", self.landing).group(1))
        for scope in ("light", "dark"):
            with self.subTest(scope=scope):
                line = self.tokens[scope]["--chart-line"]
                self.assertGreaterEqual(_contrast(line, self.surface(scope)["card"], opacity), 3.0)

    def test_marks_of_the_panel_end_reach_three_to_one(self) -> None:
        # «Конец панели» на светлой и тёмной карточке и вертикаль блока на ночной подложке.
        for scope in ("light", "dark", "night"):
            with self.subTest(scope=scope):
                self.assertGreaterEqual(_contrast(self.tokens[scope]["--chart-mark"], self.surface(scope)["card"]), 3.0)
        self.assertRegex(self.landing, r"\.vline-panel \{\s*border-left: 1px dashed var\(--chart-mark\);")

    def test_method_figure_is_softened_in_the_dark_theme_and_its_caption_stays_readable(self) -> None:
        # Белая подложка схемы была самым ярким пятном тёмной страницы; смягчённая — и подпись с ссылкой на ней
        # читаются (4,5:1 и выше), на светлой теме — тоже.
        self.assertEqual(self.tokens["dark"]["--surface-card"].lower(), "#e3e6ec")
        text = re.search(r"\.method-figure figcaption \{[^}]*color:\s*(#[0-9a-fA-F]{6})", self.landing).group(1)
        link = re.search(r"\.method-figure figcaption a \{[^}]*color:\s*(#[0-9a-fA-F]{6})", self.landing).group(1)
        for scope in ("light", "dark"):
            for name, color in (("подпись", text), ("ссылка", link)):
                with self.subTest(scope=scope, part=name):
                    self.assertGreaterEqual(_contrast(color, self.tokens[scope]["--surface-card"]), 4.5)


class SkipLinkTargetTest(unittest.TestCase):
    def test_main_takes_focus_from_the_skip_link_on_both_pages(self) -> None:
        # Без tabindex="-1" фокус после «К содержанию» оставался на body: в Safari следующий Tab шёл от начала.
        for page in ("site/index.template.html", "demo/index.html"):
            with self.subTest(page=page):
                self.assertIn('<main id="main" tabindex="-1">', _read(page))
        self.assertRegex(_read("site/site.css"), r"\nmain:focus \{\s*outline: none;")


class StylesheetCommentsTest(unittest.TestCase):
    def test_no_page_numbers_in_stylesheet_comments(self) -> None:
        # В комментарии к стилю не пишется число данных: оно устаревает вместе с данными, а размер,
        # подогнанный под конкретное значение, ломается на другом. Ищутся только комментарии: значения
        # свойств («0.42», «2.2») с числами страницы случайно совпадать вправе.
        page = _read("index.html")
        number = r"\d+(?:\u00a0\d{3})*(?:,\d+)?"
        shown = re.findall(r'<p class="stat-number">(.*?)</p>', page) + re.findall(
            r'<span class="stat-(?:label|note)">(.*?)</span>', page, re.S)
        tokens = {t for text in shown for t in re.findall(number, text)
                  if "," in t or "\u00a0" in t or len(re.sub(r"\D", "", t)) >= 4}
        self.assertGreater(len(tokens), 5, "проверка потеряла числа страницы")
        for name in ("site/site.css", "site/landing.css", "demo/demo.css"):
            comments = " ".join(re.findall(r"/\*.*?\*/", _read(name), re.S))
            self.assertTrue(comments, f"в {name} нет комментариев — проверка ничего не читает")
            for token in sorted(tokens):
                for variant in {token, token.replace(",", "."), token.replace("\u00a0", " ")}:
                    with self.subTest(file=name, number=variant):
                        # «4,5:1» — отношение контраста из требований доступности, а не число страницы.
                        self.assertIsNone(
                            re.search(r"(?<![\w.,-])" + re.escape(variant) + r"(?![\w.,]|:\d)", comments),
                            f"в комментарии {name} встречается число данных {variant!r}",
                        )

    def test_strip_numbers_are_sized_by_the_window_height_and_the_cell_width(self) -> None:
        # Правило размера описано словами: число — доля высоты окна, но не больше доли ширины ячейки (cqw), в пределах
        # от читаемого минимума до размера на высоком окне.
        css = _read("site/landing.css")
        block = re.search(r"@supports \(width: 1cqw\) \{(.*?)\n\}", css, re.S).group(1)
        self.assertRegex(block, r"font-size:\s*clamp\([^;]*\d+cqw[^;]*\)")
        comment = css[: css.index("@supports (width: 1cqw)")].rsplit("/*", 1)[1]
        self.assertIn("доля высоты окна", comment)
        self.assertIn("доли ширины ячейки", comment)


class MaterialsTemplateTest(unittest.TestCase):
    def test_pdf_link_does_not_wrap_away_from_its_conjunction(self) -> None:
        # «HTML и» на конце строки и одинокий «PDF» на следующей. Неразрывный пробел не
        # помогает: ссылка «PDF» — inline-block со своими отступами, и браузер переносит
        # строку перед ней. «HTML и PDF» — в одном nowrap-блоке; размер файла и знак новой вкладки — внутри
        # ссылки, чтобы они не оторвались от «PDF».
        template = _read("site/index.template.html")
        notes = re.findall(r'<p class="material-note">([^<]*<span class="material-pair">.*?)</p>', template)
        self.assertEqual(len(notes), 2, "ожидаются карточки отчёта и слайдов")
        for note in notes:
            with self.subTest(note=note):
                self.assertRegex(
                    note,
                    r'<span class="material-pair">HTML и <a [^>]*>PDF · \$\{\w+\}'
                    r'<span class="ext" aria-hidden="true">&nbsp;↗</span><span class="visually-hidden">[^<]*</span></a></span>$')
        rule = re.search(r"\.material-pair \{([^}]*)\}", _read("site/landing.css")).group(1)
        self.assertRegex(rule, r"white-space:\s*nowrap")


class MenuBreakpointTest(unittest.TestCase):
    def test_menu_collapses_at_the_same_width_in_styles_and_both_page_scripts(self) -> None:
        # Ширину, с которой меню прячется в кнопку, задаёт стиль, а раскрывают и закрывают его скрипты главной и
        # страницы прогноза: три числа расходятся — и меню на каких-то ширинах окажется открытым без кнопки или
        # закрытым без возможности открыть. Самая длинная подпись меню — «Прогноз по муниципалитету» — требует
        # порога выше планшетного портрета: ниже строка меню не помещалась рядом с названием сайта.
        css = _read("site/site.css")
        found = re.search(r"@media \(max-width: (\d+)px\) \{\s*\.nav-toggle \{\s*display: inline-flex;", css)
        self.assertIsNotNone(found, "в стилях нет правила кнопки меню")
        width = found.group(1)
        self.assertGreaterEqual(int(width), 820)
        for name in ("site/landing.js", "demo/demo.js"):
            with self.subTest(file=name):
                self.assertIn(f'window.matchMedia("(max-width: {width}px)")', _read(name))
        self.assertRegex(css, r"\.nav-link \{[^}]*white-space:\s*nowrap")


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
