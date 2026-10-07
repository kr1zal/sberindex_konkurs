"""Иконка сайта: `favicon.svg` — знак «Одно число», `favicon.ico` из него (`scripts/make_favicon.py`).

У сайта не было иконки вкладки (`href="data:,"`), а отчёт и слайды, не зная о ней, просят `/favicon.ico`
и получали 404 — единственную запись об ошибке в консоли всего обхода. Файл в корне закрывает этот запрос.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("make_favicon", ROOT / "scripts" / "make_favicon.py")
HAS_PILLOW = importlib.util.find_spec("PIL") is not None
make_favicon = None
if HAS_PILLOW:
    make_favicon = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(make_favicon)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class FaviconSvgTest(unittest.TestCase):
    def test_svg_draws_the_same_two_curves_as_the_brand_mark_in_the_header(self) -> None:
        # Знак в иконке — тот же, что у названия сайта в шапке: геометрия двух линий совпадает до символа.
        svg = _read("favicon.svg")
        template = _read("site/index.template.html")
        for part in ("mark-a", "mark-b"):
            with self.subTest(part=part):
                header = re.search(rf'<path class="{part}" d="([^"]+)"', template).group(1)
                self.assertIn(f'd="{header}"', svg)
        self.assertTrue(svg.lstrip().startswith("<svg"))
        self.assertIn('xmlns="http://www.w3.org/2000/svg"', svg)
        # Без внешних ресурсов: ни ссылок, ни шрифтов, ни встроенных скриптов.
        self.assertNotRegex(svg, r"(?i)https?://(?!www\.w3\.org/2000/svg)|<script|href=|<image|@import")

    def test_both_pages_link_the_icon_and_the_root_file_answers_the_frozen_pages(self) -> None:
        self.assertIn('<link rel="icon" href="favicon.svg" type="image/svg+xml">', _read("index.html"))
        self.assertIn('<link rel="icon" href="../favicon.svg" type="image/svg+xml">', _read("demo/index.html"))
        # Отчёт и слайды запрашивают /favicon.ico сами — при раздаче корня репозитория файл там есть.
        self.assertTrue((ROOT / "favicon.ico").is_file())
        self.assertEqual((ROOT / "favicon.ico").read_bytes()[:4], b"\x00\x00\x01\x00", "не формат ICO")


@unittest.skipUnless(HAS_PILLOW, "для иконки нужен Pillow")
class FaviconIcoTest(unittest.TestCase):
    def test_path_parser_knows_move_cubic_and_smooth_commands(self) -> None:
        curves = make_favicon.parse_path("M2 18 C7 14 10 20 14 13 S21 9 24 5")
        self.assertEqual(curves[0], [(2.0, 18.0), (7.0, 14.0), (10.0, 20.0), (14.0, 13.0)])
        # S отражает вторую опорную точку предыдущей кривой относительно её конца: (2·14−10, 2·13−20).
        self.assertEqual(curves[1], [(14.0, 13.0), (18.0, 6.0), (21.0, 9.0), (24.0, 5.0)])
        with self.assertRaises(ValueError):
            make_favicon.parse_path("M0 0 L5 5")

    def test_ico_holds_three_sizes_and_the_mark_is_drawn(self) -> None:
        from PIL import Image

        icon = Image.open(ROOT / "favicon.ico")
        self.assertEqual(icon.format, "ICO")
        self.assertEqual(sorted(icon.info["sizes"]), [(16, 16), (32, 32), (48, 48)])
        for size in (16, 32, 48):
            with self.subTest(size=size):
                icon.size = (size, size)
                frame = icon.copy().convert("RGBA")
                self.assertEqual(frame.getpixel((0, 0))[3], 0, "углы скруглены: левый верхний пиксель прозрачен")
                colors = {frame.getpixel((x, y))[:3] for x in range(size) for y in range(size)
                          if frame.getpixel((x, y))[3] == 255}
                # Есть и тёмный фон, и янтарная линия: знак не пустой квадрат.
                self.assertTrue(any(r > 200 and 140 < g < 200 and b < 110 for r, g, b in colors), "нет янтарной линии")
                self.assertTrue(any(r < 20 and g < 25 and b < 45 for r, g, b in colors), "нет тёмного фона")

    def test_ico_is_reproduced_from_the_svg_by_the_script(self) -> None:
        # Описание воспроизведения — DEVELOPMENT.md: тот же вызов даёт те же пиксели. Сравниваются пиксели
        # каждого размера, а не байты файла: упаковка PNG внутри ICO зависит от версии Pillow.
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            rebuilt = Path(folder) / "favicon.ico"
            with contextlib.redirect_stdout(io.StringIO()):
                code = make_favicon.main(["--svg", str(ROOT / "favicon.svg"), "--out", str(rebuilt)])
            self.assertEqual(code, 0)
            committed, fresh = Image.open(ROOT / "favicon.ico"), Image.open(rebuilt)
            self.assertEqual(committed.info["sizes"], fresh.info["sizes"])
            for size in (16, 32, 48):
                with self.subTest(size=size):
                    committed.size = fresh.size = (size, size)
                    self.assertEqual(committed.copy().convert("RGBA").tobytes(), fresh.copy().convert("RGBA").tobytes(),
                                     "favicon.ico не совпал с пересборкой: запустите scripts/make_favicon.py")


if __name__ == "__main__":
    unittest.main()
