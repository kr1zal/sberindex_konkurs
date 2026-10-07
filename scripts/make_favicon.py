"""Иконка сайта для старых браузеров: `favicon.ico` из `favicon.svg`.

    .venv/bin/python scripts/make_favicon.py [--svg ФАЙЛ] [--out ФАЙЛ]

`favicon.svg` — единственный источник геометрии и цветов: две линии знака «Одно число» (те же кривые, что
в шапке страниц) на тёмном скруглённом квадрате. Скрипт читает из него размер холста, фон, скругление,
смещение группы и пути, рисует их с большим запасом по разрешению и сжимает до 16, 32 и 48 пикселей. Формат
ICO нужен браузерам без поддержки SVG-иконок и запросу `/favicon.ico`, который отчёт и слайды шлют сами,
не зная об иконке сайта. Pillow — из `requirements.lock`; файл воспроизводится этим же вызовом, правка знака
делается в SVG, а `.ico` пересобирается.

Разбираются только команды, которые есть в знаке: `M`, `C` и `S` с абсолютными координатами. Сам SVG читается
по атрибутам тегов, а не XML-парсером: источник — наш собственный маленький файл, и внешним сущностям XML
в нём взяться неоткуда.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

# Размеры в файле: вкладка и закладка (16), адресная строка и панель задач (32), крупные значки (48).
ICO_SIZES = [(16, 16), (32, 32), (48, 48)]

# Во сколько раз холст рисуется крупнее самого большого размера: сжатие сглаживает края линий.
SUPERSAMPLE = 8
# Точек на одну кубическую кривую: на полотне в несколько сотен пикселей ломаной уже не видно.
CURVE_STEPS = 160


def parse_path(d: str) -> list[list[tuple[float, float]]]:
    """Путь SVG из команд `M`, `C`, `S` (абсолютные) → список кубических кривых [p0, p1, p2, p3]."""
    tokens = re.findall(r"[A-Za-z]|-?\d+(?:\.\d+)?", d)
    curves: list[list[tuple[float, float]]] = []
    current = (0.0, 0.0)
    last_control: tuple[float, float] | None = None
    i = 0
    command = ""

    def take(n: int) -> list[float]:
        nonlocal i
        values = [float(t) for t in tokens[i:i + n]]
        i += n
        return values

    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
        if command == "M":
            current = tuple(take(2))
            last_control = None
        elif command == "C":
            x1, y1, x2, y2, x, y = take(6)
            curves.append([current, (x1, y1), (x2, y2), (x, y)])
            current, last_control = (x, y), (x2, y2)
        elif command == "S":
            x2, y2, x, y = take(4)
            # Первая опорная точка — отражение второй опорной предыдущей кривой относительно её конца.
            first = (2 * current[0] - last_control[0], 2 * current[1] - last_control[1]) if last_control else current
            curves.append([current, first, (x2, y2), (x, y)])
            current, last_control = (x, y), (x2, y2)
        else:
            raise ValueError(f"команда пути {command!r} не поддерживается: {d!r}")
    return curves


def sample(curves: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Точки вдоль пути — кубические кривые Безье, по `CURVE_STEPS` на каждую."""
    points: list[tuple[float, float]] = []
    for p0, p1, p2, p3 in curves:
        for step in range(CURVE_STEPS + 1):
            t = step / CURVE_STEPS
            u = 1 - t
            points.append((
                u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
                u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1],
            ))
    return points


def tag_attributes(svg: str, tag: str) -> list[dict[str, str]]:
    """Атрибуты каждого тега `<tag …>` файла, по порядку: {имя: значение}."""
    return [dict(re.findall(r'([\w:-]+)="([^"]*)"', body)) for body in re.findall(rf"<{tag}\b([^>]*)>", svg)]


def render(svg_path: Path) -> Image.Image:
    """Знак из `favicon.svg` крупным RGBA-холстом: фон со скруглением и линии с круглыми концами."""
    svg = svg_path.read_text(encoding="utf-8")
    width, height = (float(v) for v in tag_attributes(svg, "svg")[0]["viewBox"].split()[2:])
    scale = SUPERSAMPLE * max(size for size, _ in ICO_SIZES) / width
    canvas = Image.new("RGBA", (round(width * scale), round(height * scale)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    rect = tag_attributes(svg, "rect")[0]
    draw.rounded_rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                           radius=float(rect.get("rx", 0)) * scale, fill=rect["fill"])

    group = tag_attributes(svg, "g")[0]
    dx, dy = (float(v) for v in re.match(r"translate\(([-\d.]+)[ ,]+([-\d.]+)\)", group["transform"]).groups())
    for path in tag_attributes(svg, "path"):
        points = [((x + dx) * scale, (y + dy) * scale) for x, y in sample(parse_path(path["d"]))]
        stroke = float(path["stroke-width"]) * scale
        color = path["stroke"]
        draw.line(points, fill=color, width=round(stroke), joint="curve")
        radius = stroke / 2
        for x, y in (points[0], points[-1]):  # круглые концы: stroke-linecap="round"
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="favicon.ico из favicon.svg")
    parser.add_argument("--svg", type=Path, default=ROOT / "favicon.svg", help="исходник знака")
    parser.add_argument("--out", type=Path, default=ROOT / "favicon.ico", help="куда писать .ico")
    args = parser.parse_args(argv)
    canvas = render(args.svg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, format="ICO", sizes=ICO_SIZES)
    print(f"написано: {args.out} — размеры {', '.join(f'{w}×{h}' for w, h in ICO_SIZES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
