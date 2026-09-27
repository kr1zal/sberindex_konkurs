"""Замыкание зависимостей `requirements.txt` из окружения: `scripts/freeze_requirements.py`.

**Зачем.** `pip freeze` в этом `.venv` видит не только основной стек, но и всё, что
когда-либо ставилось рядом для локальных проверок — в частности Moirai (`uni2ts`,
`gluonts`, `einops`, `jaxtyping`, `hydra-core`, `lightning`, `jax` и их собственные
зависимости): `requirements.txt` намеренно их не перечисляет и ставит отдельной
командой (см. комментарий там). `requirements.lock.txt` должен фиксировать версии
только того, что действительно нужно `requirements.txt`, — транзитивно, с учётом
экстр (`timesfm[torch]`), — а не всё, что попутно стоит в окружении.

**Как считает.** Читает `requirements.txt` как корни: имя пакета и запрошенные
экстры, без версии — версия берётся из того, что установлено. Обходит граф
зависимостей по метаданным installed-дистрибутивов (`importlib.metadata`, поле
`Requires-Dist`), считая каждую пару (пакет, экстра) отдельным узлом: строка
`torch ; extra == "torch"` в METADATA `timesfm` активна только для узла
`(timesfm, "torch")`, а `jax[cuda] ; extra == "flax"` того же `timesfm` — нет,
раз `requirements.txt` просит только `timesfm[torch]`. Маркеры окружения
(`packaging.markers`) считаются для интерпретатора, который запустил скрипт.
Пакет из стека Moirai в замыкание не попадёт, если до него нет пути из корней
`requirements.txt` ни при одной активной экстре; тот же пакет, нужный и основному
стеку (через другую библиотеку), останется — фильтр по достижимости, а не по имени.

**Что пишет.** `requirements.lock.txt` рядом: заголовок (интерпретатор, платформа,
эта команда, что Moirai ставится отдельно) и `name==version` по каждому достижимому
пакету, отсортировано без учёта регистра; версия — из метаданных этого `.venv`.
Требование, для которого в `.venv` ничего не установлено (например, экстра, для
которой библиотеку так и не поставили), в файл не попадает и печатается отдельно —
как отсутствующее, а не молча опускается.

    .venv/bin/python -u scripts/freeze_requirements.py
"""
from __future__ import annotations

import platform
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock.txt"
COMMAND = ".venv/bin/python -u scripts/freeze_requirements.py"


def read_roots(path: Path) -> list[Requirement]:
    """Требования верхнего уровня: имя и экстры из `requirements.txt`, без версии —
    версии в замыкании берутся из установленного, а не из спецификаторов файла.
    Пустые строки и комментарии пропускаются, как их пропускает и pip."""
    roots = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            roots.append(Requirement(line))
    return roots


def index_installed() -> dict[str, metadata.Distribution]:
    """Установленные дистрибутивы этого интерпретатора по каноническому имени
    (PEP 503): написание в `requirements.txt` и в METADATA зависимостей может
    разойтись регистром или дефисом/подчёркиванием (`huggingface_hub` /
    `huggingface-hub`)."""
    by_name: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            by_name[canonicalize_name(name)] = dist
    return by_name


def resolve(
    roots: list[Requirement], installed: dict[str, metadata.Distribution],
) -> tuple[dict[str, metadata.Distribution], list[str]]:
    """Обходит граф зависимостей от корней `requirements.txt`. Каждый узел — пара
    (каноническое имя, экстра или None для базовой установки без экстр). Возвращает
    достижимые дистрибутивы по каноническому имени и отсортированный список
    требований, для которых в `.venv` ничего не установлено."""
    visited: set[tuple[str, str | None]] = set()
    reachable: dict[str, metadata.Distribution] = {}
    missing: set[str] = set()

    def seed(req: Requirement) -> list[tuple[str, str | None]]:
        key = canonicalize_name(req.name)
        return [(key, None)] + [(key, extra) for extra in sorted(req.extras)]

    stack: list[tuple[str, str | None]] = [node for req in roots for node in seed(req)]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        name, extra = node
        dist = installed.get(name)
        if dist is None:
            missing.add(f"{name}[{extra}]" if extra else name)
            continue
        reachable[name] = dist

        for raw in dist.requires or []:
            dep = Requirement(raw)
            if dep.marker is not None and not dep.marker.evaluate({"extra": extra or ""}):
                continue
            stack.extend(seed(dep))

    return reachable, sorted(missing)


def render_header(count: int, excluded: list[str]) -> str:
    interpreter = f"Python {platform.python_version()}"
    system = "macOS" if platform.system() == "Darwin" else platform.system()
    return (
        f"# Проверено на {interpreter}, {system} {platform.machine()}.\n"
        f"# Получен командой (пишет файл сам, обычной установки в этот .venv не требует):\n"
        f"#\n"
        f"#   {COMMAND}\n"
        f"#\n"
        f"# Замыкание зависимостей requirements.txt (с учётом экстр, например\n"
        f"# timesfm[torch]) по метаданным .venv, в котором собран этот файл — без стека\n"
        f"# Moirai (uni2ts, gluonts, einops, jaxtyping, hydra-core, lightning, jax и того,\n"
        f"# что нужно только им): Moirai ставится отдельно, см. requirements.txt. Пакет\n"
        f"# из этого стека, нужный и основному стеку тоже, в списке остаётся — фильтр\n"
        f"# по достижимости из requirements.txt, а не по имени.\n"
        f"#\n"
        f"# Пакетов в замыкании: {count}. Установлено в .venv, но не в замыкании:\n"
        f"# {len(excluded)} (Moirai и то, что стоит рядом для локальных проверок:\n"
        f"# Jupyter, тесты и прочие пакеты, которых requirements.txt не касается).\n"
    )


def main() -> int:
    roots = read_roots(REQUIREMENTS)
    installed = index_installed()
    reachable, missing = resolve(roots, installed)

    root_names = {canonicalize_name(req.name) for req in roots}
    absent_roots = sorted(root_names - set(reachable))
    if absent_roots:
        print(f"ОШИБКА: корни requirements.txt не установлены в .venv: {', '.join(absent_roots)}",
              file=sys.stderr)
        return 1

    lines = sorted(reachable.values(), key=lambda d: canonicalize_name(d.metadata["Name"]))
    excluded = sorted(
        canonicalize_name(d.metadata["Name"]) for d in installed.values()
        if canonicalize_name(d.metadata["Name"]) not in reachable
    )

    header = render_header(len(lines), excluded)
    body = "\n".join(f"{d.metadata['Name']}=={d.version}" for d in lines)
    LOCK.write_text(header + "\n" + body + "\n", encoding="utf-8")

    print(f"{LOCK.relative_to(ROOT)}: {len(lines)} пакетов (из {len(installed)} установленных)")
    if missing:
        print(f"в requirements.txt объявлено, но не установлено: {', '.join(missing)}")
    print(f"не в замыкании (осталось за бортом): {len(excluded)}")
    print(f"  {', '.join(excluded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
