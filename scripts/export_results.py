"""Снимок крупных результатов для публикации: `scripts/export_results.py`.

**Зачем.** Конвейер (`src/run.py`, `scripts/horizons.py`, `scripts/forecast_forward.py`,
`scripts/changepoints.py`) пишет `results/*.csv` вне git — начисто на каждый прогон,
а публичный клон должен нести числа, из которых собран отчёт, целиком: прогноз
на 2025 год и раздел об обнаружении разладок — без прогона стенда. Граница не по
размеру самому по себе, а по форме: крупные построчные файлы, которые отчёт читает
(`per_series.csv`, `horizons_per_series.csv`, `forecast_2025.csv`, `cp_bench.csv`,
`cp_v1/cp_bench.csv`) — снимком `.csv.gz`, сделанным этим скриптом: несжатыми они
утроили бы вес репозитория. Сводки `results/` и `results/cp_v1/` до 2 МБ (агрегаты
по детектору/модели/фолду, не построчные) коммитятся как есть, без него.

**Что делает.** Для каждого файла из `SOURCES` читает исходный CSV байтами, пишет
`<файл>.csv.gz` рядом детерминированным gzip (`gzip.GzipFile` с `mtime=0` и без имени
файла в заголовке, фиксированный уровень сжатия `COMPRESSLEVEL`) и тут же проверяет
себя: распаковывает написанное и сравнивает байты с исходником. Запись — во временный
файл рядом с окончательным путём и атомарная замена: битый снимок не остаётся на диске,
если сборка прервётся или проверка не сойдётся. Экспорт CSV, который не изменился
между двумя запусками, даёт те же байты снимка — иначе git видел бы изменения там,
где данные те же самые.

`--check`: ничего не пишет. Для каждого файла из `SOURCES` сверяет существующий
`.csv.gz` с текущим `results/*.csv` (декомпрессия и побайтовое сравнение) и завершается
кодом 1, если снимка или исходника нет либо они разошлись, — с сообщением по каждому
файлу. Запускать перед коммитом снимков: после прогона, изменившего один из этих
файлов (включая пересчёт стенда разладок), экспорт нужно повторить, иначе в git
останется старый снимок — это ловит именно `--check`.

    .venv/bin/python -u scripts/export_results.py [--check]
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Крупные построчные файлы (по счёту 27.09.2026: 44,3 / 88,7 / 19,0 / 28,1 / 6,7 МБ
# несжатыми), которые отчёт читает целиком, — не укладываются в правило «сводки
# до 2 МБ — как есть» (остальные CSV results/ и results/cp_v1/ идут в git без этого
# скрипта). cp_bench.csv и results/cp_v1/cp_bench.csv тоже коммитятся
# снимком: чистый клон должен собирать раздел об обнаружении разладок целиком,
# без прогона стенда (scripts/changepoints.py, ~7 минут).
SOURCES = (
    Path("results/per_series.csv"),
    Path("results/horizons_per_series.csv"),
    Path("results/forecast_2025.csv"),
    Path("results/cp_bench.csv"),
    Path("results/cp_v1/cp_bench.csv"),
)

# Один и тот же уровень на каждом запуске: разный уровень сжатия менял бы байты снимка
# без изменения данных, и git показывал бы различия там, где ничего не изменилось.
COMPRESSLEVEL = 6


def export_one(source: Path) -> Path:
    """Пишет `<source>.gz` детерминированно и проверяет себя декомпрессией.

    `mtime=0` и `filename=""` убирают из заголовка gzip время записи и имя исходного
    файла — оба меняются между прогонами и по умолчанию попадают в первые байты,
    из-за чего два экспорта одного и того же CSV давали бы разные байты снимка.
    Пишет во временный файл и заменяет им цель только после того, как распакованные
    байты совпали с исходником, — атомарно (`os.replace`), чтобы битый снимок
    не оставался на диске при обрыве или несовпадении."""
    data = source.read_bytes()
    target = Path(f"{source}.gz")
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=COMPRESSLEVEL,
                            mtime=0, filename="") as gz:
            gz.write(data)
    with gzip.open(tmp, "rb") as gz:
        restored = gz.read()
    if restored != data:
        tmp.unlink()
        raise ValueError(
            f"{target}: распакованные байты снимка не совпали с {source} — не записан"
        )
    os.replace(tmp, target)
    return target


def check_one(source: Path) -> str | None:
    """Текст проблемы, если снимок `<source>.gz` отсутствует, отсутствует исходник
    или снимок разошёлся с текущим содержимым `source`; иначе None."""
    target = Path(f"{source}.gz")
    if not source.exists():
        return f"{source}: исходника нет — сверить снимок не с чем"
    if not target.exists():
        return f"{target}: снимка нет — запустите export_results.py"
    with gzip.open(target, "rb") as gz:
        restored = gz.read()
    if restored != source.read_bytes():
        return f"{target}: снимок разошёлся с {source} — запустите export_results.py заново"
    return None


def mb(n: int) -> float:
    return n / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Снимок .csv.gz крупных results/ для публикации (или его проверка)"
    )
    parser.add_argument("--check", action="store_true",
                        help="ничего не писать, только сверить существующие снимки с CSV")
    args = parser.parse_args()

    problems = []
    for rel in SOURCES:
        source = ROOT / rel
        if args.check:
            problem = check_one(source)
            if problem:
                problems.append(problem)
                print(f"{rel}: {problem}")
            else:
                target = Path(f"{source}.gz")
                print(f"{rel}: снимок сходится с исходником "
                      f"({mb(source.stat().st_size):.1f} МБ → {mb(target.stat().st_size):.1f} МБ)")
            continue
        if not source.exists():
            problems.append(f"{source}: исходника нет — экспортировать нечего")
            print(f"{rel}: исходника нет")
            continue
        target = export_one(source)
        print(f"{rel}: {mb(source.stat().st_size):.2f} МБ → {target.relative_to(ROOT)}: "
              f"{mb(target.stat().st_size):.2f} МБ")

    if problems:
        print(f"\nпроблем: {len(problems)}")
        return 1
    print("\nOK" if args.check else "\nготово")
    return 0


if __name__ == "__main__":
    sys.exit(main())
