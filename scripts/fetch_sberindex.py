"""Выгрузка наборов данных СберИндекса, на которых строится перенос с длинных рядов.

Муниципальная панель — двадцать четыре месяца, и удлинить её нечем: собственный
набор СберИндекса по муниципальным образованиям ровно такой же длины. Зато рядом
в том же каталоге лежат федеральные и отраслевые ряды за семь-девять лет, собранные
той же методикой по тем же транзакциям. Они и дают историю, которой у панели нет.

## Механика выгрузки

Кнопка «Скачать» на дашборде дёргает

    GET https://sberindex.ru/api/dataset/v1//download/<slug>/parquet

(двойной слэш — не опечатка, так в самом сайте), а эндпоинт отвечает редиректом на

    GET https://sberindex.ru/api/files/download/<uuid>

Первый адрес закрыт: голый curl получает 403 независимо от заголовков — фильтрация
идёт не по User-Agent, а на уровне TLS-отпечатка. Второй адрес открыт и отдаётся
обычному curl. Поэтому выгрузка делается в два шага: uuid снимается один раз из
живой сессии браузера, дальше файл качается скриптом.

Снятые 20.09.2026 uuid'ы записаны в UUIDS ниже. Они привязаны к версии файла и
меняются при обновлении набора — если скрипт получит 404, uuid надо снять заново:
открыть страницу дашборда, в консоли браузера выполнить

    (await fetch('/api/dataset/v1//download/<slug>/parquet')).url

и подставить хвост адреса. Полный каталог — https://sberindex.ru/ru/dashboards/

## Лицензия

Наборы СберИндекса публикуются на условиях CC BY-SA 4.0, как и справочник
муниципальных образований. Источник указывается в отчёте.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "reference" / "sberindex"

FILES_URL = "https://sberindex.ru/api/files/download/{uuid}"
DATASET_URL = "https://sberindex.ru/api/dataset/v1//download/{slug}/parquet"

# Браузерный User-Agent тут не обход отказа, а условие отдачи файла: эндпоинт
# /api/files/ отвечает обычному клиенту, но заголовки живой сессии ему нужны.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

# slug -> (uuid на 20.09.2026, зачем нам этот набор)
UUIDS: dict[str, tuple[str, str]] = {
    "consumer-spending": (
        "d16f297d-828c-4230-ba1e-59cf27845bd0",
        "потребительские расходы России, 93 месяца — главный общий фактор панели",
    ),
    "consumper-spending-index-sa": (
        "562cbe16-5fe1-4260-84e5-b2125186586a",
        "индекс реальных расходов с сезонной корректировкой, 93 месяца",
    ),
    "consumer-spending-growth": (
        "f76662e3-82e2-4617-85f1-48bdd42f9fd5",
        "приросты потребительских расходов, 93 месяца",
    ),
    "median-wages": (
        "706ba66b-1d0c-468e-a13e-07db550e2577",
        "медианная зарплата по 19 отраслям, 79 месяцев",
    ),
    "oboroty-biznesa": (
        "c3da974e-fe7a-4396-bdc2-c5bab1dd9b17",
        "обороты бизнеса по 20 отраслям, 116 месяцев — самая длинная история",
    ),
    "izmenenie-obema-fot": (
        "b1154d11-bbfe-49e5-ae9c-5445a14dda00",
        "фонд оплаты труда по 20 отраслям, 113 месяцев",
    ),
    "real-key-interest-rate": (
        "04dd2b1c-fe09-4d45-a794-41479232085c",
        "ключевая ставка в реальном выражении, 157 месяцев",
    ),
    "dolya-beznala": (
        "05437791-3732-45ef-98c5-7a32f399964a",
        "доля безналичных расходов, 86 регионов — региональная детализация, но только с 2023",
    ),
    "real_estate_deals": (
        "4e7c3b69-eefe-4f48-94ee-bd628fd6a2eb",
        "цены сделок с недвижимостью, 116 месяцев × 86 регионов — "
        "единственный набор, который одновременно длинный и региональный",
    ),
    "nedelnaa-inflazia-v-razreze-analiticeskih-komponentov": (
        "e26eb28b-858e-4447-a56f-39d2d8008a8c",
        "недельная инфляция по компонентам, 117 месяцев",
    ),
    "potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov": (
        "aa49b9ae-1531-472a-a273-809d54826a96",
        "потребительская активность по возрастным группам, 69 месяцев",
    ),
    "izmenenie-kolichestva-aktivnikh-torgovo-servisnikh-tochek-po-kategoriyam": (
        "97b24513-a845-45c2-b2ab-90497e39d540",
        "число активных торгово-сервисных точек по категориям, 68 месяцев",
    ),
}


def fetch(slug: str, uuid: str, out_dir: Path, timeout: int = 60) -> int:
    """Качает один набор. Возвращает размер файла; 0 — если не получилось.

    Качаем curl'ом, а не urllib, и причина не в удобстве. sberindex.ru отдаёт
    неполную цепочку сертификатов: сертификат сайта выписан промежуточным
    центром TrustAsia, а сам промежуточный сервер не присылает. curl и браузеры
    достают недостающее звено по ссылке из расширения AIA и берут корни из
    системного хранилища; модуль ssl в Питоне не делает ни того, ни другого
    и падает на CERTIFICATE_VERIFY_FAILED — в том числе с certifi, потому что
    отсутствует не корень, а промежуточный сертификат.
    """
    target = out_dir / f"{slug}.parquet"
    result = subprocess.run(
        [
            "curl", "--silent", "--show-error", "--location",
            "--max-time", str(timeout),
            "--user-agent", HEADERS["User-Agent"],
            "--header", f"Referer: https://sberindex.ru/ru/dashboards/{slug}",
            "--header", f"Accept: {HEADERS['Accept']}",
            "--write-out", "%{http_code}",
            "--output", str(target),
            FILES_URL.format(uuid=uuid),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        print(f"  {slug}: curl не смог ({result.stderr.strip()[:100]})")
        return 0

    code = result.stdout.strip()[-3:]
    if code != "200":
        target.unlink(missing_ok=True)
        print(f"  {slug}: HTTP {code} — вероятно, uuid устарел, снимите новый")
        return 0

    # Parquet начинается магическим словом PAR1. Проверка нужна потому, что при
    # устаревшем uuid сервис отдаёт HTML-заглушку с кодом 200, и без неё в data/
    # молча ляжет страница ошибки под именем набора данных.
    with target.open("rb") as handle:
        if handle.read(4) != b"PAR1":
            size = target.stat().st_size
            target.unlink()
            print(f"  {slug}: ответ не parquet ({size} байт), файл не сохранён")
            return 0

    return target.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT_DIR), help="куда складывать parquet")
    parser.add_argument("--only", nargs="*", default=None, help="подмножество slug'ов")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    slugs = args.only or list(UUIDS)
    if unknown := set(slugs) - set(UUIDS):
        raise KeyError(f"неизвестные slug: {sorted(unknown)}")
    if shutil.which("curl") is None:
        print("нужен curl — см. объяснение в docstring функции fetch()")
        return 1

    print(f"выгрузка {len(slugs)} наборов в {out_dir}")
    failed = []
    for slug in slugs:
        uuid, purpose = UUIDS[slug]
        size = fetch(slug, uuid, out_dir)
        if size:
            print(f"  {slug}: {size / 1024:.0f} КБ — {purpose}")
        else:
            failed.append(slug)

    if failed:
        print(f"\nне выгружено: {', '.join(failed)}")
        print(f"снимите uuid заново, см. инструкцию в шапке {Path(__file__).name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
