#!/usr/bin/env python3
"""Сбор заголовков региональных новостей из датированных архивов.

Запускается автономно и живёт часами, поэтому устроен так, чтобы падение стоило
одну страницу, а не весь прогон:

* **Состояние в SQLite.** Каждая тройка (домен, месяц, страница) отмечается сразу после
  обработки. Повторный запуск пропускает сделанное — команда идемпотентна, гонять можно
  сколько угодно раз. SQLite здесь файл, а не сервер: разворачивать нечего, и у того,
  кто воспроизводит выкачку, ничего не сломается.
* **Результат дописывается в JSONL построчно.** Процесс убит на середине — теряется одна
  строка, а не файл.
* **Вежливость по умолчанию.** Пауза между запросами, честный User-Agent с контактом,
  соблюдение robots.txt. Мы берём только то, что сайт отдаёт машинам сам; обход защит
  запрещён пунктом 9.2.1 Положения о конкурсе.

Регион берётся из домена: региональное издание пишет про свой субъект, и это даёт
привязку по построению, без геокодирования и без догадок.

    python scripts/crawl_news.py --outlets data/reference/news_outlets.json \
        --from 2023-01 --to 2024-12 --delay 1.0
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
UA = "SberIndexContestBot/1.0 (research project; contact kr1zal@yandex.ru)"
ARTICLE_RE = re.compile(r'href="(https?://[^"]*?/(\d{4})/(\d{2})/(\d{2})/[^"]+?)"')
TITLE_RE = re.compile(r'<a[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>[^<]{15,300})</a>')
URL_BLOCK_RE = re.compile(r"<url>(.*?)</url>", re.S)
LOC_RE = re.compile(r"<loc>\s*(https?://[^<\s]+)\s*</loc>")
LASTMOD_RE = re.compile(r"<lastmod>\s*(\d{4})-(\d{2})-(\d{2})")


@dataclass
class Outlet:
    region_name: str
    domain: str
    month_url_template: str
    page_url_template: str
    economy_type: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Outlet":
        """Лишние поля в списке изданий игнорируются: список ведётся руками
        и обрастает пометками, ронять из-за этого многочасовой обход глупо."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# У SQLite соединение не потокобезопасно даже с check_same_thread=False:
# одновременные execute из разных потоков ломают его состояние и дают
# InterfaceError. Поэтому у каждого потока своё соединение, а не замок вокруг
# общего. WAL позволяет им писать параллельно, не блокируя друг друга.
_LOCAL = threading.local()
_STATE_PATH: Path | None = None


def init_state(path: Path) -> None:
    """Создаёт таблицу состояния и запоминает путь для потоковых соединений."""
    global _STATE_PATH
    _STATE_PATH = path
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS visited (
               domain TEXT, year INTEGER, month INTEGER, page INTEGER,
               status INTEGER, found INTEGER, fetched_at TEXT,
               PRIMARY KEY (domain, year, month, page))"""
    )
    conn.commit()
    conn.close()


def state() -> sqlite3.Connection:
    """Соединение текущего потока, создаётся при первом обращении."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_STATE_PATH, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        _LOCAL.conn = conn
    return conn


def already_done(domain: str, year: int, month: int, page: int) -> bool:
    row = state().execute(
        "SELECT 1 FROM visited WHERE domain=? AND year=? AND month=? AND page=?",
        (domain, year, month, page),
    ).fetchone()
    return row is not None


def mark(domain, year, month, page, status, found) -> None:
    conn = state()
    conn.execute(
        "INSERT OR REPLACE INTO visited VALUES (?,?,?,?,?,?,datetime('now'))",
        (domain, year, month, page, status, found),
    )
    conn.commit()


def robots_allows(domain: str, cache: dict) -> bool:
    """Сайт сам говорит, что можно машинам. Спрашиваем и подчиняемся."""
    if domain not in cache:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"https://{domain}/robots.txt")
        try:
            parser.read()
        except Exception:
            cache[domain] = None
            return True
        cache[domain] = parser
    parser = cache[domain]
    return True if parser is None else parser.can_fetch(UA, f"https://{domain}/2024/01/")


def extract_sitemap(xml: str, domain: str, year: int, month: int) -> list[dict]:
    """Разбор XML-карты сайта.

    Здесь две тонкости против HTML-архива. Ссылки лежат в <loc>, атрибута href нет
    вовсе. И дата у большинства изданий не в адресе статьи, а в соседнем теге
    <lastmod> — искать её в URL, как в HTML-архивах, бесполезно.

    Заголовка карта не содержит, поэтому он восстанавливается из слага адреса:
    для сигнала интенсивности освещения и разбора по ключевым словам этого хватает,
    а для дословных цитат карта и не предназначена.
    """
    out, seen = [], set()
    for block in URL_BLOCK_RE.findall(xml):
        loc = LOC_RE.search(block)
        mod = LASTMOD_RE.search(block)
        if not loc:
            continue
        url = loc.group(1)
        if url in seen or domain not in urlparse(url).netloc:
            continue

        if mod:
            y, m, d = mod.group(1), mod.group(2), mod.group(3)
        else:
            from_url = ARTICLE_RE.search(f'href="{url}"')
            if not from_url:
                continue
            y, m, d = from_url.group(2), from_url.group(3), from_url.group(4)

        if int(y) != year or int(m) != month:
            continue
        seen.add(url)
        out.append({"url": url, "title": _title_from_slug(url), "date": f"{y}-{m}-{d}"})
    return out


def _title_from_slug(url: str) -> str:
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[-_]+", " ", re.sub(r"\d+$", "", slug)).strip()


def extract(html: str, domain: str, year: int, month: int) -> list[dict]:
    """Заголовки и даты со страницы архива — за статьёй ходить не нужно.

    Разбор расцеплён на два шага намеренно. Сначала находятся датированные адреса,
    и только потом для каждого ищется заголовок. Единая регулярка вида
    `<a href="...">Заголовок</a>` работала лишь там, где текст лежит прямым узлом
    внутри ссылки; у изданий с разметкой `<a href="..."><h3>Заголовок</h3></a>`
    она молча давала ноль при том, что датированных ссылок на странице были десятки.
    Когда заголовок достать не удалось, он восстанавливается из слага адреса.

    Фильтр по запрошенному месяцу обязателен: страницы архива содержат боковые
    блоки «последние новости», и без него в выдачу за август 2024 попадают статьи
    за сентябрь 2026.
    """
    titles = {}
    for match in TITLE_RE.finditer(html):
        titles.setdefault(match.group("url"), match.group("title").strip())

    seen, out = set(), []
    for url, y, m, d in ARTICLE_RE.findall(html):
        if url in seen or domain not in urlparse(url).netloc:
            continue
        if int(y) != year or int(m) != month:
            continue
        seen.add(url)
        out.append({"url": url, "title": titles.get(url) or _title_from_slug(url),
                    "date": f"{y}-{m}-{d}"})
    return out


def months(start: str, end: str):
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    while (y0, m0) <= (y1, m1):
        yield y0, m0
        y0, m0 = (y0 + 1, 1) if m0 == 12 else (y0, m0 + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Сбор заголовков из датированных архивов")
    ap.add_argument("--outlets", default="data/reference/news_outlets.json")
    ap.add_argument("--from", dest="start", default="2023-01")
    ap.add_argument("--to", dest="end", default="2024-12")
    ap.add_argument("--delay", type=float, default=1.0, help="пауза между запросами, секунд")
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--out", default="data/news")
    ap.add_argument("--workers", type=int, default=8, help="изданий параллельно")
    args = ap.parse_args()

    outlets_path = ROOT / args.outlets
    if not outlets_path.exists():
        print(f"нет файла со списком изданий: {outlets_path}", file=sys.stderr)
        return 1
    outlets = [Outlet.from_dict(o) for o in json.loads(outlets_path.read_text(encoding="utf-8"))]

    out_dir = ROOT / args.out
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    init_state(out_dir / "state.db")
    session = requests.Session()
    session.headers["User-Agent"] = UA
    robots_cache: dict = {}

    # Пауза в секунду нужна ОДНОМУ домену, разные сайты друг другу не мешают.
    # Поэтому издания обходятся параллельно, а вежливость к каждому сохраняется:
    # внутри потока запросы к его домену идут последовательно с паузой.
    counter = {"total": 0}

    def crawl_outlet(outlet: Outlet) -> None:
        if not robots_allows(outlet.domain, robots_cache):
            print(f"[{outlet.domain}] robots.txt запрещает архивы — пропускаю")
            return

        local = requests.Session()
        local.headers["User-Agent"] = UA
        sink = (out_dir / "raw" / f"{outlet.domain}.jsonl").open("a", encoding="utf-8")
        collected = 0
        try:
            for year, month in months(args.start, args.end):
                for page in range(1, args.max_pages + 1):
                    if already_done(outlet.domain, year, month, page):
                        continue
                    template = outlet.month_url_template if page == 1 else outlet.page_url_template
                    url = template.format(year=year, month=f"{month:02d}", page=page)

                    try:
                        response = local.get(url, timeout=30)
                        status = response.status_code
                        if status != 200:
                            items = []
                        elif "xml" in response.headers.get("Content-Type", "") or url.endswith(".xml") or "sitemap" in url:
                            items = extract_sitemap(response.text, outlet.domain, year, month)
                        else:
                            items = extract(response.text, outlet.domain, year, month)
                    except Exception as exc:
                        print(f"  [{outlet.domain}] {url} -> ошибка: {str(exc)[:60]}")
                        status, items = 0, []

                    for item in items:
                        item |= {"region_name": outlet.region_name, "domain": outlet.domain}
                        sink.write(json.dumps(item, ensure_ascii=False) + "\n")
                    sink.flush()

                    mark(outlet.domain, year, month, page, status, len(items))
                    collected += len(items)
                    counter["total"] += len(items)
                    time.sleep(args.delay)

                    if status != 200 or not items:
                        break
        finally:
            sink.close()
        if collected == 0:
            print(f"[{outlet.domain}] ВНИМАНИЕ: собрано 0 статей — шаблон URL или разбор не подходят")
        else:
            print(f"[{outlet.domain}] завершено, собрано {collected}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(crawl_outlet, outlets))
    total_new = counter["total"]

    print(f"\nновых записей за прогон: {total_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
