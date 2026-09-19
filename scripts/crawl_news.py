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
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
UA = "SberIndexContestBot/1.0 (research project; contact kr1zal@yandex.ru)"
ARTICLE_RE = re.compile(r'href="(https?://[^"]*?/(\d{4})/(\d{2})/(\d{2})/[^"]+?)"')
TITLE_RE = re.compile(r'<a[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>[^<]{15,300})</a>')


@dataclass
class Outlet:
    region_name: str
    domain: str
    month_url_template: str
    page_url_template: str
    economy_type: str = ""


def open_state(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS visited (
               domain TEXT, year INTEGER, month INTEGER, page INTEGER,
               status INTEGER, found INTEGER, fetched_at TEXT,
               PRIMARY KEY (domain, year, month, page))"""
    )
    conn.commit()
    return conn


def already_done(conn: sqlite3.Connection, domain: str, year: int, month: int, page: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM visited WHERE domain=? AND year=? AND month=? AND page=?",
        (domain, year, month, page),
    ).fetchone()
    return row is not None


def mark(conn, domain, year, month, page, status, found) -> None:
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


def extract(html: str, domain: str, year: int, month: int) -> list[dict]:
    """Заголовок и дата берутся прямо со страницы архива — за статьёй ходить не нужно.

    Обязательна фильтрация по запрошенному месяцу: страницы архива содержат боковые
    блоки «последние новости» и «популярное», и без фильтра в выдачу за август 2024
    попадают статьи за сентябрь 2026. Корпус молча наполнялся бы неверными датами,
    а ошибка проявилась бы только на стыковке с панелью — то есть поздно.
    """
    seen, out = set(), []
    for match in TITLE_RE.finditer(html):
        url, title = match.group("url"), match.group("title").strip()
        date = ARTICLE_RE.search(f'href="{url}"')
        if not date or url in seen or domain not in urlparse(url).netloc:
            continue
        if int(date.group(2)) != year or int(date.group(3)) != month:
            continue
        seen.add(url)
        out.append(
            {"url": url, "title": title,
             "date": f"{date.group(2)}-{date.group(3)}-{date.group(4)}"}
        )
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
    args = ap.parse_args()

    outlets_path = ROOT / args.outlets
    if not outlets_path.exists():
        print(f"нет файла со списком изданий: {outlets_path}", file=sys.stderr)
        return 1
    outlets = [Outlet(**o) for o in json.loads(outlets_path.read_text(encoding="utf-8"))]

    out_dir = ROOT / args.out
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    conn = open_state(out_dir / "state.db")
    session = requests.Session()
    session.headers["User-Agent"] = UA
    robots_cache: dict = {}

    total_new = 0
    for outlet in outlets:
        if not robots_allows(outlet.domain, robots_cache):
            print(f"[{outlet.domain}] robots.txt запрещает архивы — пропускаю")
            continue

        sink = (out_dir / "raw" / f"{outlet.domain}.jsonl").open("a", encoding="utf-8")
        for year, month in months(args.start, args.end):
            for page in range(1, args.max_pages + 1):
                if already_done(conn, outlet.domain, year, month, page):
                    continue
                template = outlet.month_url_template if page == 1 else outlet.page_url_template
                url = template.format(year=year, month=f"{month:02d}", page=page)

                try:
                    response = session.get(url, timeout=30)
                    status = response.status_code
                    items = extract(response.text, outlet.domain, year, month) if status == 200 else []
                except Exception as exc:
                    print(f"  {url} -> ошибка: {str(exc)[:60]}")
                    status, items = 0, []

                for item in items:
                    item |= {"region_name": outlet.region_name, "domain": outlet.domain}
                    sink.write(json.dumps(item, ensure_ascii=False) + "\n")
                sink.flush()

                mark(conn, outlet.domain, year, month, page, status, len(items))
                total_new += len(items)
                time.sleep(args.delay)

                if status != 200 or not items:
                    break  # страницы кончились — дальше в этом месяце смысла нет
            print(f"[{outlet.domain}] {year}-{month:02d} готов, всего собрано {total_new}")
        sink.close()

    print(f"\nновых записей за прогон: {total_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
