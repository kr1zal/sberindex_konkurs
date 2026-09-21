"""Выборочная выкачка тел статей: сколько темы теряют на одних заголовках.

Весь новостной корпус построен на заголовках. Заголовок — это пятнадцать слов,
и тема в нём может не назваться вовсе: статья про решение по ключевой ставке
вполне может называться «Что будет с ипотекой». Значит тематические доли,
посчитанные по заголовкам, систематически занижены, и неизвестно, насколько.

Выкачивать 588 тысяч тел ради этого — дорого и невежливо. Поэтому берётся
выборка в двадцать тысяч статей, и решение о полной выкачке принимается
по измеренным числам, а не заранее.

**Выборка стратифицирована по месяцам**, поровну на месяц. Сравнивать предстоит
помесячные ряды долей, и равная точность в каждом месяце здесь важнее, чем
точное воспроизведение долей изданий: объёмы месяцев и так близки, от 19,7
до 30 тысяч публикаций.

## Вежливость — та же, что у краулера заголовков

Тот же User-Agent с контактом, robots.txt проверяется **на каждый адрес статьи**,
не чаще одного запроса в секунду к домену, параллельность между доменами.

Вежливость меряется частотой запросов к сайту, а не тем, ждём ли мы ответа.
Первая версия отправляла следующий запрос только после получения предыдущего,
и на медленных изданиях выходило по четыре статьи в минуту вместо шестидесяти —
двадцать тысяч статей заняли бы четыре часа. Теперь внутри домена работает
несколько потоков с общим ограничителем: интервал между **началами** запросов
к одному домену по-прежнему не меньше `--delay`, а ожидание ответов идёт
параллельно.

Состояние в SQLite: прогон идемпотентен, повторный запуск продолжает с места
обрыва. Обход защит не делается — если сайт отвечает отказом, статья просто
не попадает в выборку, и это фиксируется в отчёте.

Тело статьи вырезается из абзацев `<p>`; хранится не больше `--max-chars`
символов — для попадания в тему по корням этого с избытком, а файл остаётся
обозримым.

    .venv/bin/python -u scripts/fetch_bodies.py --sample 20000 --delay 1.0
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import sqlite3
import sys
import threading
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
UA = "SberIndexContestBot/1.0 (research project; contact kr1zal@yandex.ru)"
STATE_PATH = ROOT / "data" / "news" / "bodies_state.db"
OUT_PATH = ROOT / "data" / "news" / "bodies_sample.jsonl"

SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
PARA_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")

_local = threading.local()
_lock = threading.Lock()


class RateLimiter:
    """Не чаще одного запроса в `interval` секунд. Считает от начала запроса."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


def state() -> sqlite3.Connection:
    if not hasattr(_local, "connection"):
        _local.connection = sqlite3.connect(STATE_PATH, timeout=30)
    return _local.connection


def init_state() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(STATE_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS fetched ("
            "url TEXT PRIMARY KEY, status TEXT, chars INTEGER)"
        )


def already_done(url: str) -> bool:
    row = state().execute("SELECT 1 FROM fetched WHERE url = ?", (url,)).fetchone()
    return row is not None


def mark(url: str, status: str, chars: int) -> None:
    with _lock:
        state().execute(
            "INSERT OR REPLACE INTO fetched VALUES (?, ?, ?)", (url, status, chars)
        )
        state().commit()


def robots_parser(domain: str, cache: dict):
    """robots.txt издания. Недоступный robots.txt трактуется как запрет."""
    if domain not in cache:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"https://{domain}/robots.txt")
        try:
            parser.read()
        except Exception:
            parser = None
        cache[domain] = parser
    return cache[domain]


def extract_text(page: str, max_chars: int) -> str:
    """Текст статьи из абзацев. Без внешних библиотек — их у проверяющего может не быть."""
    page = SCRIPT_RE.sub(" ", page)
    parts = []
    for block in PARA_RE.findall(page):
        text = html.unescape(TAG_RE.sub(" ", block))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 40:  # подписи, копирайты и кнопки отсекаются длиной
            parts.append(text)
        if sum(map(len, parts)) >= max_chars:
            break
    return " ".join(parts)[:max_chars]


def fetch_domain(domain: str, rows: list[dict], args, cache: dict, counters: dict) -> None:
    parser = robots_parser(domain, cache)
    if parser is None:
        print(f"[{domain}] robots.txt недоступен — пропускаю целиком")
        with _lock:
            counters["robots_skip"] += len(rows)
        return

    limiter = RateLimiter(args.delay)
    with ThreadPoolExecutor(max_workers=args.per_domain) as pool:
        for chunk in range(args.per_domain):
            pool.submit(
                fetch_slice, domain, rows[chunk :: args.per_domain], args,
                parser, counters, limiter,
            )


def fetch_slice(domain, rows, args, parser, counters, limiter) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    for row in rows:
        url = row["url"]
        if already_done(url):
            with _lock:
                counters["cached"] += 1
            continue
        if not parser.can_fetch(UA, url):
            mark(url, "robots", 0)
            with _lock:
                counters["robots_skip"] += 1
            continue
        limiter.wait()
        try:
            response = session.get(url, timeout=args.timeout)
            if response.status_code != 200:
                mark(url, f"http{response.status_code}", 0)
                with _lock:
                    counters["http_error"] += 1
            else:
                text = extract_text(response.text, args.max_chars)
                mark(url, "ok", len(text))
                with _lock:
                    counters["ok"] += 1
                    with OUT_PATH.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(
                            {"url": url, "domain": domain, "month": str(row["month"]),
                             "title": row["title"], "body": text}, ensure_ascii=False) + "\n")
        except Exception as exc:
            mark(url, f"err:{type(exc).__name__}", 0)
            with _lock:
                counters["network_error"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--per-domain", type=int, default=6,
                        help="потоков на домен; частоту запросов это не меняет")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    init_state()
    headlines = pd.read_parquet(ROOT / "data/news/headlines.parquet")
    headlines["month"] = pd.to_datetime(headlines["date"]).dt.to_period("M")

    months = sorted(headlines["month"].unique())
    per_month = args.sample // len(months)
    random.seed(args.seed)
    chunks = []
    for month in months:
        part = headlines.loc[headlines["month"] == month]
        chunks.append(part.sample(min(per_month, len(part)), random_state=args.seed))
    sample = pd.concat(chunks, ignore_index=True)
    print(f"выборка: {len(sample)} статей, {per_month} на месяц, "
          f"{sample['domain'].nunique()} изданий, пауза {args.delay} с\n")

    counters = {k: 0 for k in ("ok", "cached", "robots_skip", "http_error", "network_error")}
    cache: dict = {}
    by_domain = [
        (domain, part.to_dict("records"))
        for domain, part in sample.groupby("domain", observed=True)
    ]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(by_domain)) as pool:
        for domain, rows in by_domain:
            pool.submit(fetch_domain, domain, rows, args, cache, counters)

    print(f"\nготово за {(time.perf_counter() - started) / 60:.1f} мин")
    for key, value in counters.items():
        print(f"  {key}: {value}")
    print(f"\nтела: {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
