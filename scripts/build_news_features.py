"""Пересборка новостных признаков после починки словаря тем.

До 25.09 корни тем искались как подстроки в любом месте слова: `stavk` ловил
«выставку», «отставку», «поставку» и «доставку», и тема ДКП больше чем наполовину
состояла из них; `cen` и `czen` в «ценах» ловили «центр». Теперь корень начинает
слово, кроме корней, законно живущих внутри сложных слов (`src.news.topic_pattern`,
`src.news.COMPOUND_ROOTS`), и признаки собираются заново: их читают `two_stage_news`
и новостные проверки.

Скрипт — единственный, кто пишет `data/news/monthly.parquet` и `data/news/national.parquet`:

1. читает `data/news/headlines.parquet` — слой заголовков он не меняет; в репозитории
   файла нет (заголовки чужих изданий, права не наши), он выкачивается локально
   `scripts/crawl_news.py`;
2. по каждой теме считает заголовки, попавшие по старому словарю и по новому, долю
   снятого и пять самых частых снятых слов; сверяет доли тем по старому словарю
   с файлом до 25.09 и печатает корреляцию старого и нового месячного ряда ДКП.
   «Старое» — всегда словарь и файл до 25.09, а не прошлая пересборка: после первой
   пересборки исходный `national.parquet` лежит в копии, и сверка идёт с ней. Нет копии —
   исходником считается `data/news`, только если старый словарь его воспроизводит;
   иначе (чистый клон, где там уже пересборка) сверка пропускается;
3. собирает оба файла заново и сверяет со старыми колонки, типы и колонки не из
   словаря — до всякой записи;
4. пишет таблицу в `results/news_dictionary.csv`, копирует исходные файлы
   в `results/backup_2026-09-25/` (копию, которая там уже есть, не трогает; не исходные
   под эту дату не кладёт) и записывает новые.

`--dry-run` печатает всё и ничего не пишет.

    .venv/bin/python -u scripts/build_news_features.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.news import (  # noqa: E402
    NATIONAL_TOPICS, monthly_features, national_features, repair_encoding, save_datasets,
    topic_pattern, transliterate,
)

NEWS = Path("data/news")
BACKUP = Path("results/backup_2026-09-25")
TABLE = Path("results/news_dictionary.csv")
FILES = ("monthly.parquet", "national.parquet")
TOP_WORDS = 5

# Словарь до 25.09, только для таблицы до/после. Корни искались как подстроки
# в любом месте слова: паттерн темы — `"|".join(roots)`.
LEGACY_TOPICS = {
    "ceny": ["cen", "czen", "tsen", "podorozh", "infl", "deshev", "tarif"],
    "dohody": ["zarplat", "dohod", "pensi", "vyplat", "posobi"],
    "zanjatost": ["rabot", "uvol", "vakans", "sokrash", "bezrabot"],
    "proizvodstvo": ["zavod", "predprijati", "proizvodstv", "fabrik", "cekh", "ceh"],
    "kredit": ["kredit", "ipotek", "bank", "dolg", "stavk"],
    "torgovlja": ["magazin", "torgov", "rynok", "rynk", "otkry", "zakry"],
    "dkp": [
        r"\bczb\b", "czentrobank", "kljuchev", "nabiullin", "reguljator",
        "stavk", "infljacz", "proczentn",
    ],
}


def prepared(headlines: pd.DataFrame) -> pd.DataFrame:
    """Заголовки в том виде, в каком их собирал `src.news.load_headlines`: регион
    и издание — строками, месяц — первым днём месяца, текст — починенным и в латинице.

    Категории из parquet оставлять нельзя: группировка по ним в `monthly_features`
    добавила бы несуществующие пары издание × регион с нулём публикаций и сломала бы
    интенсивность и число изданий. Прежний `monthly.parquet` собирался из строк
    и воспроизводится только так."""
    frame = headlines.copy()
    for column in ("region_name", "domain"):
        frame[column] = frame[column].astype(str)
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    frame["norm"] = frame["title"].map(repair_encoding).map(transliterate)
    return frame


def topic_hits(norm: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Попадания каждого заголовка в темы: по старому словарю и по новому."""
    old = pd.DataFrame({t: norm.str.contains("|".join(LEGACY_TOPICS[t]), regex=True) for t in NATIONAL_TOPICS})
    new = pd.DataFrame({t: norm.str.contains(topic_pattern(r), regex=True) for t, r in NATIONAL_TOPICS.items()})
    return old, new


def removed_words(norm: pd.Series, old: re.Pattern, new: re.Pattern) -> list[tuple[str, int]]:
    """Самые частые слова, которые старый паттерн ловил, а новый нет, с числом заголовков,
    где слово есть. Слово — токен нормализованного заголовка (`\\w+`). Проверка по токену
    та же, что в заголовке: граница начала слова смотрит на символ перед корнем, а перед
    токеном стоит не буква."""
    counts: Counter = Counter()
    verdict: dict[str, bool] = {}
    for text in norm:
        for word in set(re.findall(r"\w+", text)):
            if word not in verdict:
                verdict[word] = old.search(word) is not None and new.search(word) is None
            if verdict[word]:
                counts[word] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:TOP_WORDS]


def dictionary_table(norm: pd.Series, hits: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Таблица до/после по темам. `hits` — готовые попадания `topic_hits`, если уже посчитаны."""
    old, new = hits if hits is not None else topic_hits(norm)
    rows = []
    for topic, roots in NATIONAL_TOPICS.items():
        removed = old[topic] & ~new[topic]
        words = removed_words(
            norm[old[topic]], re.compile("|".join(LEGACY_TOPICS[topic])), re.compile(topic_pattern(roots)),
        )
        before = int(old[topic].sum())
        rows.append({
            "тема": topic,
            "заголовков до": before,
            "заголовков после": int(new[topic].sum()),
            "снято": int(removed.sum()),
            "добавлено": int((new[topic] & ~old[topic]).sum()),
            "доля снятого, %": 100.0 * int(removed.sum()) / before if before else np.nan,
            "частые снятые слова": "; ".join(f"{word} {n}" for word, n in words),
        })
    return pd.DataFrame(rows)


def legacy_shares(old_hits: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Помесячные доли тем по старому словарю — ряды `national.parquet` до 25.09. От того,
    что сейчас лежит в `data/news`, не зависят: «до» в таблице — всегда словарь до 25.09."""
    return old_hits.groupby(pd.PeriodIndex(dates.dt.to_period("M"))).mean()


def legacy_gap(shares: pd.DataFrame, stored: pd.DataFrame) -> float:
    """Наибольшее расхождение долей тем по старому словарю с файлом до 25.09 (`stored`).
    Ноль — «до» в таблице описывает ровно этот файл."""
    stored = stored.set_index("month")[[f"t_{t}" for t in shares.columns]]
    stored.columns = list(shares.columns)
    gap = (shares.reindex(stored.index) - stored).abs().to_numpy()
    return float(np.max(gap)) if np.isfinite(gap).all() else float("inf")


def aligned(new: pd.DataFrame, old: pd.DataFrame, name: str) -> pd.DataFrame:
    """Новый кадр с колонками и типами старого файла: читатели файла не должны заметить
    ничего, кроме чисел тем. Имена и порядок колонок совпадают, типы тоже, кроме
    разрешения дат: старый monthly.parquet записан pandas 3 с датами в микросекундах,
    и новый приводится к ним. Любое другое расхождение — отказ до всякой записи."""
    print(f"\n{name}: колонки старого и нового файла")
    for column in dict.fromkeys([*old.columns, *new.columns]):
        was = str(old[column].dtype) if column in old else "—"
        now = str(new[column].dtype) if column in new else "—"
        print(f"  {column:16s} {was:16s} {now}")
    if list(new.columns) != list(old.columns):
        raise SystemExit(f"{name}: колонки не те, что были в старом файле — ничего не записано")
    out = new.copy()
    for column in old.columns:
        if out[column].dtype == old[column].dtype:
            continue
        if pd.api.types.is_datetime64_dtype(out[column]) and pd.api.types.is_datetime64_dtype(old[column]):
            out[column] = out[column].astype(old[column].dtype)
            print(f"  {column}: {new[column].dtype} -> {old[column].dtype}, как в старом файле")
            continue
        raise SystemExit(
            f"{name}: тип {column} {out[column].dtype} против {old[column].dtype} в старом файле — "
            "ничего не записано"
        )
    # Словарь меняет только доли тем: всё остальное обязано совпасть со старым файлом.
    # Не совпало — пересобирается не то (так выглядела бы группировка по категориям
    # в monthly_features), и писать такое нельзя.
    keys = [c for c in old.columns if not c.startswith("t_")]
    same = out[keys].reset_index(drop=True).equals(old[keys].reset_index(drop=True))
    print(f"  колонки не из словаря ({', '.join(keys)}) совпадают со старыми: {'да' if same else 'НЕТ'}")
    if not same:
        raise SystemExit(
            f"{name}: колонки не из словаря разошлись со старым файлом — скрипт пересобирает словарь "
            "тем, а не корпус; ничего не записано"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Пересборка новостных признаков по исправленному словарю")
    parser.add_argument("--dry-run", action="store_true", help="напечатать всё и ничего не записать")
    args = parser.parse_args(argv)

    news_dir, backup, table_path = ROOT / NEWS, ROOT / BACKUP, ROOT / TABLE
    headlines_path = news_dir / "headlines.parquet"
    if not headlines_path.exists():
        raise FileNotFoundError(
            f"{headlines_path} нет — заголовков в репозитории не бывает (права чужих изданий). "
            "Собрать: python scripts/crawl_news.py --outlets data/reference/news_outlets.json "
            "--from 2023-01 --to 2024-12, затем src.news.load_headlines('data/news/raw')"
            "[['url', 'title', 'date', 'region_name', 'domain']]"
            ".to_parquet('data/news/headlines.parquet', compression='zstd', index=False)."
        )
    headlines = pd.read_parquet(headlines_path)
    old = {name: pd.read_parquet(news_dir / name) for name in FILES}
    frame = prepared(headlines)
    print(f"заголовков {len(frame)}, месяцев {frame['month'].nunique()}, изданий {frame['domain'].nunique()}")

    hits = topic_hits(frame["norm"])
    shares = legacy_shares(hits[0], headlines["date"])
    # Исходный файл до 25.09 — копия в results/backup_…, а пока её нет — data/news, но только
    # если его воспроизводит старый словарь. На чистом клоне в data/news уже пересборка:
    # сверять с ней «до» бессмысленно, а положить её в копию под датой до 25.09 — значит
    # подменить исходник, и сверка потом не сойдётся никогда. Решение по национальному
    # файлу: оба файла пересобираются вместе.
    news_is_original = legacy_gap(shares, old["national.parquet"]) < 1e-12
    copy = backup / "national.parquet"
    if copy.exists():
        gap = legacy_gap(shares, pd.read_parquet(copy))
        print(f"старые доли тем воспроизводятся старым словарём: {'да' if gap < 1e-12 else 'НЕТ'} "
              f"(наибольшее расхождение с {copy.relative_to(ROOT)} {gap:.3g})")
    elif news_is_original:
        print(f"старые доли тем воспроизводятся старым словарём: да "
              f"(исходный файл — {NEWS / 'national.parquet'}, копии ещё нет)")
    else:
        print(f"копии исходного файла нет — сверка воспроизведения пропущена: старый словарь "
              f"не воспроизводит {NEWS / 'national.parquet'} (пересборка или расхождение сборки)")
    table = dictionary_table(frame["norm"], hits)

    monthly = aligned(monthly_features(frame), old["monthly.parquet"], "monthly.parquet")
    national = aligned(national_features(headlines), old["national.parquet"], "national.parquet")
    after = national.set_index("month")
    # У постоянного ряда корреляции нет: NaN — верный ответ, предупреждение numpy лишнее.
    with np.errstate(invalid="ignore", divide="ignore"):
        table["корреляция рядов"] = [shares[t].corr(after[f"t_{t}"]) for t in table["тема"]]

    print("\nСЛОВАРЬ ДО И ПОСЛЕ: заголовки корпуса по темам; корреляция — помесячных долей по стране, "
          "словарь до 25.09 против нового")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    dkp = table.set_index("тема").loc["dkp"]
    print(f"\nтема ДКП: снято {dkp['снято']} из {dkp['заголовков до']} заголовков ({dkp['доля снятого, %']:.1f}%)")
    print(f"корреляция старого и нового месячного ряда ДКП: {dkp['корреляция рядов']:.3f}")

    if args.dry_run:
        print("\n--dry-run: ничего не записано")
        return 0

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    for name in FILES:
        target = backup / name
        if target.exists():
            print(f"копия {target.relative_to(ROOT)} уже есть — не трогаю")
        elif news_is_original:
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(news_dir / name, target)
            print(f"старый {name} скопирован в {target.relative_to(ROOT)}")
        else:
            print(f"копию {name} не создаю: старый словарь не воспроизводит {NEWS / 'national.parquet'}, "
                  "а под датой до 25.09 должен лежать исходный файл")
    save_datasets(None, monthly, news_dir, national=national)
    for name in FILES:
        written = pd.read_parquet(news_dir / name)
        if written.dtypes.to_dict() != old[name].dtypes.to_dict():
            where = BACKUP / name if (backup / name).exists() else "истории git"
            raise SystemExit(f"{name}: перечитанный файл не совпал по типам со старым; старый — в {where}")
    print(f"\nзаписано: {TABLE}, {NEWS / 'monthly.parquet'}, {NEWS / 'national.parquet'}; "
          "перечитаны — колонки и типы как в старых файлах")
    return 0


if __name__ == "__main__":
    sys.exit(main())
