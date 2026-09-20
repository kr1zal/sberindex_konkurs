"""Превращение собранных заголовков в помесячные признаки по регионам.

Три проблемы решаются здесь, и каждая иначе сломала бы сигнал.

**Разный алфавит.** Издания с HTML-архивами дают заголовки кириллицей, издания
с XML-картами — транслитом из слага адреса («sily pvo sbili desjat bpla»).
Поиск по русским словам нашёл бы только половину корпуса. Поэтому всё приводится
к латинице по той же схеме, по которой движки сайтов строят слаги, и ключевые
слова ищутся уже в ней.

**Перекос по изданиям.** Одно издание дало 190 тысяч заголовков из 560 —
треть корпуса, причём именно то, у которого самая слабая привязка к региону.
В абсолютных числах его регион перевесил бы всё остальное, а сигнал отражал бы
не местную экономику, а общероссийскую повестку в изложении одной редакции.
Поэтому интенсивность считается как отклонение от собственного среднего издания,
а не как число публикаций.

**Разная активность редакций.** Издание, публикующее 800 материалов в месяц,
и издание, публикующее 300, несопоставимы в абсолюте по той же причине.
Нормировка внутри издания снимает и это.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "cz",
    "ч": "ch", "ш": "sh", "щ": "shh", "ъ": "", "ы": "y", "ь": "", "э": "e",
    "ю": "ju", "я": "ja",
}

# Темы, за которыми мы следим. Ключи — что это значит экономически, значения —
# корни в латинице, потому что сравнение идёт после транслитерации.
# Корни даны в нескольких вариантах написания намеренно. Движки сайтов
# транслитерируют по-разному: «цены» встречается как tsen (6282 раза),
# czen (2097) и cen (1748). Один вариант ловил бы шестую часть случаев.
TOPICS = {
    "ceny": ["cen", "czen", "tsen", "podorozh", "infl", "deshev", "tarif"],
    "dohody": ["zarplat", "dohod", "pensi", "vyplat", "posobi"],
    "zanjatost": ["rabot", "uvol", "vakans", "sokrash", "bezrabot"],
    "proizvodstvo": ["zavod", "predprijati", "proizvodstv", "fabrik", "cekh", "ceh"],
    "kredit": ["kredit", "ipotek", "bank", "dolg", "stavk"],
    "torgovlja": ["magazin", "torgov", "rynok", "rynk", "otkry", "zakry"],
}


CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")


def repair_encoding(text: str) -> str:
    """Чинит UTF-8, прочитанный как Latin-1.

    Двенадцать изданий из восемнадцати отдают UTF-8, не объявляя charset,
    и requests по спецификации HTTP откатывается на ISO-8859-1. Кириллица
    приходит в виде 'Ð\x94Ð¾Ñ\x87Ñ\x8c' вместо 'Дочь' — молча, без ошибки,
    и поиск по словам на таком тексте даёт ноль.

    Признак порчи — не форма строки, а результат починки: пробуем преобразование
    и принимаем его, только если в тексте появилась кириллица, которой не было.
    Попытка распознать порчу по набору символов уже подвела: между искажёнными
    байтами стоят управляющие, и диапазонная регулярка их не ловила.

    Преобразование обратимо, поэтому перекачивать 284 тысячи собранных
    заголовков не требуется — чиним при загрузке.
    """
    if CYRILLIC_RE.search(text):
        return text
    try:
        candidate = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return candidate if CYRILLIC_RE.search(candidate) else text


def transliterate(text: str) -> str:
    """Кириллица в латиницу по схеме, которой движки сайтов строят слаги."""
    lowered = text.lower()
    return "".join(TRANSLIT.get(ch, ch) for ch in lowered)


def load_headlines(raw_dir: str | Path) -> pd.DataFrame:
    """Читает все jsonl, приводит заголовки к единому алфавиту."""
    rows = []
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("url")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["title"] = df["title"].map(repair_encoding)
    df["norm"] = df["title"].map(transliterate)
    return df


def monthly_features(headlines: pd.DataFrame) -> pd.DataFrame:
    """Признаки регион × месяц: интенсивность и доли тем.

    Интенсивность — z-оценка числа публикаций внутри издания. Ноль означает
    «издание писало столько же, сколько обычно», а не «ничего не происходило».
    Доли тем считаются от числа публикаций месяца, поэтому от активности редакции
    не зависят вовсе.
    """
    if headlines.empty:
        return pd.DataFrame()

    for topic, roots in TOPICS.items():
        pattern = "|".join(roots)
        headlines[f"t_{topic}"] = headlines["norm"].str.contains(pattern, regex=True, na=False)

    topic_cols = [f"t_{t}" for t in TOPICS]
    grouped = headlines.groupby(["domain", "region_name", "month"])
    agg = grouped.agg(n_articles=("url", "size"), **{c: (c, "mean") for c in topic_cols})
    agg = agg.reset_index()

    # нормировка внутри издания: снимает и перекос по объёму, и разницу активности
    stats = agg.groupby("domain")["n_articles"].agg(["mean", "std"])
    agg = agg.join(stats, on="domain")
    agg["intensity"] = ((agg["n_articles"] - agg["mean"]) / agg["std"].replace(0, pd.NA)).fillna(0.0)

    # если регион закрыт несколькими изданиями, усредняем их между собой
    out = agg.groupby(["region_name", "month"]).agg(
        intensity=("intensity", "mean"),
        n_articles=("n_articles", "sum"),
        n_outlets=("domain", "nunique"),
        **{c: (c, "mean") for c in topic_cols},
    )
    return out.reset_index()
