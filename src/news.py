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
# транслитерируют по-разному: «цены» в начале слова встречается как czen
# (5 986 заголовков), tsen (1 431) и cen (156); один czen терял бы 21% случаев.
#
# Корень начинает слово — границу перед ним ставит `topic_pattern`, — кроме корней
# из `COMPOUND_ROOTS`. Слова, которые начинаются так же, но значат другое, отсекаются
# просмотром в самом корне: cen(?!tr) — не «центр», bank(?!et) — не «банкет».
TOPICS = {
    "ceny": ["cen(?!tr)", "czen(?!tr)", "tsen(?!tr)", "podorozh", "infl", "deshev", "tarif"],
    "dohody": ["zarplat", "dohod", "pensi", "vyplat", "posobi"],
    "zanjatost": ["rabot", "uvol", "vakans", "sokrash", "bezrabot"],
    "proizvodstvo": ["zavod", "predprijati", "proizvodstv", "fabrik", "cekh", "ceh"],
    # «Долго…» — не долг: долго, долгосрочный, долгожданный, а с «в» после «о» —
    # долговременный и долговечный. «Долгов», «долговой» — долг.
    "kredit": ["kredit", "ipotek", "bank(?!et)", "dolg(?!o(?!v)|ovrem|ovech)", "stavk"],
    "torgovlja": ["magazin", "torgov", "rynok", "rynk", "otkry", "zakry"],
}

# Корни, которые законно живут внутри сложных слов: им граница начала слова не ставится.
# Первая починка словаря поставила её всем корням и срезала вместе с мусором слова по теме.
# Решено по словам, которые граница снимала (пересборка 25.09; числа — заголовки корпуса):
# без границы идёт корень, который внутри слова почти всегда по теме. Граница осталась,
# где внутри слова в основном чужое: kredit — дискредитация 426 против авто- и микрокредита
# 154; dohod — ледоход и судоходство 75 против подоходного и сверхдохода 33; pensi —
# компенсировать 113 против предпенсионеров 13; torgov — наркоторговец 57 против
# внешнеторгового 11; otkry, zakry — первооткрыватель, «незакрытые границы»; rabot —
# заработок, разработка, обработка; cen/czen/tsen — оценка, сцена, процент; stavk —
# выставка, поставка; proczentn — стопроцентный. У остальных корней внутри слова меньше
# десятка случаев, и граница им не мешает.
COMPOUND_ROOTS = {
    "zavod",                        # нефтезавод 65, авиазавод 71, автозавод 75, хлебозавод, молокозавод
    "fabrik",                       # птицефабрика 105; «сфабриковать» 6 терпимо
    "proizvodstv",                  # промпроизводство 33, фармпроизводство 10; «воспроизводство» 7 терпимо
    "predprijati",                  # сельхоз-, пром-, микро-, госпредприятие — все 81 по теме
    "bank(?!et)",                   # Центробанк 656, Сбербанк 275, Райффайзенбанк, Газпромбанк
    "(?<!gazo)reguljator",          # мегарегулятор — это ЦБ; газорегуляторный (7) — нет
    "deshev",                       # подешевели и т. п. 355, удешевление 24 — всё о ценах
    "vyplat",                       # невыплата 47, соцвыплаты 50
    "ipotek",                       # соципотека 182, промипотека 6
    "dolg(?!o(?!v)|ovrem|ovech)",   # госдолг 94; «недолговечный» 24 отсекает просмотр
    "magazin",                      # зоомагазин 13, автомагазин 9
    # авторынок 116, потребрынок 6 против Крынки 17 и наркорынка 3 — без границы
    "rynok", "rynk",
}

# Граница начала слова: перед корнем не строчная латинская буква. Этого достаточно,
# потому что ищем по тексту после `transliterate`: она сначала понижает регистр,
# потом переводит кириллицу в латиницу, и буква русского слова там — всегда a-z.
WORD_START = r"(?<![a-z])"


def topic_pattern(roots: list[str]) -> str:
    r"""Регулярка темы — одна на все места, где ищутся темы.

    Корень начинает слово. До 25.09 корни искались как подстроки в любом месте слова,
    и тема ловила чужие слова: `stavk` — «выставку», «отставку», «поставку» и «доставку»,
    это 61% попаданий темы ДКП; `cen` и `czen` — «центр». Корни из `COMPOUND_ROOTS`
    идут без границы: они законно живут внутри сложных слов — нефтезавод, Сбербанк,
    госдолг, подешевели. Корни, уже начинающиеся с `\b` (`\bczb\b`), остаются как есть.
    """
    return "|".join(
        root if root.startswith(r"\b") or root in COMPOUND_ROOTS else WORD_START + root for root in roots
    )


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
    """Читает все jsonl, приводит заголовки к единому алфавиту.

    Результат — это и есть слой `data/news/headlines.parquet`, которого нет
    в репозитории (заголовки чужих изданий, права не наши): его собирают локально
    из архива `scripts/crawl_news.py` и сохраняют `save_datasets`."""
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
        headlines[f"t_{topic}"] = headlines["norm"].str.contains(topic_pattern(roots), regex=True, na=False)

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


def save_datasets(
    headlines: pd.DataFrame | None, features: pd.DataFrame, out_dir: str | Path,
    national: pd.DataFrame | None = None,
) -> None:
    """Сохраняет слои. Сжатие zstd выбрано не из вкуса: без него файл заголовков
    весит 56 МБ и GitHub предупреждает о превышении рекомендуемого порога.
    Колонка с адресом оставлена намеренно — она даёт прослеживаемость каждого
    заголовка до первоисточника, а это прямо работает на воспроизводимость.

    `headlines.parquet` в репозиторий не входит — заголовки чужих изданий, права
    на них не наши; в git идут только `monthly.parquet` и `national.parquet`,
    а этот слой пересобирается локально из архива `scripts/crawl_news.py`.

    `headlines=None` — слой заголовков не переписывается. Так пересобирает признаки
    `scripts/build_news_features.py`: заголовков она не меняет, а их файл записан
    pandas 3, и перезапись из .venv поменяла бы его побайтно при тех же данных.
    `national` — национальный ряд (`national_features`), пишется рядом."""
    out = Path(out_dir)
    if headlines is not None:
        frame = headlines.copy()
        for col in ("region_name", "domain"):
            frame[col] = frame[col].astype("category")
        frame.to_parquet(out / "headlines.parquet", compression="zstd", index=False)
    features.to_parquet(out / "monthly.parquet", index=False)
    if national is not None:
        national.to_parquet(out / "national.parquet", index=False)


# ---------------------------------------------------------------------------
# Национальный уровень
# ---------------------------------------------------------------------------
# Региональные признаки проверялись против муниципальных рядов и дали ноль
# трижды. Но разложение ошибки показало, что весь выигрыш сидит в попадании
# в общероссийское движение, а межрядовая составляющая не двигается ничем.
# Значит проверять новости надо на том же разрешении, где живёт сигнал, —
# на национальном, против федерального агрегата.
#
# Тема денежно-кредитной политики выделена отдельно от `kredit` и `ceny`:
# три месяца массового согласия, найденные задним числом по полному ряду,
# пришлись на разворот и крупные изменения реальной ставки — гипотеза, которую
# стоило проверить: если новости вообще на что-то реагируют, то на это.
# Корни даны в той же транслитерации, которой
# движки сайтов строят слаги: «ЦБ» → czb, «Центробанк» → czentrobank,
# «ключевая» → kljuchev, «инфляция» → infljacz.
NATIONAL_TOPICS = dict(TOPICS) | {
    "dkp": [
        r"\bczb\b", "czentrobank", "kljuchev", "nabiullin", "(?<!gazo)reguljator",
        "stavk", "infljacz", "proczentn",
    ],
}


def national_features(headlines: pd.DataFrame) -> pd.DataFrame:
    """Помесячный национальный ряд: интенсивность и доли тем по всему корпусу.

    Интенсивность считается так же, как в региональной версии, — z-оценкой
    числа публикаций внутри издания, усреднённой по изданиям. Нормировка внутри
    издания нужна и здесь: восемнадцать изданий отличаются объёмом на порядок,
    и без неё «интенсивность по стране» была бы интенсивностью самого крупного.

    Доли тем считаются от всех публикаций месяца сразу, а не усреднением
    региональных долей: усреднение дало бы каждому региону равный вес
    независимо от числа публикаций, а нас интересует, о чём писали в стране.

    Оговорка, которая идёт в отчёт: корпус покрывает восемнадцать регионов,
    а не всю страну. «Национальный» здесь означает «сводный по собранному
    корпусу», и федеральной репрезентативности у него нет.
    """
    if headlines.empty:
        return pd.DataFrame()

    frame = headlines.copy()
    if "norm" not in frame:
        frame["norm"] = frame["title"].map(repair_encoding).map(transliterate)
    if "month" not in frame:
        frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M")

    topic_cols = []
    for topic, roots in NATIONAL_TOPICS.items():
        column = f"t_{topic}"
        frame[column] = frame["norm"].str.contains(topic_pattern(roots), regex=True, na=False)
        topic_cols.append(column)

    by_outlet = frame.groupby(["domain", "month"], observed=True).size().rename("n").reset_index()
    stats = by_outlet.groupby("domain", observed=True)["n"].agg(["mean", "std"])
    by_outlet = by_outlet.join(stats, on="domain")
    by_outlet["z"] = (
        (by_outlet["n"] - by_outlet["mean"]) / by_outlet["std"].replace(0, pd.NA)
    ).fillna(0.0)

    out = frame.groupby("month", observed=True).agg(
        n_articles=("url", "size"),
        n_outlets=("domain", "nunique"),
        **{c: (c, "mean") for c in topic_cols},
    )
    out["intensity"] = by_outlet.groupby("month", observed=True)["z"].mean()
    return out.reset_index()
