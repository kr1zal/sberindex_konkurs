"""Что теряется на заголовках: тематические признаки по заголовку против тела.

Весь корпус построен на заголовках. Заголовок — пятнадцать слов, и тема в нём
может не назваться: статья про решение по ставке вполне называется «Что будет
с ипотекой». Значит доли тем занижены, и вопрос только в том, насколько.

Считается на выборке из `scripts/fetch_bodies.py` и отвечает на три вопроса:

1. **Насколько вырастает доля попаданий** в каждую тему, если смотреть в тело.
2. **Сколько статей, не попавших в тему по заголовку, попадают по телу** —
   это и есть прямая мера потери.
3. **Меняется ли связь помесячного ряда доли с целевой величиной.** Именно
   ради этого всё и делается: доля сама по себе нас не интересует, интересует
   её способность объяснять движение федерального агрегата.

## Два контроля, без которых третий вопрос читается неверно

**Шум выборки.** Двадцать тысяч из пятисот восьмидесяти восьми — это 3,4%,
и помесячный ряд по выборке шумнее, чем по всему корпусу. Поэтому считается
третий ряд: доли по заголовкам, но на той же выборке. Сравнение «выборка
по заголовкам против всего корпуса по заголовкам» показывает шум выборки,
и на его фоне уже видно, что даёт тело.

**Равный текстовый бюджет, и это важнее.** Первая версия сравнивала доли,
посчитанные по всему телу целиком, и получила поразительный результат: все семь
тем разом перевернули корреляцию с остатком прогноза с отрицательной
на +0,27…+0,62, монотонно растущую с горизонтом. Семь несвязанных тем не могут
вести себя одинаково — и не вели. Средняя длина тела в выборке гуляет по месяцам
от 2 613 до 3 421 символа и сама коррелирует с остатком на −0,30, −0,49 и −0,60
по горизонтам. Длиннее текст — больше шансов попасть в любую тему, и вся находка
была длиной текста, а не темой.

Поэтому тело обрезается до `--budget` символов, а статьи с телом короче бюджета
из сравнения исключаются. После этого каждая статья вносит ровно один и тот же
объём текста, и месяцы становятся сравнимы.

    .venv/bin/python -u scripts/compare_headline_body.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.external import CANDIDATES, choose_model, load_aggregate  # noqa: E402
from src.news import NATIONAL_TOPICS, repair_encoding, topic_pattern, transliterate  # noqa: E402

BODIES = ROOT / "data" / "news" / "bodies_sample.jsonl"
HORIZONS = (1, 2, 3)
BUDGET = 1500  # символов тела на статью; всё, что короче, выбывает из сравнения

# Издание, у которого заголовки восстановлены из URL-слагов, а не взяты со страницы:
# в корпусе они получаются транслитом и обрывочными. На таких «заголовках» тело
# выигрывает по определению, а не потому, что тема не называется в заголовке.
# Плюс это треть всей выборки — региональное агентство публикует вдесятеро больше
# остальных. Считается отдельно, чтобы не выдать дефект одного издания за вывод.
SLUG_TITLES = "riamo.ru"


def read_jsonl(path: Path) -> pd.DataFrame:
    """Построчное чтение. `splitlines()` здесь нельзя: он режет и по юникодным
    разделителям строк, которых в текстах статей достаточно, и ломает записи."""
    rows = [json.loads(line) for line in path.read_bytes().split(b"\n") if line.strip()]
    return pd.DataFrame(rows)


def topic_hits(text: pd.Series) -> pd.DataFrame:
    """Попадания в темы тем же паттерном, что у признаков корпуса (`topic_pattern`):
    иначе заголовки и тела сравнивались бы разными словарями."""
    normalised = text.fillna("").map(repair_encoding).map(transliterate)
    return pd.DataFrame({
        topic: normalised.str.contains(topic_pattern(roots), regex=True, na=False)
        for topic, roots in NATIONAL_TOPICS.items()
    })


def residuals_of_aggregate(months: pd.PeriodIndex) -> dict[int, pd.Series]:
    """Остаток сезонного прогноза федерального агрегата — та самая целевая величина."""
    aggregate = load_aggregate(ROOT / "data/reference/sberindex")
    log_series = np.log(aggregate)
    periods, values = list(log_series.index), log_series.to_numpy(dtype=float)
    model_name, _ = choose_model(values, max(HORIZONS))
    out: dict[int, pd.Series] = {}
    for h in HORIZONS:
        data = {}
        for position, period in enumerate(periods):
            if period not in months or position < 25 or position + h >= len(periods):
                continue
            predicted = CANDIDATES[model_name](values[: position + 1], max(HORIZONS))
            data[period] = float(values[position + h] - predicted[h - 1])
        out[h] = pd.Series(data).sort_index()
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    if not BODIES.exists():
        print(f"нет {BODIES.relative_to(ROOT)} — сначала scripts/fetch_bodies.py")
        return 1

    raw = read_jsonl(BODIES)
    raw["month"] = pd.PeriodIndex(raw["month"], freq="M")
    raw["body"] = raw["body"].fillna("")
    raw["blen"] = raw["body"].str.len()
    print(f"скачано: {len(raw)} статей, {raw['month'].nunique()} месяцев, "
          f"{raw['domain'].nunique()} изданий")
    print(f"тел короче 100 символов: {int((raw['blen'] < 100).sum())} "
          f"({(raw['blen'] < 100).mean() * 100:.1f}%)")
    print(f"медианная длина тела {int(raw['blen'].median())} символов "
          f"против {int(raw['title'].str.len().median())} у заголовка")

    # Контроль: сама длина тела не должна объяснять целевую величину лучше тем.
    by_month_len = raw.groupby("month")["blen"].mean()
    print(f"средняя длина тела по месяцам: от {by_month_len.min():.0f} "
          f"до {by_month_len.max():.0f} — разброс {by_month_len.std():.0f}")

    share = (raw["domain"] == SLUG_TITLES).mean()
    print(f"доля {SLUG_TITLES} в выборке: {share * 100:.0f}% — считается отдельно, "
          f"у него заголовки восстановлены из слагов")
    raw = raw.loc[raw["domain"] != SLUG_TITLES].copy()
    print(f"в анализе остаётся {len(raw)} статей, {raw['domain'].nunique()} изданий")

    sample = raw.loc[raw["blen"] >= BUDGET].copy()
    sample["body"] = sample["body"].str[:BUDGET]
    kept = sample.groupby("month").size()
    print(f"\nв сравнение идут статьи с телом от {BUDGET} символов: {len(sample)} "
          f"({len(sample) / len(raw) * 100:.0f}%), по месяцам от {kept.min()} до {kept.max()}")
    total = len(sample)

    by_title = topic_hits(sample["title"])
    by_body = topic_hits(sample["title"] + " " + sample["body"])

    print("ВОПРОС 1-2. Попадания в тему по заголовку и по телу")
    print(f"{'тема':16s} {'по заголовку':>13s} {'с телом':>9s} {'рост':>7s} "
          f"{'добрано статей':>15s} {'из не попавших':>15s}")
    rows = []
    for topic in NATIONAL_TOPICS:
        t_hit, b_hit = by_title[topic], by_body[topic]
        added = int((~t_hit & b_hit).sum())
        missed_before = int((~t_hit).sum())
        rows.append({
            "тема": topic,
            "доля по заголовку": t_hit.mean(),
            "доля с телом": b_hit.mean(),
            "рост, раз": b_hit.mean() / t_hit.mean() if t_hit.mean() else np.nan,
            "добрано": added,
            "доля добранных из непопавших": added / missed_before if missed_before else np.nan,
        })
        print(f"{topic:16s} {t_hit.mean():13.4f} {b_hit.mean():9.4f} "
              f"{b_hit.mean() / t_hit.mean() if t_hit.mean() else float('nan'):6.1f}× "
              f"{added:15d} {added / missed_before * 100 if missed_before else float('nan'):14.1f}%")

    print("\n\nВОПРОС 3. Связь помесячного ряда доли с остатком прогноза агрегата")
    full = pd.read_parquet(ROOT / "data/news/national.parquet")
    full["month"] = pd.PeriodIndex(full["month"], freq="M")
    full = full.set_index("month").sort_index()

    monthly_title = by_title.groupby(sample["month"]).mean()
    monthly_body = by_body.groupby(sample["month"]).mean()
    months = monthly_title.index
    residuals = residuals_of_aggregate(months)

    residual_len = {h: by_month_len.reindex(residuals[h].index).corr(residuals[h])
                    for h in HORIZONS}
    print("сначала контроль — корреляция самой средней длины тела с остатком:")
    print("  " + " | ".join(f"h={h}: {residual_len[h]:+.3f}" for h in HORIZONS))
    print("  если доли тем коррелируют примерно так же и все в одну сторону,")
    print("  то найдена длина текста, а не тема.\n")
    print("столбцы: корреляция помесячного ряда доли с остатком, по горизонтам\n")
    print(f"{'тема':16s} {'источник':26s} " + "".join(f"{'h=' + str(h):>8s}" for h in HORIZONS))
    correlations = []
    for topic in NATIONAL_TOPICS:
        series = {
            "весь корпус, заголовки": full[f"t_{topic}"].reindex(months),
            "выборка, заголовки": monthly_title[topic],
            "выборка, заголовки + тело": monthly_body[topic],
        }
        for label, values in series.items():
            line = f"{topic if label.startswith('весь') else '':16s} {label:26s} "
            record = {"тема": topic, "источник": label}
            for h in HORIZONS:
                common = values.index.intersection(residuals[h].index)
                rho = values.reindex(common).corr(residuals[h].reindex(common))
                record[f"corr_h{h}"] = rho
                line += f"{rho:8.3f}"
            correlations.append(record)
            print(line)
        print()

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "headline_vs_body.csv", index=False)
    pd.DataFrame(correlations).to_csv(ROOT / "results" / "headline_vs_body_corr.csv", index=False)

    print("\nСКОЛЬКО СТОИТ ПОЛНАЯ ВЫКАЧКА")
    rate = total / 18  # статей на издание в этой выборке
    print(f"  выборка {total} статей заняла около {rate * 1.3 / 60:.0f} мин при паузе 1 с")
    print(f"  весь корпус — 588 586 статей, то есть примерно "
          f"{588586 / 18 * 1.3 / 3600:.1f} ч в том же режиме")
    print("\nсохранено: results/headline_vs_body.csv, results/headline_vs_body_corr.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
