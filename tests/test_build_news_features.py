"""Пересборка новостных признаков: `scripts/build_news_features.py`.

Всё во временном каталоге: заголовки синтетические, «старые» файлы признаков
собираются прежним словарём, как лежали до 25.09. Живые `data/news/` и `results/`
тесты не трогают.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import news  # noqa: E402

# scripts/ — не пакет: скрипт грузится по пути.
_spec = importlib.util.spec_from_file_location("build_news_features", ROOT / "scripts" / "build_news_features.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

# Доля ДКП по месяцам: прежним словарём 2/3, 1, 2/3 («выставка», «отставка», «поставка»
# шли в тему), новым — 1/3, 1/3, 2/3. Ряды меняются от месяца к месяцу, и их корреляция
# определена: −0,5.
ROWS = [
    ("https://a.ru/1", "ЦБ повысил ключевую ставку", "2024-01-10", "Область А", "a.ru"),
    ("https://a.ru/2", "Выставка цветов в центре города", "2024-01-12", "Область А", "a.ru"),
    ("https://b.ru/1", "Цены на бензин выросли", "2024-01-15", "Область Б", "b.ru"),
    ("https://b.ru/2", "Отставка министра", "2024-02-03", "Область Б", "b.ru"),
    ("https://a.ru/3", "Поставка газа в Европу", "2024-02-05", "Область А", "a.ru"),
    ("https://b.ru/3", "Инфляция замедлилась", "2024-02-20", "Область Б", "b.ru"),
    ("https://a.ru/4", "Регулятор снизил ставку", "2024-03-05", "Область А", "a.ru"),
    ("https://b.ru/4", "Набиуллина выступила", "2024-03-12", "Область Б", "b.ru"),
    ("https://a.ru/5", "Погода в выходные", "2024-03-20", "Область А", "a.ru"),
]


def corpus() -> pd.DataFrame:
    """Заголовки в раскладке `headlines.parquet`: регион и издание — категориями."""
    frame = pd.DataFrame(ROWS, columns=["url", "title", "date", "region_name", "domain"])
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("region_name", "domain"):
        frame[column] = frame[column].astype("category")
    return frame


def legacy_features(headlines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Региональные и национальные признаки прежним словарём: корни — подстроки."""
    legacy = builder.LEGACY_TOPICS
    with mock.patch.object(news, "TOPICS", {k: v for k, v in legacy.items() if k != "dkp"}), \
            mock.patch.object(news, "NATIONAL_TOPICS", legacy), \
            mock.patch.object(news, "topic_pattern", lambda roots: "|".join(roots)):
        return news.monthly_features(builder.prepared(headlines)), news.national_features(headlines)


class DictionaryTableTest(unittest.TestCase):
    def test_counts_share_and_removed_words(self):
        norm = pd.Series(["vystavka cvetov", "czb snizil stavku", "postavka gaza", "vystavka i stavka"])
        dkp = builder.dictionary_table(norm).set_index("тема").loc["dkp"]
        self.assertEqual((dkp["заголовков до"], dkp["заголовков после"], dkp["снято"], dkp["добавлено"]),
                         (4, 2, 2, 0))
        self.assertAlmostEqual(dkp["доля снятого, %"], 50.0)
        # Слово считается по заголовкам: «vystavka» снята и там, где заголовок остался в теме.
        self.assertEqual(dkp["частые снятые слова"], "vystavka 2; postavka 1")


class MainTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.news_dir = self.root / "data" / "news"
        self.news_dir.mkdir(parents=True)
        (self.root / "results").mkdir()
        self.backup = self.root / "results" / "backup_2026-09-25"
        headlines = corpus()
        headlines.to_parquet(self.news_dir / "headlines.parquet", index=False)
        monthly, national = legacy_features(headlines)
        monthly.to_parquet(self.news_dir / "monthly.parquet", index=False)
        national.to_parquet(self.news_dir / "national.parquet", index=False)

    def files(self) -> dict[str, bytes]:
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def run_main(self, *args: str) -> str:
        out = io.StringIO()
        with mock.patch.object(builder, "ROOT", self.root), contextlib.redirect_stdout(out):
            builder.main(list(args))
        return out.getvalue()

    def test_dry_run_prints_everything_and_writes_nothing(self):
        before = self.files()
        out = self.run_main("--dry-run")
        self.assertEqual(self.files(), before)
        self.assertIn("корреляция старого и нового месячного ряда ДКП", out)
        self.assertIn("старые доли тем воспроизводятся старым словарём: да", out)

    def test_run_backs_up_old_files_and_rebuilds_them(self):
        before = self.files()
        self.run_main()
        for name in ("monthly.parquet", "national.parquet"):
            with self.subTest(name):
                self.assertEqual((self.backup / name).read_bytes(), before[f"data/news/{name}"])
        self.assertEqual((self.news_dir / "headlines.parquet").read_bytes(), before["data/news/headlines.parquet"])

        national = pd.read_parquet(self.news_dir / "national.parquet")
        old = pd.read_parquet(self.backup / "national.parquet")
        self.assertEqual(list(national.columns), list(old.columns))
        self.assertEqual(national.dtypes.to_dict(), old.dtypes.to_dict())
        # «Выставка», «отставка» и «поставка» из темы ДКП сняты.
        self.assertEqual(national["t_dkp"].tolist(), [1 / 3, 1 / 3, 2 / 3])
        self.assertEqual(old["t_dkp"].tolist(), [2 / 3, 1.0, 2 / 3])

        monthly = pd.read_parquet(self.news_dir / "monthly.parquet")
        old_monthly = pd.read_parquet(self.backup / "monthly.parquet")
        self.assertEqual(monthly.dtypes.to_dict(), old_monthly.dtypes.to_dict())

        table = pd.read_csv(self.root / "results" / "news_dictionary.csv")
        self.assertEqual(table["тема"].tolist(), list(news.NATIONAL_TOPICS))

    def test_rerun_compares_with_the_original_not_with_the_previous_rebuild(self):
        # После первой пересборки в data/news лежит уже она, а исходный файл — в копии.
        # «До» в таблице и проверка воспроизведения должны по-прежнему относиться к исходному.
        before = self.files()
        self.run_main()
        first = pd.read_csv(self.root / "results" / "news_dictionary.csv")
        out = self.run_main()
        second = pd.read_csv(self.root / "results" / "news_dictionary.csv")
        self.assertIn("старые доли тем воспроизводятся старым словарём: да", out)
        pd.testing.assert_frame_equal(second, first)
        self.assertAlmostEqual(second.set_index("тема").loc["dkp", "корреляция рядов"], -0.5)
        for name in ("monthly.parquet", "national.parquet"):
            with self.subTest(name):
                self.assertEqual((self.backup / name).read_bytes(), before[f"data/news/{name}"])

    def test_existing_backup_is_not_overwritten(self):
        # Копия от прошлого запуска — исходный файл, а в data/news лежит уже другой:
        # перезапиши сборщик копию, байты бы разошлись.
        self.backup.mkdir(parents=True)
        earlier = (self.news_dir / "national.parquet").read_bytes()
        (self.backup / "national.parquet").write_bytes(earlier)
        current = pd.read_parquet(self.news_dir / "national.parquet")
        current["t_dkp"] = 0.0
        current.to_parquet(self.news_dir / "national.parquet", index=False)
        self.run_main()
        self.assertEqual((self.backup / "national.parquet").read_bytes(), earlier)
        self.assertTrue((self.backup / "monthly.parquet").exists())

    def test_columns_other_than_before_stop_before_any_write(self):
        national = pd.read_parquet(self.news_dir / "national.parquet")
        national["t_extra"] = 0.0
        national.to_parquet(self.news_dir / "national.parquet", index=False)
        before = self.files()
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertEqual(self.files(), before)


if __name__ == "__main__":
    unittest.main()
