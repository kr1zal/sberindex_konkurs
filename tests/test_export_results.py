"""Снимок крупных результатов: `scripts/export_results.py`.

Файлы — синтетические байты во временном каталоге, живые `results/*.csv` тесты
не читают и не пишут. Проверяется то, ради чего скрипт существует: два экспорта
одного и того же CSV дают побайтово одинаковый `.csv.gz` (иначе git видел бы
изменения там, где данные те же самые), заголовок gzip без времени и имени файла,
и `--check` ловит и отсутствие снимка, и его расхождение с исходником.
"""
from __future__ import annotations

import gzip
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# scripts/ — не пакет: скрипт грузится по пути, как в tests/test_backfill_r2.py.
_spec = importlib.util.spec_from_file_location("export_results", ROOT / "scripts" / "export_results.py")
export_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_results)

CONTENT = ("mo,fold,mae\n" + "".join(f"мо_{i},{i % 3},{1000 + i}.{i:03d}\n" for i in range(500))).encode("utf-8")


class ExportOneTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.source = self.dir / "per_series.csv"
        self.source.write_bytes(CONTENT)

    def test_decompressed_bytes_match_source(self):
        target = export_results.export_one(self.source)
        self.assertEqual(target, self.dir / "per_series.csv.gz")
        with gzip.open(target, "rb") as gz:
            self.assertEqual(gz.read(), CONTENT)

    def test_two_exports_of_unchanged_csv_are_byte_identical(self):
        # Условие, ради которого детерминизм нужен: CSV не менялся между прогонами —
        # повторный экспорт не должен показывать git несуществующее изменение.
        first = export_results.export_one(self.source).read_bytes()
        second = export_results.export_one(self.source).read_bytes()
        self.assertEqual(first, second)

    def test_no_temp_file_left_behind(self):
        export_results.export_one(self.source)
        leftovers = list(self.dir.glob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_temp_file_is_removed_when_verification_fails(self):
        # Порча при проверке (диск, прерывание) не должна оставлять .tmp рядом с целью:
        # запись и сверка идут в try/finally, снимающем .tmp при любом исключении.
        fake_readback = mock.MagicMock()
        fake_readback.__enter__.return_value.read.return_value = b"corrupted"
        with mock.patch.object(export_results.gzip, "open", return_value=fake_readback):
            with self.assertRaises(ValueError):
                export_results.export_one(self.source)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])
        self.assertFalse((self.dir / "per_series.csv.gz").exists())  # цель тоже не подменена мусором

    def test_temp_file_is_removed_when_write_fails(self):
        # Сбой посреди записи (диск кончился) — до сверки дело не доходит, и без try/finally
        # вокруг записи недописанный .tmp оставался рядом с целью: предыдущий тест этого не ловит.
        with mock.patch.object(export_results.gzip.GzipFile, "write", side_effect=OSError("диск")):
            with self.assertRaises(OSError):
                export_results.export_one(self.source)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])
        self.assertFalse((self.dir / "per_series.csv.gz").exists())

    def test_gzip_header_has_no_filename_and_zero_mtime(self):
        # RFC 1952: байт 3 — флаги (бит 0x08 = FNAME), байты 4..7 — MTIME.
        # Оба должны быть пустыми, иначе два экспорта расходятся байтами без причины в данных.
        target = export_results.export_one(self.source)
        header = target.read_bytes()[:10]
        flags, mtime = header[3], header[4:8]
        self.assertEqual(flags & 0x08, 0, "в заголовке осталось имя файла (флаг FNAME)")
        self.assertEqual(mtime, b"\x00\x00\x00\x00")


class CheckOneTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.source = self.dir / "per_series.csv"
        self.source.write_bytes(CONTENT)

    def test_matching_snapshot_passes(self):
        export_results.export_one(self.source)
        self.assertIsNone(export_results.check_one(self.source))

    def test_missing_snapshot_is_reported(self):
        problem = export_results.check_one(self.source)
        self.assertIsNotNone(problem)
        self.assertIn("снимка нет", problem)

    def test_missing_source_is_reported(self):
        export_results.export_one(self.source)
        self.source.unlink()
        problem = export_results.check_one(self.source)
        self.assertIsNotNone(problem)
        self.assertIn("исходника нет", problem)

    def test_stale_snapshot_after_source_changes_is_reported(self):
        export_results.export_one(self.source)
        self.source.write_bytes(CONTENT + "мо_500,0,9999.000\n".encode("utf-8"))
        problem = export_results.check_one(self.source)
        self.assertIsNotNone(problem)
        self.assertIn("разошёлся", problem)


class MainCliTest(unittest.TestCase):
    """`main()` — та же пара функций поверх `SOURCES`/`ROOT`, только через argparse;
    оба подменяются, чтобы не трогать настоящие results/."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "results").mkdir()
        self.source = self.root / "results" / "per_series.csv"
        self.source.write_bytes(CONTENT)
        self.sources_patch = mock.patch.object(export_results, "SOURCES", (Path("results/per_series.csv"),))
        self.root_patch = mock.patch.object(export_results, "ROOT", self.root)
        self.sources_patch.start()
        self.root_patch.start()
        self.addCleanup(self.sources_patch.stop)
        self.addCleanup(self.root_patch.stop)

    def run_main(self, *argv: str) -> int:
        with mock.patch.object(sys, "argv", ["export_results.py", *argv]):
            return export_results.main()

    def test_export_then_check_succeeds(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.run_main("--check"), 0)

    def test_check_without_export_fails(self):
        self.assertEqual(self.run_main("--check"), 1)

    def test_check_after_source_changes_fails(self):
        self.run_main()
        self.source.write_bytes(CONTENT + "мо_500,0,9999.000\n".encode("utf-8"))
        self.assertEqual(self.run_main("--check"), 1)


class TrackedCsvSizeTest(unittest.TestCase):
    """`.gitignore`: `!results/*.csv` (и `!results/cp_v1/*.csv`) снимает игнор с любого
    CSV верхнего уровня без предела по размеру — крупные файлы обязаны идти только
    снимком `.csv.gz` (`SOURCES`). Сеть на случай, если такой файл всё же закоммитят
    обычным CSV: живой git-индекс, а не список файлов на диске, — git не обязателен
    в каждом окружении, тест пропускается, если его нет."""

    LIMIT_BYTES = 2 * 1024 * 1024

    def test_every_tracked_csv_is_under_two_megabytes(self):
        try:
            done = subprocess.run(
                ["git", "ls-files", "--", "results"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError:
            self.skipTest("git недоступен в этом окружении")
        if done.returncode != 0:
            self.skipTest(f"git ls-files не сработал: {done.stderr.strip()}")

        tracked_csv = [line for line in done.stdout.splitlines() if line.endswith(".csv")]
        self.assertTrue(tracked_csv, "git ls-files results не нашёл ни одного CSV — проверять нечего")

        oversized = []
        for rel in tracked_csv:
            size = (ROOT / rel).stat().st_size
            if size > self.LIMIT_BYTES:
                oversized.append(f"{rel} ({size / (1024 * 1024):.1f} МБ)")
        self.assertEqual(
            oversized, [],
            "отслеживаемые CSV крупнее 2 МБ: снимком через export_results.py — "
            + ", ".join(oversized),
        )


if __name__ == "__main__":
    unittest.main()
