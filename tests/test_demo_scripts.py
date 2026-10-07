"""Скрипты стенда: ранжирование поиска и ключ изломов в легенде графика.

Поиск стенда (`demo/demo.js::searchMunicipalities`) и поиск на обложке главной
(`site/landing-lib.js::searchRows`) — две копии одного правила, и подсказки у них обязаны совпадать.
Файл стенда целиком загружается в пустом контексте с заглушками страницы: настоящий код, а не его
копия в тесте. Без node проверки поиска пропускаются.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUERIES = [
    "город орёл", "орёл", "городской округ город орёл", "казань", "город казань", "михайловский",
    "новосибирск", "район ардатов", "ленинский район", "санкт-петербург", "город", "округ горелово",
    "ё", "  орел  ", "татарстан казань", "ъъъ", "",
    "каз", "казан", "ардатов", "татарстан", "свердловская", "екатеринбург", "орл",
]


def run_both(queries: list[str], limit: int) -> dict:
    """Результаты обоих поисков на настоящем списке рядов: id найденных строк и их число. Быстрые
    кнопки (`index.quick`) передаются обоим: порядок подсказок зависит и от них."""
    script = """
        const vm = require('vm'), fs = require('fs');
        const [demoPath, libPath, indexPath, payload] = process.argv.slice(1);
        const input = JSON.parse(payload);
        const noop = () => {};
        const page = {
          document: { getElementById: () => null, addEventListener: noop },
          window: {}, console: { error: noop }, LineChart: {},
          fetch: () => Promise.reject(new Error('в тесте нет сети')),
        };
        const stand = vm.runInNewContext(
          fs.readFileSync(demoPath, 'utf8') + '\\n({searchMunicipalities, buildSearchIndex, regionLabel});', page);
        const lib = vm.runInNewContext(fs.readFileSync(libPath, 'utf8') + '\\nLandingLib;', {});
        const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
        const standRows = stand.buildSearchIndex(index.series, index.quick);
        const homeRows = lib.rowsFromIndex(index.series, index.quick);
        const out = {};
        input.queries.forEach((query) => {
          const a = stand.searchMunicipalities(query, standRows, input.limit);
          const b = lib.searchRows(homeRows, query, input.limit);
          out[query] = { stand: a.matches.map((r) => r.seriesId), home: b.matches.map((r) => r.id),
                         standTotal: a.total, homeTotal: b.total };
        });
        out.labels = standRows.map((row, i) => [stand.regionLabel(row), lib.regionLabel(homeRows[i])]);
        console.log(JSON.stringify(out));
    """
    done = subprocess.run(
        ["node", "-e", script, str(ROOT / "demo" / "demo.js"), str(ROOT / "site" / "landing-lib.js"),
         str(ROOT / "demo" / "data" / "index.json"), json.dumps({"queries": queries, "limit": limit})],
        capture_output=True, text=True, check=True, timeout=120,
    )
    return json.loads(done.stdout)


@unittest.skipUnless(shutil.which("node"), "для скриптов стенда нужен node")
class SearchRankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.small = run_both(QUERIES, 8)
        cls.full = run_both(QUERIES, 100000)

    def test_stand_and_cover_search_return_the_same_suggestions(self) -> None:
        for query in QUERIES:
            with self.subTest(query=query):
                self.assertEqual(self.small[query]["stand"], self.small[query]["home"])
                self.assertEqual(self.small[query]["standTotal"], self.small[query]["homeTotal"])
                # Не только первые подсказки: весь порядок результатов.
                self.assertEqual(self.full[query]["stand"], self.full[query]["home"])

    def test_every_word_starting_a_name_word_goes_before_a_plain_inclusion(self) -> None:
        got = self.small["город орёл"]["stand"]
        self.assertEqual(got[0], "городской округ город Орёл")
        # «орел» внутри «Горелово» — вхождение, а не начало слова: оно ниже Орла.
        gorelovo = [i for i in self.full["город орёл"]["stand"] if "Горелово" in i]
        self.assertEqual(len(gorelovo), 1)
        self.assertGreater(self.full["город орёл"]["stand"].index(gorelovo[0]), 0)

    def test_kazan_goes_first_on_both_pages(self) -> None:
        # «каз», «казан» и Enter без выбора открывали «Казанский муниципальный район» Тюменской области:
        # внутри яруса сортировка была по алфавиту, и Казань стояла четвёртой. Теперь Казань первая и там,
        # и там — ряд быстрого выбора, а за ним по длине совпавшего слова: Казанский, Казачинский, …
        kazan = "городской округ город Казань"
        for query in ("каз", "казан", "казань", "город казань"):
            for page in ("stand", "home"):
                with self.subTest(query=query, page=page):
                    self.assertEqual(self.small[query][page][0], kazan)
        self.assertEqual(self.small["казан"]["stand"], [kazan, "Казанский муниципальный район"])
        self.assertEqual(self.small["каз"]["stand"][:4], [
            kazan, "Казанский муниципальный район", "Казачинский муниципальный район",
            "Казачинско-Ленский муниципальный район"])

    def test_quick_cities_go_before_the_other_matches(self) -> None:
        # Города быстрого выбора, подошедшие под запрос, — первыми, а не по алфавиту внутри яруса.
        self.assertEqual(self.small["новосибирск"]["stand"][:2], [
            "городской округ город Новосибирск", "Новосибирский муниципальный район"])
        self.assertEqual(self.small["город"]["stand"][:4], [
            "городской округ город Екатеринбург", "городской округ город Казань",
            "городской округ город Новосибирск", "городской округ город Орёл"])
        # Запрос по региону: самый известный город региона — первым.
        self.assertEqual(self.small["татарстан"]["stand"][0], "городской округ город Казань")
        self.assertEqual(self.small["свердловская"]["stand"][0], "городской округ город Екатеринбург")

    def test_previous_orders_do_not_change(self) -> None:
        self.assertEqual(self.small["казань"]["stand"], ["городской округ город Казань"])
        self.assertEqual(self.small["екатеринбург"]["stand"], ["городской округ город Екатеринбург"])
        # Без ряда быстрого выбора ярусы работают как раньше: название, начинающееся с запроса,
        # выше слова в середине названия, а ряды с одинаковым совпавшим словом идут по алфавиту.
        self.assertEqual(self.small["ленинский район"]["stand"][:2], [
            "Ленинский муниципальный район #1", "Ленинский муниципальный район #2"])
        self.assertEqual(self.small[""]["stand"], [])
        self.assertEqual(self.small["ъъъ"]["stand"], [])

    def test_both_pages_label_regions_the_same_way(self) -> None:
        # Подпись региона в подсказке — одна и та же строка на главной и на стенде; у ряда без региона
        # рядом различитель из генератора («регион не определён · 24,3 тыс. ₽ в мес.»).
        labels = self.small["labels"]
        self.assertTrue(all(stand == home for stand, home in labels))
        index = json.loads((ROOT / "demo" / "data" / "index.json").read_text(encoding="utf-8"))
        for (series_id, region, _oktmo, _file, *mean), (stand, _home) in zip(index["series"], labels):
            if region:
                self.assertEqual(stand, region)
            else:
                self.assertEqual(len(mean), 1, f"у ряда без региона {series_id!r} нет различителя")
                self.assertEqual(stand, f"регион не определён · {mean[0]}\u00a0тыс.\u00a0₽ в\u00a0мес.")

    def test_suggestion_list_scrolls_into_view_on_the_stand_like_on_the_cover(self) -> None:
        # На телефоне список подсказок стенда уходил за низ экрана на несколько строк и не докручивался.
        demo = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        body = re.search(r"function openListbox\(\) \{.*?\n  \}\n", demo, re.S).group(0)
        self.assertIn('listbox.scrollIntoView({ block: "nearest" })', body)


class MarkerKeyInLegendTest(unittest.TestCase):
    """Вертикальные штриховые линии на графике МО — изломы; ключ в легенде говорит об этом."""

    def test_chart_spec_asks_for_the_break_key_and_the_chart_draws_it(self) -> None:
        demo = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        spec = re.search(r"function buildMoChartSpec.*?\n}\n", demo, re.S).group(0)
        self.assertRegex(spec, r'markerKeys:\s*\[\{\s*kind:\s*"break",\s*label:\s*"изломы"\s*\}\]')
        chart = (ROOT / "demo" / "linechart.js").read_text(encoding="utf-8")
        legend = re.search(r"function renderLegend\(\) \{.*?\n    \}\n", chart, re.S).group(0)
        # Ключ есть только у вида отметки, который на графике нарисован, и берёт тот же цвет.
        self.assertIn("markerKeys", legend)
        self.assertIn("markers", legend)
        self.assertIn("chart-legend-key-marker-", legend)

    def test_key_has_the_dash_and_the_colour_of_the_break_line(self) -> None:
        css = (ROOT / "demo" / "demo.css").read_text(encoding="utf-8")
        key = re.search(r"\.chart-legend-key-marker-break \{([^}]*)\}", css).group(1)
        self.assertRegex(key, r"--legend-color:\s*var\(--chart-break\)")
        self.assertRegex(key, r"border-top-style:\s*dashed")
        line = re.search(r"\.chart-marker-break \{([^}]*)\}", css).group(1)
        self.assertRegex(line, r"stroke:\s*var\(--chart-break\)")
        self.assertRegex(line, r"stroke-dasharray")


if __name__ == "__main__":
    unittest.main()
