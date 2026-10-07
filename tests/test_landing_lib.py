"""Чистые помощники главной (`site/landing-lib.js`): их запускает node, страница не нужна.

Файл не обращается к `document` и `window`, поэтому грузится так же, как раскладка подписей
стенда: код читается в пустом контексте, а наружу выходит объект `LandingLib`. Проверяется то,
что на странице видно глазами и что раньше ломалось незаметно: подписи оси при смене шкалы,
итоговый текст счёта чисел, ранжирование поиска, кнопка «Открыть прогноз», ширина истории на
узком графике. Без node проверки пропускаются.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "site" / "landing-lib.js"
INDEX_JSON = ROOT / "demo" / "data" / "index.json"

NBSP = " "


def run_lib(body: str, payload: object = None) -> object:
    """Выполняет `body` с готовым `lib` (LandingLib) и `input` (payload); результат — JSON-значение,
    которое `body` отдаёт через `return`."""
    script = (
        "const vm = require('vm'), fs = require('fs');"
        "const lib = vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8') + '\\nLandingLib;', {});"
        "const input = JSON.parse(process.argv[2]);"
        "const result = (() => {" + body + "})();"
        "console.log(JSON.stringify(result));"
    )
    done = subprocess.run(
        ["node", "-e", script, str(LIB), json.dumps(payload)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(done.stdout)


@unittest.skipUnless(shutil.which("node"), "для помощников главной нужен node")
class TickLabelsTest(unittest.TestCase):
    def test_whole_steps_give_whole_percent_labels(self) -> None:
        got = run_lib(
            "const s = lib.niceScale(input[0], input[1], 5);"
            "return {step: s.step, labels: s.ticks.map((t) => lib.percentTick(t, s.step))};",
            [0.8, 1.2],
        )
        self.assertEqual(got["labels"], ["80%", "90%", "100%", "110%", "120%"])

    def test_fractional_step_labels_keep_one_decimal(self) -> None:
        # Шаг 2,5% при «целых» подписях дал бы «98% / 100% / 103%» — соседние отметки
        # перестали бы различаться на глаз и расходились бы со шкалой.
        got = run_lib(
            "const s = lib.niceScale(input[0], input[1], 5);"
            "return {step: s.step, labels: s.ticks.map((t) => lib.percentTick(t, s.step))};",
            [1, 1.11],
        )
        self.assertAlmostEqual(got["step"], 0.025)
        self.assertEqual(
            got["labels"], ["100,0%", "102,5%", "105,0%", "107,5%", "110,0%", "112,5%"])
        self.assertEqual(len(set(got["labels"])), len(got["labels"]))

    def test_tick_labels_group_digits_use_comma_and_minus(self) -> None:
        got = run_lib(
            "return [lib.formatTick(4500, 500), lib.formatTick(12000, 1000), lib.formatTick(0.5, 0.5),"
            " lib.formatTick(-5, 1), lib.formatTick(-0.0001, 1), lib.formatTick(7, 2.5)];"
        )
        self.assertEqual(got, [f"4{NBSP}500", f"12{NBSP}000", "0,5", "−5", "0", "7,0"])

    def test_step_decimals(self) -> None:
        got = run_lib("return [10, 2, 1, 0.5, 2.5, 0.25, 0.1].map((step) => lib.stepDecimals(step));")
        self.assertEqual(got, [0, 0, 0, 1, 1, 2, 1])


@unittest.skipUnless(shutil.which("node"), "для помощников главной нужен node")
class TickSetsTest(unittest.TestCase):
    """Ось раздела «Почему это прогноз одного числа»: две быстрые смены шага подряд оставляли
    на оси две шкалы насовсем — «уходящим» помечался каждый раз первый набор, а набор
    предыдущего шага не убирался никогда."""

    SCRIPT = """
        const sets = lib.createTickSets();
        const log = [];
        const snap = (label) => log.push({label, ids: sets.ids(), live: sets.liveIds()});
        const a = sets.show(false); snap('первый набор');
        const b = sets.show(true); snap('шаг 2');
        const c = sets.show(true); snap('шаг 3 раньше, чем ушёл набор шага 1');
        // таймеры уходящих наборов сработали
        b.leaving.concat(c.leaving).forEach((id) => sets.finish(id)); snap('после таймеров');
        const d = sets.show(true); snap('медленный клик');
        d.leaving.forEach((id) => sets.finish(id)); snap('медленный клик, таймер');
        const e = sets.show(false); snap('без анимации');
        return {log, a, b, c, d, e};
    """

    def setUp(self) -> None:
        self.got = run_lib(self.SCRIPT)

    def test_exactly_one_live_set_at_every_moment(self) -> None:
        for entry in self.got["log"]:
            with self.subTest(moment=entry["label"]):
                self.assertEqual(len(entry["live"]), 1)

    def test_after_the_timers_only_the_last_set_stays(self) -> None:
        log = {entry["label"]: entry for entry in self.got["log"]}
        c = self.got["c"]["added"]
        self.assertEqual(log["после таймеров"]["ids"], [c])
        self.assertEqual(log["после таймеров"]["live"], [c])
        # Быстрые клики не оставляют сирот и в следующий раз: медленный клик тоже кончается одним набором.
        self.assertEqual(log["медленный клик, таймер"]["ids"], [self.got["d"]["added"]])

    def test_every_previous_set_leaves_exactly_once(self) -> None:
        a, b, c = (self.got[key] for key in "abc")
        self.assertEqual(b["leaving"], [a["added"]])
        # Набор шага 1 уже уходит: второй раз он не помечается и таймер ему не ставится.
        self.assertEqual(c["leaving"], [b["added"]])
        log = {entry["label"]: entry for entry in self.got["log"]}
        self.assertEqual(log["шаг 3 раньше, чем ушёл набор шага 1"]["ids"], [a["added"], b["added"], c["added"]])

    def test_without_animation_previous_sets_are_removed_at_once(self) -> None:
        # Первая отрисовка и системная настройка «без анимаций»: прежние наборы убираются сразу.
        self.assertEqual(self.got["e"]["removed"], [self.got["d"]["added"]])
        self.assertEqual(self.got["e"]["leaving"], [])
        log = {entry["label"]: entry for entry in self.got["log"]}
        self.assertEqual(log["без анимации"]["ids"], [self.got["e"]["added"]])


@unittest.skipUnless(shutil.which("node"), "для помощников главной нужен node")
class CountedTextTest(unittest.TestCase):
    """Счёт чисел в карточках: на каждом кадре — текст с теми же знаками, в конце — исходный."""

    def tokens(self, text: str) -> list[dict]:
        return run_lib("return lib.numberTokens(input);", text)

    def counted(self, text: str, progress: float) -> str:
        return run_lib("return lib.countedText(input[0], lib.numberTokens(input[0]), input[1]);", [text, progress])

    def test_tokens_cover_decimals_ranges_and_grouped_digits(self) -> None:
        tokens = self.tokens(f"3,8–4,5% и 1{NBSP}581 ₽ и 40%")
        self.assertEqual([t["value"] for t in tokens], [3.8, 4.5, 1581, 40])
        self.assertEqual([t["digits"] for t in tokens], [1, 1, 0, 0])
        self.assertEqual([t["grouped"] for t in tokens], [False, False, True, False])

    def test_final_text_is_the_original_byte_for_byte(self) -> None:
        page = (ROOT / "index.html").read_text(encoding="utf-8")
        numbers = re.findall(r'<p class="stat-number">(.*?)</p>', page)
        self.assertGreaterEqual(len(numbers), 4, "на странице нет чисел карточек")
        samples = numbers + [f"1{NBSP}581 → 1{NBSP}217 ₽", "0,961 → 0,978", "7,0–17,1%", "100,0%"]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(self.counted(text, 1), text)

    def test_start_of_the_count_keeps_the_signs_and_the_words(self) -> None:
        self.assertEqual(self.counted("23,0%", 0), "0,0%")
        self.assertEqual(self.counted("3,8–4,5%", 0), "0,0–0,0%")
        self.assertEqual(self.counted(f"1{NBSP}500 ₽ в месяц", 0), "0 ₽ в месяц")

    def test_middle_of_the_count_scales_every_number_and_keeps_the_text_between(self) -> None:
        self.assertEqual(self.counted(f"10,0% и 2{NBSP}000 ₽", 0.5), f"5,0% и 1{NBSP}000 ₽")
        self.assertEqual(self.counted("текст без чисел", 0.3), "текст без чисел")


@unittest.skipUnless(shutil.which("node"), "для помощников главной нужен node")
class SearchRankingTest(unittest.TestCase):
    """Поиск на обложке: тот же порядок подсказок, что на стенде, и кнопка по набранному запросу."""

    ROWS = [
        ["внутригородская территория города федерального значения муниципальный округ Горелово",
         "Санкт-Петербург", None, 0],
        ["городской округ город Орёл", "Орловская область", None, 1],
        ["Орловский муниципальный округ", "Орловская область", None, 1],
        ["городской округ город Казань", "Республика Татарстан", None, 2],
        ["Михайловский муниципальный район #1", None, None, 3],
        ["Михайловский муниципальный район #2", None, None, 3],
    ]

    def ranked(self, query: str, limit: int = 10) -> list[str]:
        got = run_lib(
            "const rows = lib.rowsFromIndex(input[0]); return lib.searchRows(rows, input[1], input[2]).matches.map((r) => r.id);",
            [self.ROWS, query, limit],
        )
        return got

    def test_every_word_of_the_query_starting_a_name_word_beats_a_plain_inclusion(self) -> None:
        # «орел» внутри слова «Горелово» совпадало по алфавиту раньше, чем сам Орёл.
        got = self.ranked("город орёл")
        self.assertEqual(got[0], "городской округ город Орёл")
        self.assertIn("внутригородская территория города федерального значения муниципальный округ Горелово", got)
        self.assertLess(got.index("городской округ город Орёл"), got.index(self.ROWS[0][0]))

    def test_tiers_name_start_then_word_start_then_inclusion(self) -> None:
        # «город»: названия, которые им начинаются, — первыми; затем название со словом «города»;
        # внутри яруса — по алфавиту.
        self.assertEqual(
            self.ranked("город"),
            ["городской округ город Казань", "городской округ город Орёл", self.ROWS[0][0]],
        )
        # «орёл»: слово «Орёл» начинается с запроса, «Горелово» — только содержит его.
        self.assertEqual(self.ranked("орёл"), ["городской округ город Орёл", self.ROWS[0][0]])
        self.assertEqual(self.ranked("михайловский"),
                         ["Михайловский муниципальный район #1", "Михайловский муниципальный район #2"])
        self.assertEqual(self.ranked("городской округ город орёл"), ["городской округ город Орёл"])

    def test_region_words_and_case_and_yo(self) -> None:
        self.assertEqual(self.ranked("КАЗАНЬ татарстан"), ["городской округ город Казань"])
        self.assertEqual(self.ranked("  орел  "), self.ranked("Орёл"))

    def test_empty_and_unknown_queries_find_nothing(self) -> None:
        self.assertEqual(self.ranked(""), [])
        self.assertEqual(self.ranked("   "), [])
        self.assertEqual(self.ranked("ъъъъ"), [])

    def test_fewer_extra_letters_go_first_inside_a_tier(self) -> None:
        # «бе»: оба названия начинаются с запроса; по алфавиту Белозерский раньше Бердска, а по числу
        # букв сверх набранного — Бердск (шесть букв) раньше Белозерского (одиннадцать).
        rows = [["Белозерский муниципальный район", "Вологодская область", None, 0],
                ["Бердск городской округ", "Новосибирская область", None, 1]]
        got = run_lib(
            "const rows = lib.rowsFromIndex(input); return lib.searchRows(rows, 'бе', 5).matches.map((r) => r.id);",
            rows,
        )
        self.assertEqual(got, ["Бердск городской округ", "Белозерский муниципальный район"])

    def test_quick_rows_go_before_every_tier(self) -> None:
        # Город быстрого выбора — выше названия, начинающегося с запроса, хотя сам он во втором ярусе:
        # «казан» с Enter открывал «Казанский муниципальный район» Тюменской области, а не Казань.
        rows = [["Казанский муниципальный район", "Тюменская область", None, 0],
                ["городской округ город Казань", "Республика Татарстан", None, 1],
                ["Казачинский муниципальный район", "Красноярский край", None, 2]]
        script = ("const rows = lib.rowsFromIndex(input.rows, input.quick);"
                  "return lib.searchRows(rows, input.query, 5).matches.map((r) => r.id);")
        quick = [{"id": "городской округ город Казань", "short": "Казань"}]
        self.assertEqual(
            run_lib(script, {"rows": rows, "quick": quick, "query": "каз"}),
            ["городской округ город Казань", "Казанский муниципальный район", "Казачинский муниципальный район"],
        )
        # Без быстрого выбора порядок прежний по ярусам: названия, начинающиеся с запроса, раньше Казани.
        self.assertEqual(
            run_lib(script, {"rows": rows, "quick": None, "query": "каз"}),
            ["Казанский муниципальный район", "Казачинский муниципальный район", "городской округ город Казань"],
        )
        # Быстрый выбор не вытаскивает ряд, который под запрос не подходит.
        self.assertEqual(run_lib(script, {"rows": rows, "quick": quick, "query": "казачин"}),
                         ["Казачинский муниципальный район"])

    def test_region_label_distinguishes_rows_without_a_region_by_the_mean_with_its_unit_and_period(self) -> None:
        # Число стоит с единицей и периодом, которые пишет генератор (`mean_unit`): «24,3» без «на человека»
        # читалось бы как расходы всего района. Слов о периоде скрипт не держит — что передали, то и показывает.
        rows = [["городской округ город Орёл", "Орловская область", None, 1],
                ["Михайловский муниципальный район #1", None, None, 3, "23,7"],
                ["Ардатовский муниципальный район", None, None, 3]]
        unit = f"тыс.{NBSP}₽ на человека в{NBSP}месяц за{NBSP}2024 год"
        script = ("const rows = lib.rowsFromIndex(input.rows, null, input.unit);"
                  "return rows.map((row) => lib.regionLabel(row));")
        self.assertEqual(run_lib(script, {"rows": rows, "unit": unit}), [
            "Орловская область",
            f"регион не определён · 23,7{NBSP}{unit}",
            "регион не определён",
        ])
        # Другой период в данных — другая подпись: слов скрипт не вписывает.
        other = f"тыс.{NBSP}₽ на человека в{NBSP}месяц за{NBSP}последние 12 месяцев панели"
        self.assertEqual(run_lib(script, {"rows": rows, "unit": other})[1], f"регион не определён · 23,7{NBSP}{other}")
        # Без единицы числа без пояснения не показывают.
        self.assertEqual(run_lib(script, {"rows": rows, "unit": None})[1], "регион не определён")

    def test_ranking_on_the_real_series_list(self) -> None:
        got = run_lib(
            "const index = JSON.parse(require('fs').readFileSync(input.path, 'utf8'));"
            "const rows = lib.rowsFromIndex(index.series);"
            "return {orel: lib.searchRows(rows, 'город орёл', 3).matches.map((r) => r.id),"
            " kazan: lib.searchRows(rows, 'казань', 3).matches.map((r) => r.id), n: rows.length};",
            {"path": str(INDEX_JSON)},
        )
        self.assertEqual(got["orel"][0], "городской округ город Орёл")
        self.assertEqual(got["kazan"], ["городской округ город Казань"])
        self.assertGreater(got["n"], 1000)

    def test_open_button_goes_where_the_typed_query_points(self) -> None:
        script = (
            "const rows = lib.rowsFromIndex(input.rows);"
            "const pick = (query, highlighted) => { const t = lib.pickTarget(rows, query, highlighted);"
            " return t && t.id; };"
            "return {typed: pick('казань', null), highlighted: pick('казань', rows[1]), empty: pick('', null),"
            " blank: pick('   ', null), none: pick('ъъъъ', null), loading: lib.pickTarget(null, 'казань', null),"
            " highlightedWhileLoading: lib.pickTarget(null, 'казань', rows[2]).id};"
        )
        got = run_lib(script, {"rows": self.ROWS})
        self.assertEqual(got["typed"], "городской округ город Казань")
        self.assertEqual(got["highlighted"], self.ROWS[1][0])
        self.assertIsNone(got["empty"])
        self.assertIsNone(got["blank"])
        self.assertIsNone(got["none"])
        self.assertIsNone(got["loading"])
        self.assertEqual(got["highlightedWhileLoading"], self.ROWS[2][0])


@unittest.skipUnless(shutil.which("node"), "для помощников главной нужен node")
class HistoryKeepTest(unittest.TestCase):
    """График проверки фактом: на узком экране история короче, чтобы кружки месяцев проверки
    не наползали друг на друга."""

    HISTORY, CHECK, STEP = 36, 12, 14

    def keep(self, widths: list[float]) -> list[int]:
        return run_lib(
            "return input.widths.map((w) => lib.historyKeep(input.history, input.check, w, input.step));",
            {"widths": widths, "history": self.HISTORY, "check": self.CHECK, "step": self.STEP},
        )

    def test_dots_keep_their_distance_at_every_width(self) -> None:
        widths = list(range(120, 1400, 7))
        for width, keep in zip(widths, self.keep(widths)):
            with self.subTest(width=width):
                self.assertGreaterEqual(keep, 1)
                self.assertLessEqual(keep, self.HISTORY)
                points = keep + self.CHECK
                if keep > 1:
                    self.assertGreaterEqual(width / (points - 1), self.STEP)
                if keep < self.HISTORY:
                    self.assertLess(width / points, self.STEP, "историю можно было оставить длиннее")

    def test_narrower_graph_never_keeps_more_history(self) -> None:
        widths = list(range(100, 1400, 5))
        kept = self.keep(widths)
        self.assertEqual(kept, sorted(kept))

    def test_wide_graph_keeps_everything_and_unknown_width_does_not_cut(self) -> None:
        self.assertEqual(self.keep([2000, 0]), [self.HISTORY, self.HISTORY])


class LibraryFileTest(unittest.TestCase):
    def test_file_has_no_page_access(self) -> None:
        # Файл целиком запускается в пустом контексте: обращение к странице упало бы там, а не
        # в браузере, но и в тексте его быть не должно.
        source = LIB.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\b(?:document|window|localStorage|navigator)\s*[.\[]", source))
        for forbidden in ("fetch(", "addEventListener", "requestAnimationFrame", "setTimeout"):
            with self.subTest(word=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
