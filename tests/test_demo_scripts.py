"""Скрипты страницы прогноза и главной: ранжирование поиска и поле поиска страницы прогноза, ключ изломов в
легенде графика.

Поиска на главной нет (он — только на странице прогноза), поэтому порядок подсказок проверяется на одной
копии правила — `demo/demo.js::searchMunicipalities`. Файл страницы целиком загружается в пустом контексте
с заглушками страницы: настоящий код, а не его копия в тесте. Поле поиска проверяется на заглушке DOM,
у которой обработчики вызываются событиями. Без node проверки поиска пропускаются.
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


def run_stand(queries: list[str], limit: int) -> dict:
    """Результаты поиска страницы прогноза на настоящем списке рядов: id найденных строк и их число, а также
    подпись региона каждой строки. Быстрые кнопки (`index.quick`) передаются поиску: порядок подсказок
    зависит и от них."""
    script = """
        const vm = require('vm'), fs = require('fs');
        const [demoPath, indexPath, payload] = process.argv.slice(1);
        const input = JSON.parse(payload);
        const noop = () => {};
        const page = {
          document: { getElementById: () => null, addEventListener: noop },
          window: {}, console: { error: noop }, LineChart: {},
          fetch: () => Promise.reject(new Error('в тесте нет сети')),
        };
        const stand = vm.runInNewContext(
          fs.readFileSync(demoPath, 'utf8') + '\\n({searchMunicipalities, buildSearchIndex, regionLabel});', page);
        const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
        const standRows = stand.buildSearchIndex(index.series, index.quick, index.mean_unit);
        const out = {};
        input.queries.forEach((query) => {
          const a = stand.searchMunicipalities(query, standRows, input.limit);
          out[query] = { stand: a.matches.map((r) => r.seriesId), standTotal: a.total };
        });
        out.labels = standRows.map((row) => stand.regionLabel(row));
        console.log(JSON.stringify(out));
    """
    done = subprocess.run(
        ["node", "-e", script, str(ROOT / "demo" / "demo.js"), str(ROOT / "demo" / "data" / "index.json"),
         json.dumps({"queries": queries, "limit": limit})],
        capture_output=True, text=True, check=True, timeout=120,
    )
    return json.loads(done.stdout)


# Страница, которой нет: ровно столько DOM, чтобы скрипт запустился, а его обработчики можно было вызвать событиями.
# Скрипт настоящий, а не копия его логики в тесте; всё, чего заглушка не умеет, скрипты страниц ловят сами
# (каждый блок главной запускается под try).
FAKE_DOM = r"""
const vm = require('vm'), fs = require('fs');
const noop = () => {};
function makeElement(id) {
  return {
    id, value: "", textContent: "", hidden: false, className: "", innerHTML: "", href: "",
    style: {}, dataset: {}, listeners: {}, attrs: {}, calls: [],
    classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); },
    setAttribute(name, value) { this.attrs[name] = String(value); },
    getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; },
    removeAttribute(name) { delete this.attrs[name]; },
    appendChild(child) { return child; }, append: noop, focus: noop, scrollIntoView: noop,
    querySelector: () => null, querySelectorAll: () => [], closest: () => null, contains: () => false,
    select() { this.calls.push('select'); },
    setSelectionRange(start, end) { this.calls.push(`range ${start}-${end}`); },
  };
}
const elements = {};
const document = {
  activeElement: null,
  getElementById: (id) => elements[id] || (elements[id] = makeElement(id)),
  createElement: (tag) => makeElement(tag),
  createTextNode: (text) => ({ textContent: text }),
  querySelectorAll: () => [],
  addEventListener: noop,
};
function fire(target, type, extra) {
  const event = Object.assign({ defaultPrevented: false, target, preventDefault() { this.defaultPrevented = true; } }, extra);
  (target.listeners[type] || []).forEach((handler) => handler(event));
  return event;
}
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const input = JSON.parse(process.argv[1]);
"""


def run_page(body: str, payload: object) -> object:
    """Выполняет `body` над заглушкой страницы (`FAKE_DOM`); `input` — payload, результат — через `return`."""
    script = FAKE_DOM + "(async () => {" + body + "})().then((result) => console.log(JSON.stringify(result)));"
    done = subprocess.run(["node", "-e", script, json.dumps(payload)], capture_output=True, text=True, check=True,
                          timeout=60)
    return json.loads(done.stdout)


@unittest.skipUnless(shutil.which("node"), "для скриптов стенда нужен node")
class SearchRankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.small = run_stand(QUERIES, 8)
        cls.full = run_stand(QUERIES, 100000)

    def test_only_the_forecast_page_has_a_search_and_the_cover_keeps_no_copy_of_its_rule(self) -> None:
        # Раньше правило порядка подсказок жило в двух копиях (страница прогноза и поле на обложке) и тест сверял
        # их друг с другом. Поле с обложки удалено вместе со своей копией: правило одно, `searchMunicipalities`.
        library = (ROOT / "site" / "landing-lib.js").read_text(encoding="utf-8")
        landing = (ROOT / "site" / "landing.js").read_text(encoding="utf-8")
        for name in ("searchRows", "rowsFromIndex", "pickTarget", "regionLabel", "extraLetters"):
            with self.subTest(name=name):
                self.assertNotIn(name, library)
                self.assertNotIn(name, landing)
        self.assertIn("function searchMunicipalities", (ROOT / "demo" / "demo.js").read_text(encoding="utf-8"))

    def test_every_word_starting_a_name_word_goes_before_a_plain_inclusion(self) -> None:
        got = self.small["город орёл"]["stand"]
        self.assertEqual(got[0], "городской округ город Орёл")
        # «орел» внутри «Горелово» — вхождение, а не начало слова: оно ниже Орла.
        gorelovo = [i for i in self.full["город орёл"]["stand"] if "Горелово" in i]
        self.assertEqual(len(gorelovo), 1)
        self.assertGreater(self.full["город орёл"]["stand"].index(gorelovo[0]), 0)

    def test_kazan_goes_first(self) -> None:
        # «каз», «казан» и Enter без выбора открывали «Казанский муниципальный район» Тюменской области:
        # внутри яруса сортировка была по алфавиту, и Казань стояла четвёртой. Теперь Казань первая — ряд
        # быстрого выбора, а за ним по длине совпавшего слова: Казанский, Казачинский, …
        kazan = "городской округ город Казань"
        for query in ("каз", "казан", "казань", "город казань"):
            with self.subTest(query=query):
                self.assertEqual(self.small[query]["stand"][0], kazan)
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

    def test_region_label_of_a_suggestion_follows_the_index_data(self) -> None:
        # Подпись региона в подсказке: регион ряда; у ряда без региона рядом различитель из генератора — число,
        # единица «на человека в месяц» и период, как в `mean_unit` («регион не определён · <число> тыс. ₽ на человека
        # в месяц за <период>»). Слов о периоде скрипт не держит: что передали, то и показывает.
        labels = self.small["labels"]
        index = json.loads((ROOT / "demo" / "data" / "index.json").read_text(encoding="utf-8"))
        self.assertIn("на человека в\u00a0месяц за\u00a0", index["mean_unit"])
        self.assertEqual(len(labels), len(index["series"]))
        for (series_id, region, _oktmo, _file, *mean), label in zip(index["series"], labels):
            if region:
                self.assertEqual(label, region)
            else:
                self.assertEqual(len(mean), 1, f"у ряда без региона {series_id!r} нет различителя")
                self.assertEqual(label, f"регион не определён · {mean[0]}\u00a0{index['mean_unit']}")

    def test_suggestion_list_scrolls_into_view_on_the_forecast_page(self) -> None:
        # На телефоне список подсказок страницы прогноза уходил за низ экрана на несколько строк и не докручивался.
        demo = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        body = re.search(r"function openListbox\(\) \{.*?\n  \}\n", demo, re.S).group(0)
        self.assertIn('listbox.scrollIntoView({ block: "nearest" })', body)


@unittest.skipUnless(shutil.which("node"), "для скриптов стенда нужен node")
class SearchFieldSelectionTest(unittest.TestCase):
    """Страница прогноза: в заполненном поле фокус выделяет имя, и набор заменяет его, а не дописывается к нему
    («городской округ город Орёлказан», и список отвечал «Ничего не нашлось»). Выделение мышью не теряется."""

    SCRIPT = """
        const page = vm.createContext({ document, window: {}, console: { error: noop }, LineChart: {}, setTimeout,
          fetch: () => Promise.reject(new Error('в тесте нет сети')) });
        vm.runInContext(fs.readFileSync(input.demo, 'utf8'), page);
        page.index = { n_series: input.series.length };
        page.index.searchRows = vm.runInContext('buildSearchIndex', page)(input.series, [], input.unit);
        vm.runInContext('setupCombobox(index)', page);
        const field = document.getElementById('mo-search');
        const leave = () => { document.activeElement = null; fire(field, 'blur'); };
        field.value = 'городской округ город Орёл';
        const out = {};

        // Щелчок мышью по полю без фокуса: mousedown, focus, mouseup в той же точке. Имя выделяется на mouseup, и
        // щелчок каретку поверх выделения не ставит. Выделение, оставшееся от прошлого раза, щелчок не отменяет.
        field.selectionStart = 0;
        field.selectionEnd = 26;
        fire(field, 'mousedown', { clientX: 300, clientY: 200 });
        document.activeElement = field;
        fire(field, 'focus');
        out.onFocus = field.calls.slice();
        out.clickUp = fire(field, 'mouseup', { clientX: 301, clientY: 200 }).defaultPrevented;
        out.afterClick = field.calls.slice();
        // Следующий щелчок по полю, где фокус уже есть: каретка как обычно.
        field.calls.length = 0;
        fire(field, 'mousedown', { clientX: 300, clientY: 200 });
        out.secondUp = fire(field, 'mouseup', { clientX: 300, clientY: 200 }).defaultPrevented;
        out.afterSecond = field.calls.slice();

        // Протяжка мышью, начатая в поле без фокуса: выделение читателя остаётся, имя целиком не выделяется.
        leave();
        field.calls.length = 0;
        fire(field, 'mousedown', { clientX: 300, clientY: 200 });
        document.activeElement = field;
        fire(field, 'focus');
        out.dragUp = fire(field, 'mouseup', { clientX: 360, clientY: 202 }).defaultPrevented;
        out.afterDrag = field.calls.slice();
        delete field.selectionStart;
        delete field.selectionEnd;

        // Фокус клавишей Tab: mousedown не было, имя выделяется сразу.
        leave();
        field.calls.length = 0;
        document.activeElement = field;
        fire(field, 'focus');
        out.byTab = field.calls.slice();

        // Мышь отпустили вне поля (mouseup поля не было), фокус ушёл: признак щелчка не залипает, и следующий
        // фокус с клавиатуры выделяет имя.
        leave();
        fire(field, 'mousedown', { clientX: 300, clientY: 200 });
        document.activeElement = field;
        fire(field, 'focus');
        leave();
        field.calls.length = 0;
        document.activeElement = field;
        fire(field, 'focus');
        out.afterStaleClick = field.calls.slice();

        // Пустое поле: выделять нечего, и список не открывается.
        leave();
        field.value = '';
        field.calls.length = 0;
        document.activeElement = field;
        fire(field, 'focus');
        out.empty = field.calls.slice();
        out.listHidden = document.getElementById('mo-listbox').hidden;
        return out;
    """

    @classmethod
    def setUpClass(cls) -> None:
        series = [["городской округ город Орёл", "Орловская область", "54-701-000-000", 0]]
        cls.got = run_page(cls.SCRIPT, {"demo": str(ROOT / "demo" / "demo.js"), "series": series, "unit": "тыс. ₽"})
        cls.whole = ["select", f"range 0-{len('городской округ город Орёл')}"]

    def test_click_selects_the_whole_name_when_it_ends_and_leaves_no_caret_over_it(self) -> None:
        # Даже если в поле осталось выделение с прошлого раза: браузер не сворачивает его до mouseup (выделенный текст
        # можно перетащить), и по состоянию выделения щелчок от протяжки не отличить — отличают по сдвигу мыши.
        self.assertEqual(self.got["onFocus"], [], "на mousedown имя не выделяют: mouseup всё равно поставил бы каретку")
        self.assertTrue(self.got["clickUp"])
        self.assertEqual(self.got["afterClick"], self.whole)

    def test_next_clicks_in_the_focused_field_work_as_usual(self) -> None:
        self.assertFalse(self.got["secondUp"])
        self.assertEqual(self.got["afterSecond"], [])

    def test_a_selection_made_by_dragging_is_kept(self) -> None:
        # Мышь сместилась от точки нажатия — это протяжка, и выделение читателя остаётся как есть.
        self.assertFalse(self.got["dragUp"])
        self.assertEqual(self.got["afterDrag"], [])

    def test_keyboard_focus_selects_the_name_at_once(self) -> None:
        self.assertEqual(self.got["byTab"], self.whole)
        # Щелчок, мышь от которого отпустили вне поля, не оставляет признака: фокус с клавиатуры после него — тоже выделяет.
        self.assertEqual(self.got["afterStaleClick"], self.whole)

    def test_empty_field_has_nothing_to_select_and_opens_no_list(self) -> None:
        self.assertEqual(self.got["empty"], ["select", "range 0-0"])
        self.assertTrue(self.got["listHidden"])

    def test_the_name_stays_selected_when_a_choice_replaces_the_value_of_the_focused_field(self) -> None:
        # Выбор из списка не уводит фокус с поля: имя нового МО выделено, и следующий набор заменяет его.
        demo = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        start = demo.index("searchInput.value = seriesId;")
        self.assertIn("if (document.activeElement === searchInput) selectFieldText(searchInput);",
                      demo[start:start + 300])


class ErrorsCaptionTest(unittest.TestCase):
    """Пояснение к таблице ошибок страницы прогноза: столбцы — «окна проверки», как на главной, и один мостик к слову
    отчёта в скобках. Отчёт зовёт их фолдами везде, а главная этого слова не пишет, поэтому читатель, который сверяет
    таблицу с отчётом, находит его здесь, в ближайшем к отчёту месте."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = (ROOT / "demo" / "demo.js").read_text(encoding="utf-8")
        cls.index = json.loads((ROOT / "demo" / "data" / "index.json").read_text(encoding="utf-8"))

    def test_the_report_word_stands_once_in_the_texts_of_the_page_and_only_as_the_bridge(self) -> None:
        # Тексты страницы — разметка и строки скрипта. Комментарии скрипта о фолдах — про его данные (`index.folds`,
        # протокол отчёта) и читателю не видны: в счёт идут только строки кода.
        code = [line.strip() for line in self.demo.splitlines()
                if re.search(r"(?i)фолд", line) and not line.strip().startswith(("//", "/*", "*"))]
        self.assertEqual(len(code), 1, f"слово отчёта стоит больше чем в одном тексте страницы: {code}")
        self.assertIn("(в отчёте — фолды)", code[0])
        markup = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
        self.assertNotRegex(markup, r"(?i)фолд")
        self.assertNotIn("проверочные окна", self.demo)

    @unittest.skipUnless(shutil.which("node"), "для скриптов стенда нужен node")
    def test_caption_names_the_windows_and_takes_their_length_from_the_data(self) -> None:
        script = """
            const vm = require('vm'), fs = require('fs');
            const [demoPath, indexPath] = process.argv.slice(1);
            const noop = () => {};
            const page = {
              document: { getElementById: () => null, addEventListener: noop },
              window: {}, console: { error: noop }, LineChart: {},
              fetch: () => Promise.reject(new Error('в тесте нет сети')),
            };
            const stand = vm.runInNewContext(fs.readFileSync(demoPath, 'utf8') + '\\n({errorsCaptionText});', page);
            console.log(JSON.stringify(stand.errorsCaptionText(JSON.parse(fs.readFileSync(indexPath, 'utf8')))));
        """
        done = subprocess.run(
            ["node", "-e", script, str(ROOT / "demo" / "demo.js"), str(ROOT / "demo" / "data" / "index.json")],
            capture_output=True, text=True, check=True, timeout=120)
        caption = json.loads(done.stdout)
        # Длина окна — по границам первого окна проверки из данных, своим счётом и своим склонением.
        first = self.index["folds"][0]
        (y1, m1), (y2, m2) = (tuple(int(part) for part in first[key].split("-")) for key in ("test_from", "test_to"))
        months = (y2 - y1) * 12 + (m2 - m1) + 1
        tail = months % 100
        word = ("месяцев" if 11 <= tail <= 14
                else {1: "месяцу", 2: "месяца", 3: "месяца", 4: "месяца"}.get(tail % 10, "месяцев"))
        self.assertEqual(
            caption,
            f"Средняя абсолютная ошибка (MAE), {self.index['unit']}. Столбцы — окна проверки (в отчёте — фолды) "
            f"по\u00a0{months}\u00a0{word}: модель учится на всех месяцах до окна и прогнозирует его. "
            "Крупно — этот муниципалитет, мелко под числом — MAE той же модели в среднем по всей панели.")


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


class TooltipTest(unittest.TestCase):
    """Подсказка графика: единица измерения и закрытие клавишей Escape."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = (ROOT / "demo" / "linechart.js").read_text(encoding="utf-8")

    def test_tooltip_names_the_unit_under_the_month(self) -> None:
        # Число без единицы («Факт 21 722») читалось как «чего?»: единица спецификации графика — в подсказке.
        show = re.search(r"function showTooltip\(.*?\n    \}\n", self.chart, re.S).group(0)
        self.assertIn("if (currentSpec.yUnit)", show)
        self.assertIn('unit.className = "chart-tooltip-unit"', show)
        self.assertIn("unit.textContent = currentSpec.yUnit", show)
        self.assertLess(show.index("chart-tooltip-month"), show.index("chart-tooltip-unit"))
        self.assertLess(show.index("chart-tooltip-unit"), show.index("chart-tooltip-row"))

    def test_escape_closes_the_tooltip_and_the_listener_is_removed_with_the_chart(self) -> None:
        # Подсказка, появившаяся при наведении, закрывается без движения указателя (WCAG 1.4.13).
        self.assertRegex(
            self.chart, r'function onDocKeyDown\(event\) \{\s*if \(event\.key === "Escape"\) hideTooltip\(\);\s*\}')
        self.assertIn('document.addEventListener("keydown", onDocKeyDown);', self.chart)
        destroy = re.search(r"destroy\(\) \{.*?\n      \},", self.chart, re.S).group(0)
        self.assertIn('document.removeEventListener("keydown", onDocKeyDown);', destroy)

    def test_unit_line_is_styled_for_the_dark_tooltip(self) -> None:
        css = (ROOT / "demo" / "demo.css").read_text(encoding="utf-8")
        rule = re.search(r"\.chart-tooltip-unit \{([^}]*)\}", css).group(1)
        self.assertIn("var(--night-soft)", rule)


class TableScrollHintTest(unittest.TestCase):
    def test_scrollable_tables_have_an_edge_shadow_that_appears_only_when_there_is_more_to_scroll(self) -> None:
        # На телефоне таблица шире рамки прокручивалась вбок без признака: два правых столбца были невидимы.
        # Тень — фон: закраска «local» едет с содержимым и скрывает тень «scroll» там, где прокручивать нечего.
        css = (ROOT / "demo" / "demo.css").read_text(encoding="utf-8")
        rule = re.search(r"\n\.table-scroll \{([^}]*)\}", css).group(1)
        self.assertIn("overflow-x: auto", rule)
        self.assertEqual(rule.count("linear-gradient"), 2)
        self.assertEqual(rule.count("radial-gradient"), 2)
        self.assertEqual(len(re.findall(r"no-repeat local", rule)), 2)
        self.assertEqual(len(re.findall(r"no-repeat scroll", rule)), 2)
        # Закраска — цвета карточки, на которой лежит таблица (и светлой, и ночной).
        self.assertEqual(rule.count("var(--surface)"), 2)


if __name__ == "__main__":
    unittest.main()
