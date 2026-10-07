/* demo/demo.js — демонстрационный стенд: загрузка данных, поиск МО и быстрые кнопки,
 * адрес страницы, сборка блоков. Числа и подписи графиков рисует общий demo/linechart.js —
 * этот файл отвечает за то, ОТКУДА берутся ряды для него и что написано на странице.
 *
 * Видимый текст живёт в двух местах: разметка demo/index.html (то, что не зависит
 * от данных) и функции ниже с суффиксом Text (то, что зависит — числа, названия
 * моделей, месяцы). Числа в разметке не пишем — только через эти функции, из JSON.
 */
"use strict";

// ---------------------------------------------------------------------------
// Числа и месяцы — форматирование как в отчёте (report/report.qmd, scripts/build_site.py):
// неразрывный пробел в разрядах, запятая, минус «−». Проценты агрегата приходят уже
// готовой строкой из demo/data/aggregate.json (тем же num(v, 1), что в отчёте) — здесь
// только достраивается «%», без повторного округления.
// ---------------------------------------------------------------------------

function groupThousands(digitsStr) {
  let s = digitsStr;
  const groups = [];
  while (s.length > 3) {
    groups.unshift(s.slice(-3));
    s = s.slice(0, -3);
  }
  groups.unshift(s);
  return groups.join("\u00a0");
}

function numFmt(value, digits, opts) {
  opts = opts || {};
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  let v = Math.round(value * Math.pow(10, digits)) / Math.pow(10, digits);
  if (Object.is(v, -0)) v = 0;
  const neg = v < 0;
  const text = Math.abs(v).toFixed(digits);
  const parts = text.split(".");
  const grouped = groupThousands(parts[0]);
  const sign = neg ? "−" : (opts.sign && v > 0 ? "+" : "");
  return sign + grouped + (parts[1] ? "," + parts[1] : "");
}

function rubFmt(value) {
  return numFmt(value, 0);
}

/** Достраивает «%» к уже готовой строке процента из aggregate.json — без своего
 * округления (см. комментарий вверху файла). "—" остаётся "—", без "%" на конце. */
function withPercent(formatted) {
  return formatted === "—" ? formatted : formatted + "%";
}

function pluralRu(n, one, few, many) {
  const m = Math.abs(Math.trunc(n)) % 100;
  if (m >= 11 && m <= 14) return many;
  const last = m % 10;
  if (last === 1) return one;
  if (last >= 2 && last <= 4) return few;
  return many;
}

const MONTH_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь",
                    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
const MONTH_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня",
                    "июля", "августа", "сентября", "октября", "ноября", "декабря"];

function parseYM(ym) {
  const [y, m] = ym.split("-").map(Number);
  return { y, m };
}

function monthNomYear(ym) {
  const { y, m } = parseYM(ym);
  return `${MONTH_NOM[m - 1]} ${y}`;
}

function monthGenYear(ym) {
  const { y, m } = parseYM(ym);
  return `${MONTH_GEN[m - 1]} ${y}`;
}

/** «февраль–март 2025» (один год) или «январь 2025» (from===to) — для таблиц и подписей,
 * не для оси графика (там короткие подписи — LineChart.formatMonthShort). */
function monthRangeNom(fromYm, toYm) {
  if (fromYm === toYm) return monthNomYear(fromYm);
  const a = parseYM(fromYm);
  const b = parseYM(toYm);
  if (a.y === b.y) return `${MONTH_NOM[a.m - 1]}–${MONTH_NOM[b.m - 1]} ${a.y}`;
  return `${monthNomYear(fromYm)} – ${monthNomYear(toYm)}`;
}

/** Короткая форма диапазона для тесных мест (узкие заголовки таблицы ошибок) —
 * «апр–июн 2024», год один раз, месяцы — сокращения LineChart.formatMonthShort. */
function monthRangeAxis(fromYm, toYm) {
  if (fromYm === toYm) return LineChart.formatMonthShort(fromYm, true);
  const a = parseYM(fromYm);
  const b = parseYM(toYm);
  return a.y === b.y
    ? `${LineChart.formatMonthShort(fromYm, false)}–${LineChart.formatMonthShort(toYm, true)}`
    : `${LineChart.formatMonthShort(fromYm, true)} – ${LineChart.formatMonthShort(toYm, true)}`;
}

/** «с апреля по декабрь 2024 года» (один год) или «с октября 2023 по март 2024 года»:
 * «с» берёт родительный падеж, «по» в значении «включительно» — винительный. */
function monthSpanText(fromYm, toYm) {
  const a = parseYM(fromYm);
  const b = parseYM(toYm);
  return a.y === b.y
    ? `с ${MONTH_GEN[a.m - 1]} по ${MONTH_NOM[b.m - 1]} ${b.y} года`
    : `с ${MONTH_GEN[a.m - 1]} ${a.y} по ${MONTH_NOM[b.m - 1]} ${b.y} года`;
}

/** «1, 3 и 6» — перечень чисел через запятую с «и» перед последним. */
function andJoinRu(items) {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} и ${items[items.length - 1]}`;
}

function monthsBetweenInclusive(fromYm, toYm) {
  const a = parseYM(fromYm);
  const b = parseYM(toYm);
  return (b.y - a.y) * 12 + (b.m - a.m) + 1;
}

function dateWordsFromISO(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return `${d} ${MONTH_GEN[m - 1]} ${y}`;
}

// ---------------------------------------------------------------------------
// Детектор изломов — короткие словари для текста. Названия моделей (по роли и
// пояснение под ним) генератор кладёт прямо в данные — models[].name и models[].note,
// полное название — models[].label и forecast_rule[].label (scripts/build_site.py,
// словари ROLE_NAMES и MODEL_LABELS): те же названия на главной и здесь, и второго,
// JS-словаря с тем же текстом не держим.
// ---------------------------------------------------------------------------

// Как в отчёте (report/report.qmd :: _CP_DET / _CP_ON) — те же подписи для тех же
// понятий и там, и здесь.
const CP_DETECTOR_LABELS = {
  pelt: "PELT", binseg: "BinSeg", window: "Window", bottomup: "BottomUp",
  kernel_rbf: "ядровой (RBF)", cusum: "CUSUM",
};
const CP_MODE_LABELS_ON = {
  ratio: "на темпах роста", raw: "на сырых значениях", deseason: "без профиля месяца",
};

function formatPenalty(p) {
  return numFmt(p, Number.isInteger(p) ? 0 : 1);
}

/** «детектор PELT на темпах роста, штраф 1» — чем искали изломы; вставляется в скобки. */
function breaksMethodologyText(breaksInfo) {
  const det = CP_DETECTOR_LABELS[breaksInfo.detector] || breaksInfo.detector;
  const on = CP_MODE_LABELS_ON[breaksInfo.mode] || breaksInfo.mode;
  return `детектор ${det} ${on}, штраф ${formatPenalty(breaksInfo.penalty)}`;
}

// ---------------------------------------------------------------------------
// Поиск МО: без учёта регистра, «ё» = «е», по всем словам запроса в строке
// «название + регион». Порядок: сначала города быстрого выбора (их обычно и ищут), затем три
// яруса — название начинается с запроса; каждое слово запроса начинает какое-то слово названия;
// любое вхождение. Внутри яруса — по числу букв сверх набранного в совпавших словах (Казань
// раньше Казанского и Казачинского), затем по алфавиту. Тот же порядок на главной
// (site/landing-lib.js::searchRows): подсказки там и здесь совпадают.
// ---------------------------------------------------------------------------

function normalizeSearchText(s) {
  return s.toLowerCase().replace(/ё/g, "е");
}

function buildSearchIndex(seriesRows, quick, meanUnit) {
  // seriesRows — index.json.series: [[series_id, регион|null, ОКТМО|null, файл, средние расходы], ...];
  // пятый элемент есть только у рядов без региона. quick — index.json.quick: [{id, short}, ...].
  // meanUnit — index.json.mean_unit: единица этого числа вместе с периодом, её пишет генератор.
  const featured = new Set((quick || []).map((item) => item.id));
  return seriesRows.map(([seriesId, region, oktmo, fileNumber, mean]) => ({
    seriesId, region, oktmo, fileNumber,
    mean: mean === undefined ? null : mean,
    meanUnit: meanUnit || null,
    featured: featured.has(seriesId),
    nameNorm: normalizeSearchText(seriesId),
    haystack: normalizeSearchText(region ? `${seriesId} ${region}` : seriesId),
  }));
}

const NO_REGION_TEXT = "регион не определён";

/** Подпись региона в подсказке. У ряда без региона рядом — различитель из генератора: средние расходы,
 * число, единица и период («<число> тыс. ₽ на человека в месяц за <период>»). Число без единицы читалось бы
 * как расходы всего района, поэтому без единицы различителя нет. Тот же текст собирает главная
 * (site/landing-lib.js::regionLabel). */
function regionLabel(row) {
  return row.region || (row.mean && row.meanUnit ? `${NO_REGION_TEXT} · ${row.mean}\u00a0${row.meanUnit}` : NO_REGION_TEXT);
}

/** Букв сверх набранного в словах названия, которым отвечают слова запроса: для каждого слова —
 * кратчайшее слово названия, которое с него начинается, а при его отсутствии — содержащее его.
 * Слово, найденное только в регионе, ничего не добавляет. */
function extraLetters(nameWords, words) {
  return words.reduce((sum, word) => {
    const starting = nameWords.filter((nameWord) => nameWord.startsWith(word));
    const pool = starting.length ? starting : nameWords.filter((nameWord) => nameWord.includes(word));
    if (!pool.length) return sum;
    return sum + Math.min(...pool.map((nameWord) => nameWord.length)) - word.length;
  }, 0);
}

function searchMunicipalities(query, rows, limit) {
  const words = normalizeSearchText(query.trim()).split(/\s+/).filter(Boolean);
  if (!words.length) return { matches: [], total: 0 };
  const phrase = words.join(" ");
  const found = [];
  for (const row of rows) {
    if (!words.every((w) => row.haystack.includes(w))) continue;
    // «город орёл»: слово «орел» внутри «Горелово» — не начало слова, и Орёл идёт выше него.
    const nameWords = row.nameNorm.split(/\s+/);
    let tier = 2;
    if (row.nameNorm.startsWith(phrase)) tier = 0;
    else if (words.every((w) => nameWords.some((nameWord) => nameWord.startsWith(w)))) tier = 1;
    found.push({ row, tier, extra: extraLetters(nameWords, words) });
  }
  const collator = new Intl.Collator("ru");
  found.sort((a, b) => (Number(b.row.featured) - Number(a.row.featured))
    || (a.tier - b.tier) || (a.extra - b.extra) || collator.compare(a.row.seriesId, b.row.seriesId));
  return { matches: found.slice(0, limit).map((item) => item.row), total: found.length };
}

// ---------------------------------------------------------------------------
// Загрузка данных — index.json один раз, demo/data/mo/<n>.json по требованию,
// с кэшем по номеру файла (один файл может отвечать за много рядов одного региона
// или за все ряды без региона).
// ---------------------------------------------------------------------------

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`не удалось загрузить ${path}: ${response.status}`);
  return response.json();
}

const moFileCache = new Map();

async function fetchMoFile(fileNumber) {
  if (!moFileCache.has(fileNumber)) {
    // Отклонённый промис не остаётся в кэше: иначе один сбой сети закрывает весь
    // регион до перезагрузки страницы, а не только эту попытку — следующий выбор
    // того же региона попробует загрузить заново.
    moFileCache.set(fileNumber, fetchJson(`data/mo/${fileNumber}.json`).catch((error) => {
      moFileCache.delete(fileNumber);
      throw error;
    }));
  }
  return moFileCache.get(fileNumber);
}

// ---------------------------------------------------------------------------
// Адрес страницы: ?mo=<series_id>, history.pushState/popstate.
// ---------------------------------------------------------------------------

function getMoParam() {
  return new URLSearchParams(window.location.search).get("mo");
}

function pushMoParam(seriesId) {
  const url = new URL(window.location.href);
  url.searchParams.set("mo", seriesId);
  window.history.pushState({ mo: seriesId }, "", url.pathname + url.search);
}

// ---------------------------------------------------------------------------
// Тексты страницы, зависящие от данных.
// ---------------------------------------------------------------------------

function introText(index) {
  // Ошибки моделей (таблица ниже) — на тестовых месяцах фолдов, а не на всей
  // панели: ошибка меряется там, где есть факт, с которым сравнить прогноз фолда.
  const forecastYear = parseYM(index.forecast_months[0]).y;
  const firstFold = index.folds[0];
  const lastFold = index.folds[index.folds.length - 1];
  const total = index.n_series;
  return `Любой из ${rubFmt(total)} ` +
    `${pluralRu(total, "муниципалитета", "муниципалитетов", "муниципалитетов")} панели: ` +
    `расходы по месяцам, прогноз на ${forecastYear} год, изломы ряда (точки структурных изменений) и то, как модели ошибались ` +
    `на нём при проверке на истории — ${monthSpanText(firstFold.test_from, lastFold.test_to)}.`;
}

function searchHintText(index) {
  const total = index.n_series;
  const noRegion = index.n_no_region;
  // Ряды с суффиксом « #N» в series_id — те самые, о которых говорит последняя фраза
  // («их различает номер после «#»»), а не любые ряды с неуникальным базовым именем:
  // у части омонимов вторая копия выпала из панели по пропускам, и счёт по имени
  // занижал бы число.
  const names = index.n_hash_names;
  const series = index.n_hash_series;
  return "Ищите по любым словам из названия или региона. " +
    `У ${rubFmt(noRegion)} из ${rubFmt(total)} ` +
    `${pluralRu(total, "муниципального образования", "муниципальных образований", "муниципальных образований")} ` +
    `регион не определён, в том числе у ${rubFmt(series)} ` +
    `${pluralRu(series, "одноимённого ряда", "одноимённых рядов", "одноимённых рядов")} ` +
    `(${rubFmt(names)} ${pluralRu(names, "название", "названия", "названий")}) — их различает номер после «#», ` +
    `а в подсказках рядом — средние расходы, ${index.mean_unit}.`;
}

function invalidMoMessage(seriesId) {
  return `«${seriesId}» нет в списке муниципальных образований — показано муниципальное образование по умолчанию.`;
}

function noRegionExplanationText(seriesId) {
  let text = "В выгрузке СберИндекса у муниципальных образований нет ни ОКТМО, ни региона — " +
    "только название. Регион берётся из справочника по названию. Это название носят " +
    "муниципалитеты нескольких регионов или его там нет — в обоих случаях мы не стали " +
    "приписывать регион наугад.";
  const suffix = / #(\d+)$/.exec(seriesId);
  if (suffix) {
    text += ` «#${suffix[1]}» — номер ряда среди одноимённых, по порядку в выгрузке.`;
  }
  return text;
}

/** forecast_rule мержит отрезки по паре (модель, горизонт) — у одной модели на
 * нескольких горизонтах подряд получится несколько отрезков (см. пример в
 * докстринге build_site.py). Для текста они группируются заново, уже только
 * по модели, иначе одно и то же название повторяется в тексте по разу на горизонт. */
function groupForecastRuleByModel(rule) {
  const groups = [];
  rule.forEach((seg) => {
    const last = groups[groups.length - 1];
    if (last && last.model === seg.model) {
      last.to = seg.to;
      last.horizons.push(seg.horizon);
    } else {
      groups.push({ model: seg.model, label: seg.label, from: seg.from, to: seg.to, horizons: [seg.horizon] });
    }
  });
  return groups;
}

function forecastRuleText(index) {
  const forecastYear = parseYM(index.forecast_months[0]).y;
  const groups = groupForecastRuleByModel(index.forecast_rule);
  const parts = groups.map((g) => {
    const range = g.from === g.to ? monthNomYear(g.from) : monthRangeNom(g.from, g.to);
    const horizons = g.horizons.length === 1
      ? `горизонт ${g.horizons[0]} мес.`
      : `горизонты ${andJoinRu(g.horizons.map(String))} мес.`;
    return `${range} — «${g.label}» (${horizons})`;
  });
  // Месяц берётся с самого короткого горизонта, который его покрывает — не с «отдельного»
  // горизонта у каждой модели порознь; иначе непонятно, почему у месяца вообще одна модель,
  // а не несколько сразу.
  return "Прогноз на каждый месяц — с самого короткого горизонта, который его покрывает, " +
    `от рекомендуемой на этом горизонте модели: ${parts.join("; ")}. ` +
    "Пунктир — доли муниципалитета по прошлым месяцам, умноженные на опубликованный федеральный индекс. " +
    `Муниципальный разрез ${forecastYear} года ещё не опубликован, поэтому прогноз по муниципалитету ` +
    "фактом не проверен.";
}

function breaksListText(mo, breaksInfo) {
  const method = breaksMethodologyText(breaksInfo);
  if (!mo.breaks.length) {
    return `Изломов в ряду не найдено — искали задним числом по всему ряду (${method}).`;
  }
  const months = mo.breaks.map((ym) => monthNomYear(ym)).join(", ");
  return `Штриховые вертикальные линии — изломы ряда: ${months}. Найдены задним числом ` +
    `по всему ряду (${method}), это не сигнал в реальном времени.`;
}

function errorsCaptionText(index) {
  const testLen = monthsBetweenInclusive(index.folds[0].test_from, index.folds[0].test_to);
  return `Средняя абсолютная ошибка (MAE), ${index.unit}. Столбцы — фолды, проверочные окна ` +
    `по ${testLen} ${pluralRu(testLen, "месяцу", "месяца", "месяцев")}: модель учится на всех месяцах ` +
    "до окна и прогнозирует его. Крупно — этот муниципалитет, мелко под числом — " +
    "MAE той же модели в среднем по всей панели.";
}

function aggregateLeadText(agg) {
  const forecastYear = parseYM(agg.check.months[0]).y;
  return "Первый этап двухэтапной модели прогнозирует федеральный ряд СберИндекса. " +
    `Его прогноз от ${monthGenYear(agg.origin)} на ${forecastYear} год — единственный в работе, ` +
    "который уже можно сверить с фактом: средняя ошибка за год " +
    `${withPercent(agg.mape.two_stage)} против ${withPercent(agg.mape.naive)} у правила ` +
    `«${agg.rule_names.naive}» и ${withPercent(agg.mape.seasonal_naive)} у правила «${agg.rule_names.seasonal_naive}».`;
}

function aggregateCaptionText(agg) {
  const forecastYear = parseYM(agg.check.months[0]).y;
  return `Совокупные потребительские расходы России по данным СберИндекса, ${agg.unit}. ` +
    `Точки на линии факта — месяцы ${forecastYear} года, которых модель не видела. ` +
    `Прогнозы — от ${monthGenYear(agg.origin)}: первый этап (модель «${agg.model_label}»), ` +
    `правило «${agg.rule_names.naive}», правило «${agg.rule_names.seasonal_naive}». ` +
    "Ошибка в таблице — (прогноз − факт) / факт: минус — прогноз ниже факта.";
}

// ---------------------------------------------------------------------------
// График выбранного МО.
// ---------------------------------------------------------------------------

// Название столбца «Числа графика» и подпись того же ряда в легенде графика — один
// и тот же текст, не два разных о том же самом.
const KNOWN_SERIES_LABEL = "Если федеральный индекс за месяц уже опубликован";

function buildMoChartSpec(index, seriesId, mo, legendEl) {
  const months = index.panel_months.concat(index.forecast_months);
  const nPanel = index.panel_months.length;

  const factValues = new Array(months.length).fill(null);
  mo.fact.forEach((v, i) => { factValues[i] = v; });

  // Точка стыка (nPanel - 1): факт последнего месяца панели, повторённый в начале линий
  // прогноза и «известного» пунктира, чтобы линии визуально соединялись с фактом, а не
  // начинались с разрыва. joinAt отмечает её для LineChart — подсказка графика эту точку
  // не подписывает «Прогноз»/«Если индекс уже опубликован»: число там и так факт,
  // а не прогноз ни на йоту.
  const forecastValues = new Array(months.length).fill(null);
  forecastValues[nPanel - 1] = mo.fact[nPanel - 1];
  mo.forecast.forEach((v, i) => { forecastValues[nPanel + i] = v; });

  const knownValues = new Array(months.length).fill(null);
  knownValues[nPanel - 1] = mo.fact[nPanel - 1];
  mo.known.forEach((v, i) => { knownValues[nPanel + i] = v; });

  const markers = [{ atMonth: months[nPanel - 1], kind: "end", label: "конец панели" }];
  mo.breaks.forEach((ym) => {
    markers.push({ atMonth: ym, kind: "break", label: LineChart.formatMonthShort(ym, true) });
  });

  return {
    months,
    series: [
      { id: "fact", label: "Факт", color: "var(--chart-fact)", values: factValues },
      {
        id: "forecast", label: "Прогноз", color: "var(--chart-forecast)",
        values: forecastValues, joinAt: nPanel - 1,
      },
      {
        id: "known", label: KNOWN_SERIES_LABEL,
        color: "var(--chart-known)", dash: "dashed", values: knownValues, joinAt: nPanel - 1,
      },
    ],
    markers,
    yFormat: rubFmt,
    yUnit: index.unit,
    // Вертикальные штриховые линии — изломы: без ключа в легенде не видно, что они значат.
    markerKeys: [{ kind: "break", label: "изломы" }],
    ariaLabel: `Линейный график: расходы «${seriesId}» по месяцам — факт ` +
      `${monthSpanText(months[0], months[nPanel - 1])} и прогноз ` +
      `${monthSpanText(months[nPanel], months[months.length - 1])}, ${index.unit}.`,
    legendEl,
  };
}

// Ячейка данных таблицы: прочерк («—», нет числа для этого месяца/фолда) выглядит
// приглушённо — той же меткой, что и настоящий пропуск в numFmt.
function appendDataCell(row, text) {
  const td = document.createElement("td");
  td.textContent = text;
  if (text === "—") td.className = "cell-dash";
  row.appendChild(td);
  return td;
}

function renderMoNumbersTable(index, mo) {
  const months = index.panel_months.concat(index.forecast_months);
  const nPanel = index.panel_months.length;
  const table = document.getElementById("mo-numbers-table");
  table.innerHTML = "";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Месяц", "Факт", "Прогноз", KNOWN_SERIES_LABEL].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  months.forEach((ym, i) => {
    const tr = document.createElement("tr");
    const rowTh = document.createElement("th");
    rowTh.setAttribute("scope", "row");
    rowTh.textContent = LineChart.formatMonthShort(ym, true);
    tr.appendChild(rowTh);
    [
      i < nPanel ? rubFmt(mo.fact[i]) : "—",
      i >= nPanel ? rubFmt(mo.forecast[i - nPanel]) : "—",
      i >= nPanel ? rubFmt(mo.known[i - nPanel]) : "—",
    ].forEach((text) => appendDataCell(tr, text));
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
}

// ---------------------------------------------------------------------------
// Таблица «Как модели ошибались на этом МО» — MAE по фолдам и в среднем; строкой
// ниже каждого числа — то же по всей панели (index.panel_mae), мельче и бледнее.
// ---------------------------------------------------------------------------

function modelMeansForMo(index, mo) {
  const result = {};
  index.models.forEach((model) => {
    const perFold = mo.mae[model.id] || new Array(index.folds.length).fill(null);
    const finite = perFold.filter((v) => v !== null && v !== undefined);
    result[model.id] = { perFold, mean: finite.length ? finite.reduce((a, b) => a + b, 0) / finite.length : null };
  });
  return result;
}

function bestPerColumn(index, means) {
  const nFolds = index.folds.length;
  const best = new Array(nFolds + 1).fill(null);
  for (let col = 0; col < nFolds; col += 1) {
    const values = index.models
      .map((m) => means[m.id].perFold[col])
      .filter((v) => v !== null && v !== undefined);
    best[col] = values.length ? Math.min(...values) : null;
  }
  const meanValues = index.models.map((m) => means[m.id].mean).filter((v) => v !== null);
  best[nFolds] = meanValues.length ? Math.min(...meanValues) : null;
  return best;
}

function renderErrorsTable(index, mo) {
  const table = document.getElementById("errors-table");
  table.innerHTML = "";
  const nFolds = index.folds.length;

  // Компактные подписи фолдов — общие для широких заголовков (две строки в <th>)
  // и узких карточек (data-label каждой ячейки, см. demo.css): «апр–июн 2024».
  // В карточке — она же вместе с длиной обучения: на телефоне <thead> скрыт
  // стилем целиком, и без этого «обучение N мес.» там негде взять.
  const foldLabels = index.folds.map((fold) => monthRangeAxis(fold.test_from, fold.test_to));
  const foldCardLabels = index.folds.map(
    // Неразрывный пробел перед «мес.» — иначе на узкой карточке (390 px) подпись
    // переносится между числом и словом: «обучение 15 / мес.».
    (fold) => `${monthRangeAxis(fold.test_from, fold.test_to)} · обучение ${fold.train_months} мес.`
  );

  // display: grid и block (demo.css) снимают встроенную табличную семантику — явные role
  // восстанавливают её для скринридера и ничего не меняют там, где браузер сохранил её сам.
  const thead = document.createElement("thead");
  thead.setAttribute("role", "rowgroup");
  const headRow = document.createElement("tr");
  headRow.setAttribute("role", "row");
  const thModel = document.createElement("th");
  thModel.setAttribute("role", "columnheader");
  thModel.textContent = "Модель";
  headRow.appendChild(thModel);
  index.folds.forEach((fold, i) => {
    const th = document.createElement("th");
    th.setAttribute("role", "columnheader");
    const monthsLine = document.createElement("span");
    monthsLine.className = "fold-head-months";
    monthsLine.textContent = foldLabels[i];
    const trainLine = document.createElement("span");
    trainLine.className = "fold-head-train";
    trainLine.textContent = `обучение ${fold.train_months} мес.`;
    th.append(monthsLine, trainLine);
    headRow.appendChild(th);
  });
  const thMean = document.createElement("th");
  thMean.setAttribute("role", "columnheader");
  thMean.textContent = "Среднее";
  headRow.appendChild(thMean);
  thead.appendChild(headRow);

  const means = modelMeansForMo(index, mo);
  const best = bestPerColumn(index, means);
  const panelMae = index.panel_mae;

  const tbody = document.createElement("tbody");
  tbody.setAttribute("role", "rowgroup");
  index.models.forEach((model) => {
    const tr = document.createElement("tr");
    tr.setAttribute("role", "row");
    const rowTh = document.createElement("th");
    rowTh.setAttribute("scope", "row");
    rowTh.setAttribute("role", "rowheader");
    // Название по роли — первой, крупной строкой; пояснение — второй, приглушённой: то же название
    // и то же пояснение, что у полос горизонтов на главной (генератор: model_names).
    const primary = document.createElement("span");
    primary.className = "role-primary";
    primary.textContent = model.name;
    const secondary = document.createElement("span");
    secondary.className = "cell-sub role-desc";
    secondary.textContent = model.note;
    rowTh.append(primary, secondary);
    tr.appendChild(rowTh);

    const { perFold, mean } = means[model.id];
    // Фолдов может быть не три — сколько их, решает конфиг, а не разметка таблицы:
    // и длина запасного массива, и индекс столбца «Среднее» берутся из nFolds,
    // а не зашиты числом.
    const panelRow = panelMae[model.id] || new Array(nFolds + 1).fill(null);

    perFold.forEach((v, col) => {
      tr.appendChild(errorCell(v, panelRow[col], v !== null && v === best[col], foldCardLabels[col]));
    });
    tr.appendChild(errorCell(mean, panelRow[nFolds], mean !== null && mean === best[nFolds], "Среднее"));
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
}

function errorCell(value, panelValue, isBest, narrowLabel) {
  const td = document.createElement("td");
  td.setAttribute("role", "cell");
  const classes = [];
  if (isBest) classes.push("cell-best");
  if (value === null) classes.push("cell-dash");
  if (classes.length) td.className = classes.join(" ");
  // Ниже 900 px строка таблицы становится карточкой (demo.css): data-label — это
  // подпись столбца у ячейки, которую иначе показывал бы скрытый <thead>.
  td.dataset.label = narrowLabel;
  td.textContent = value === null ? "—" : rubFmt(value);
  const sub = document.createElement("span");
  sub.className = "cell-sub";
  sub.textContent = `панель ${panelValue === null || panelValue === undefined ? "—" : rubFmt(panelValue)}`;
  td.appendChild(sub);
  return td;
}

// ---------------------------------------------------------------------------
// Федеральный агрегат.
// ---------------------------------------------------------------------------

function buildAggregateChartSpec(agg, legendEl) {
  const months = agg.history.months.concat(agg.check.months);
  const nHist = agg.history.months.length;

  const factLine = new Array(months.length).fill(null);
  agg.history.values.forEach((v, i) => { factLine[i] = v; });
  agg.check.actual.forEach((v, i) => { factLine[nHist + i] = v; });

  const factDots = new Array(months.length).fill(null);
  agg.check.actual.forEach((v, i) => { factDots[nHist + i] = v; });

  function ruleValues(key) {
    const arr = new Array(months.length).fill(null);
    arr[nHist - 1] = agg.history.values[nHist - 1];
    agg.check.forecast[key].forEach((v, i) => { arr[nHist + i] = v; });
    return arr;
  }

  // joinAt — та же точка стыка, что и на графике МО (buildMoChartSpec, см. её
  // комментарий): факт последнего месяца истории, продублированный в начале трёх
  // линий прогноза ради непрерывности линии, но не прогноз в подсказке.
  const series = [
    { id: "fact", label: "Факт", color: "var(--chart-fact)", values: factLine },
    { id: "fact-check", label: "Факт", color: "var(--chart-fact)", mode: "dots", values: factDots, overlay: true },
    {
      id: "two_stage", label: "Прогноз первого этапа", color: "var(--chart-forecast)",
      values: ruleValues("two_stage"), joinAt: nHist - 1,
    },
    {
      id: "naive", label: `Правило «${agg.rule_names.naive}»`,
      color: "var(--chart-known)", dash: "dashed", values: ruleValues("naive"), joinAt: nHist - 1,
    },
    {
      id: "seasonal_naive", label: `Правило «${agg.rule_names.seasonal_naive}»`,
      color: "var(--chart-third)", dash: "dotted", values: ruleValues("seasonal_naive"), joinAt: nHist - 1,
    },
  ];
  const markers = [{ atMonth: agg.origin, kind: "end", label: "конец панели" }];

  return {
    // На оси и в подсказке — целые (деления и так круглые числа, сотые ни к чему);
    // точные сотые — только в таблице ниже (renderAggregateTable), где это уже число, а не деление.
    months, series, markers, yFormat: rubFmt, yUnit: agg.unit,
    ariaLabel: `Линейный график: совокупные потребительские расходы России по месяцам — факт ` +
      `${monthSpanText(months[0], months[months.length - 1])} и прогнозы от ${monthGenYear(agg.origin)}: ` +
      `первый этап двухэтапной модели, правила «${agg.rule_names.naive}» и ` +
      `«${agg.rule_names.seasonal_naive}», ${agg.unit}.`,
    legendEl,
  };
}

function renderAggregateTable(agg) {
  // Как в таблице отчёта (report/report.qmd, ~2148-2163): факт и прогноз целыми
  // (agg.check.actual_rub/forecast_rub — готовые строки из Python, см. withPercent
  // вверху файла про ту же логику у процентов), ошибка — только в процентах (свой
  // знак у каждой из трёх), без рублёвого разноса и рублёвой средней ошибки — они
  // здесь ничего не добавляют к процентам.
  const table = document.getElementById("agg-table");
  table.innerHTML = "";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  [
    "Месяц", "Факт", "Прогноз", "Ошибка, %",
    `«${agg.rule_names.naive}», %`, `«${agg.rule_names.seasonal_naive}», %`,
  ].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  agg.check.months.forEach((ym, i) => {
    const tr = document.createElement("tr");
    const rowTh = document.createElement("th");
    rowTh.setAttribute("scope", "row");
    rowTh.textContent = LineChart.formatMonthShort(ym, true);
    tr.appendChild(rowTh);
    [
      agg.check.actual_rub[i],
      agg.check.forecast_rub.two_stage[i],
      withPercent(agg.check.error_pct.two_stage[i]),
      withPercent(agg.check.error_pct.naive[i]),
      withPercent(agg.check.error_pct.seasonal_naive[i]),
    ].forEach((text) => appendDataCell(tr, text));
    tbody.appendChild(tr);
  });

  const tr = document.createElement("tr");
  const rowTh = document.createElement("th");
  rowTh.setAttribute("scope", "row");
  rowTh.textContent = "Средняя абсолютная ошибка";
  tr.appendChild(rowTh);
  [
    "—", "—",
    withPercent(agg.mape.two_stage), withPercent(agg.mape.naive), withPercent(agg.mape.seasonal_naive),
  ].forEach((text) => appendDataCell(tr, text));
  tbody.appendChild(tr);

  table.append(thead, tbody);
}

let aggregateChart = null;

async function renderAggregateBlock() {
  const agg = await fetchJson("data/aggregate.json");
  document.getElementById("agg-lead").textContent = aggregateLeadText(agg);
  document.getElementById("agg-chart-caption").textContent = aggregateCaptionText(agg);
  const spec = buildAggregateChartSpec(agg, document.getElementById("agg-legend"));
  aggregateChart = LineChart.create(document.getElementById("agg-chart"), spec);
  renderAggregateTable(agg);
}

// ---------------------------------------------------------------------------
// Выбранный МО: разметка + график + таблицы. seriesId может прийти из адреса,
// из поиска или быть default_mo — все три пути идут через эту функцию.
// ---------------------------------------------------------------------------

let moChart = null;
let selectSeq = 0;

async function selectMo(index, requestedSeriesId, { pushUrl }) {
  // Счётчик выбора: выбор МО из ещё не загруженного региона и следом — другого,
  // уже загруженного, гонит два fetch параллельно, и без счётчика первый ответ,
  // придя позже, перезаписывает график и таблицы под именем уже показанного
  // второго МО. seq, захваченный в замыкании, — метка «это ещё я?».
  const seq = ++selectSeq;

  const known = index.seriesById.has(requestedSeriesId);
  const seriesId = known ? requestedSeriesId : index.default_mo;
  const entry = index.seriesById.get(seriesId);

  let file;
  try {
    file = await fetchMoFile(entry.fileNumber);
  } catch (error) {
    // Пишем в #mo-invalid-note, только если это ещё наш выбор: устаревший провал (второй
    // выбор уже сменил seq, пока этот файл падал) не должен затирать уже показанный
    // результат более свежего запроса сообщением об ошибке.
    if (seq === selectSeq) {
      const note = document.getElementById("mo-invalid-note");
      note.textContent =
        `Не удалось загрузить данные «${seriesId}». Обновите страницу или попробуйте другое.`;
      note.hidden = false;
    }
    throw error; // console.error — в selectMoSafe, единственном месте, откуда вызывают
  }
  if (seq !== selectSeq) return; // выбор сменился, пока грузился файл — наш ответ устарел

  // Всё видимое — шапка, поле поиска, примечания, график, таблицы, адрес — меняется
  // одним куском и только здесь, после await: до этой строки на экране остаётся
  // прежний МО целиком, а не смесь нового имени со старым графиком или наоборот.
  const invalidNote = document.getElementById("mo-invalid-note");
  if (known) {
    invalidNote.hidden = true;
  } else {
    invalidNote.textContent = invalidMoMessage(requestedSeriesId);
    invalidNote.hidden = false;
  }

  document.getElementById("mo-search").value = seriesId;
  document.getElementById("mo-name").textContent = seriesId;
  markQuickChip(seriesId);
  // Заголовок вкладки — с названием текущего МО: у ссылки с ?mo= иначе был бы всегда
  // один и тот же заголовок вне зависимости от того, что на ней открыто.
  document.title = `${seriesId} — прогноз по муниципалитету`;

  const metaEl = document.getElementById("mo-meta");
  const noteEl = document.getElementById("mo-no-region-note");
  if (entry.region) {
    metaEl.textContent = entry.region;
    if (entry.oktmo) {
      // Код ОКТМО — группы через дефис: на узком экране он идёт отдельной строкой целиком
      // (demo.css), а не рвётся посередине, и разделитель перед ним не нужен.
      const separator = document.createElement("span");
      separator.className = "meta-separator";
      separator.textContent = " · ";
      const code = document.createElement("span");
      code.className = "oktmo";
      code.textContent = `ОКТМО ${entry.oktmo}`;
      metaEl.append(separator, code);
    }
    noteEl.hidden = true;
  } else {
    metaEl.textContent = "Регион не определён";
    noteEl.textContent = noRegionExplanationText(seriesId);
    noteEl.hidden = false;
  }

  const mo = file[seriesId];

  const legendEl = document.getElementById("mo-legend");
  const spec = buildMoChartSpec(index, seriesId, mo, legendEl);
  if (moChart) moChart.update(spec);
  else moChart = LineChart.create(document.getElementById("mo-chart"), spec);

  document.getElementById("mo-forecast-caption").textContent = forecastRuleText(index);
  document.getElementById("mo-breaks-caption").textContent = breaksListText(mo, index.breaks);
  document.getElementById("errors-caption").textContent = errorsCaptionText(index);

  renderMoNumbersTable(index, mo);
  renderErrorsTable(index, mo);

  // Повторный выбор того же МО (кнопка и поиск это допускают) не добавляет в историю ещё
  // одну запись: иначе «назад» оставалось бы на той же странице.
  if (pushUrl && getMoParam() !== seriesId) pushMoParam(seriesId);
}

function selectMoSafe(index, seriesId, opts) {
  // Текст ошибки — уже внутри selectMo (только если сбой ещё относится к текущему
  // выбору); здесь — журнал для отладки, второй раз DOM не трогаем.
  selectMo(index, seriesId, opts).catch((error) => {
    console.error("Не удалось показать муниципальное образование", error);
  });
}

// ---------------------------------------------------------------------------
// Быстрые кнопки: несколько МО одним нажатием. Список и подписи — index.quick (константа
// генератора), выбор — тем же путём, что из поиска, поэтому адрес ?mo=, кнопка «назад»
// и защита от гонки работают так же.
// ---------------------------------------------------------------------------

function setupQuickChips(index) {
  const box = document.getElementById("quick-chips");
  (index.quick || []).forEach((item) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = item.short;
    // Полное название — подсказкой при наведении: короткая подпись («Михайловский р-н #2»)
    // не говорит, какой именно ряд за ней стоит.
    chip.title = item.id;
    chip.dataset.mo = item.id;
    chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("click", () => selectMoSafe(index, item.id, { pushUrl: true }));
    box.appendChild(chip);
  });
}

/** Нажатой выглядит кнопка того МО, которое показано сейчас, — а не того, что выбрано
 * и ещё грузится: вызывается из selectMo после смены содержимого страницы. */
function markQuickChip(seriesId) {
  document.getElementById("quick-chips").querySelectorAll(".chip").forEach((chip) => {
    chip.setAttribute("aria-pressed", String(chip.dataset.mo === seriesId));
  });
}

// ---------------------------------------------------------------------------
// Шапка: на телефоне меню свёрнуто, на широком экране ссылки всегда видны. Та же логика,
// что в site/landing.js: скрипты страниц друг друга не подключают.
// ---------------------------------------------------------------------------

function setupNav() {
  const menu = document.getElementById("nav-menu");
  if (!menu || !window.matchMedia) return;
  const narrow = window.matchMedia("(max-width: 839px)");
  const sync = () => { menu.open = !narrow.matches; };
  sync();
  if (narrow.addEventListener) narrow.addEventListener("change", sync);
  menu.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && narrow.matches && menu.open) {
      menu.open = false;
      const toggle = menu.querySelector("summary");
      if (toggle) toggle.focus();
    }
  });
  document.addEventListener("click", (event) => {
    if (narrow.matches && menu.open && !menu.contains(event.target)) menu.open = false;
  });
}

// ---------------------------------------------------------------------------
// Поиск: комбобокс + listbox по образцу ARIA (WAI-ARIA APG combobox).
// ---------------------------------------------------------------------------

function setupCombobox(index) {
  const input = document.getElementById("mo-search");
  const listbox = document.getElementById("mo-listbox");
  const statusEl = document.getElementById("mo-search-status");
  const RESULT_LIMIT = 8;
  let currentOptions = [];
  let activeIndex = -1;

  function openListbox() {
    listbox.hidden = false;
    input.setAttribute("aria-expanded", "true");
    // На телефоне список уходит за нижний край окна: докручиваем ровно настолько, чтобы он влез
    // (так же, как на главной).
    listbox.scrollIntoView({ block: "nearest" });
  }

  function closeListbox() {
    listbox.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    activeIndex = -1;
  }

  function setActive(i) {
    const options = listbox.querySelectorAll('[role="option"]');
    options.forEach((el) => el.setAttribute("aria-selected", "false"));
    activeIndex = i;
    if (i >= 0 && i < options.length) {
      options[i].setAttribute("aria-selected", "true");
      input.setAttribute("aria-activedescendant", options[i].id);
      options[i].scrollIntoView({ block: "nearest" });
    } else {
      input.removeAttribute("aria-activedescendant");
    }
  }

  function commitSelection(seriesId) {
    closeListbox();
    selectMoSafe(index, seriesId, { pushUrl: true });
  }

  // Короткий итог для живой области — не сам список: подробности и так на экране
  // в listbox, а на каждую букву запроса произносить их скринридеру целиком было бы
  // избыточно.
  function matchesStatusText(total) {
    if (!total) return "Ничего не нашлось";
    return `Найдено ${rubFmt(total)} ${pluralRu(total, "совпадение", "совпадения", "совпадений")}`;
  }

  function noMatchesText() {
    // Число — то же, что во вступлении к поиску (index.n_series): в панель вошли
    // только ряды без длинных пропусков, и это стоит сказать здесь тоже, а не
    // только один раз наверху страницы.
    const total = index.n_series;
    return "Ничего не нашлось — попробуйте часть названия или регион. В список вошли " +
      `${rubFmt(total)} ${pluralRu(total, "муниципальное образование", "муниципальных образования", "муниципальных образований")} ` +
      "без длинных пропусков в данных — остальные в панель не попали.";
  }

  function renderOptions(query) {
    listbox.innerHTML = "";
    if (!query.trim()) { closeListbox(); currentOptions = []; statusEl.textContent = ""; return; }

    const { matches, total } = searchMunicipalities(query, index.searchRows, RESULT_LIMIT);
    currentOptions = matches;
    activeIndex = -1;
    statusEl.textContent = matchesStatusText(total);

    if (!matches.length) {
      const li = document.createElement("li");
      li.className = "suggestion-status";
      // role="option" + aria-disabled, а не текст без роли внутри role="listbox" —
      // так строка остаётся допустимым потомком listbox. currentOptions пуст,
      // поэтому ни стрелки, ни Enter её всё равно не выберут.
      li.setAttribute("role", "option");
      li.setAttribute("aria-disabled", "true");
      li.textContent = noMatchesText();
      listbox.appendChild(li);
      openListbox();
      return;
    }

    matches.forEach((row, i) => {
      const li = document.createElement("li");
      li.id = `mo-option-${i}`;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", "false");
      li.className = "suggestion";
      const name = document.createElement("span");
      name.className = "suggestion-name";
      name.textContent = row.seriesId;
      const region = document.createElement("span");
      region.className = "suggestion-region";
      region.textContent = regionLabel(row);
      li.append(name, region);
      li.addEventListener("click", () => commitSelection(row.seriesId));
      listbox.appendChild(li);
    });

    if (total > matches.length) {
      const rest = total - matches.length;
      const li = document.createElement("li");
      li.className = "suggestion-status";
      li.setAttribute("role", "option");
      li.setAttribute("aria-disabled", "true");
      li.textContent = `И ещё ${rubFmt(rest)} ${pluralRu(rest, "совпадение", "совпадения", "совпадений")} — уточните запрос`;
      listbox.appendChild(li);
    }
    openListbox();
  }

  // mousedown до click — иначе фокус успевает уйти с поля до выбора пункта.
  listbox.addEventListener("mousedown", (event) => event.preventDefault());

  input.addEventListener("input", () => renderOptions(input.value));
  input.addEventListener("focus", () => { if (input.value.trim()) renderOptions(input.value); });
  // Уход с поля клавишей Tab закрывает список: следом идут быстрые кнопки, и открытый
  // список закрывал бы их от глаз. Выбор мышью поле не покидает — см. mousedown выше.
  input.addEventListener("blur", closeListbox);

  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (listbox.hidden) renderOptions(input.value);
      else setActive(Math.min(currentOptions.length - 1, activeIndex + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      if (!listbox.hidden) setActive(Math.max(0, activeIndex - 1));
    } else if (event.key === "Enter") {
      if (!listbox.hidden && currentOptions.length) {
        event.preventDefault();
        commitSelection(currentOptions[activeIndex >= 0 ? activeIndex : 0].seriesId);
      }
    } else if (event.key === "Escape") {
      closeListbox();
    }
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search")) closeListbox();
  });
}

// ---------------------------------------------------------------------------
// Инициализация.
// ---------------------------------------------------------------------------

async function init() {
  const index = await fetchJson("data/index.json");
  index.searchRows = buildSearchIndex(index.series, index.quick, index.mean_unit);
  index.seriesById = new Map(index.searchRows.map((row) => [row.seriesId, row]));

  document.getElementById("intro-text").textContent = introText(index);
  document.getElementById("search-hint").textContent = searchHintText(index);
  document.getElementById("footer-built").textContent = `собрано ${dateWordsFromISO(index.built)}`;

  setupCombobox(index);
  setupQuickChips(index);

  window.addEventListener("popstate", (event) => {
    const seriesId = (event.state && event.state.mo) || getMoParam() || index.default_mo;
    selectMoSafe(index, seriesId, { pushUrl: false });
  });

  // Агрегат и выбранное МО грузятся каждый своим fetch и не зависят друг от друга:
  // иначе сбой при загрузке первого МО оставлял бы блок агрегата на «Загрузка
  // данных…» навсегда — до renderAggregateBlock() код не доходил бы. Первый выбор
  // МО идёт тем же safe-путём (selectMoSafe), что и последующие — из поиска и popstate.
  renderAggregateBlock().catch((error) => {
    console.error("Не удалось построить блок федерального агрегата", error);
  });
  selectMoSafe(index, getMoParam() || index.default_mo, { pushUrl: false });
}

// Меню шапки не ждёт данных: сбой загрузки не должен оставлять его раскрытым на телефоне.
setupNav();

init().catch((error) => {
  console.error("Не удалось инициализировать страницу", error);
  const note = document.getElementById("mo-invalid-note");
  if (note) {
    note.textContent = "Не удалось загрузить данные. Обновите страницу.";
    note.hidden = false;
  }
});
