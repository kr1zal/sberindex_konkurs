/* demo/demo.js — демонстрационный стенд: загрузка данных, поиск МО, адрес страницы,
 * сборка блоков. Числа и подписи графиков рисует общий demo/linechart.js — этот файл
 * отвечает за то, ОТКУДА берутся ряды для него и что написано на странице.
 *
 * Видимый текст живёт в двух местах: разметка demo/index.html (то, что не зависит
 * от данных) и функции ниже с суффиксом Text (то, что зависит — числа, названия
 * моделей, месяцы). Числа в разметке не пишем — только через эти функции, из JSON.
 */
"use strict";

// ---------------------------------------------------------------------------
// Числа и месяцы — форматирование как в отчёте (report/report.qmd, scripts/build_site.py):
// неразрывный пробел в разрядах, запятая, минус «−», проценты — с одним знаком.
// ---------------------------------------------------------------------------

function groupThousands(digitsStr) {
  let s = digitsStr;
  const groups = [];
  while (s.length > 3) {
    groups.unshift(s.slice(-3));
    s = s.slice(0, -3);
  }
  groups.unshift(s);
  return groups.join(" ");
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

function pctFmt(value, opts) {
  const formatted = numFmt(value, 1, opts);
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
// Предложный падеж («в январе») — как _MONTH_IN в report/report.qmd.
const MONTH_PREP = ["январе", "феврале", "марте", "апреле", "мае", "июне",
                     "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"];

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

function monthPrepYear(ym) {
  const { y, m } = parseYM(ym);
  return `${MONTH_PREP[m - 1]} ${y}`;
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
// Роли моделей, детектор изломов — короткие словари для текста. Названия самих
// моделей (полные, для второй, приглушённой строки) генератор кладёт прямо
// в данные — models[].label и forecast_rule[].label (scripts/build_site.py,
// словарь MODEL_LABELS) — второго, JS-словаря с тем же текстом не держим.
// ---------------------------------------------------------------------------

// Короткая, первая строка ячейки таблицы ошибок — по роли, не по полному названию
// модели (то — второй, приглушённой строкой, model.label). Roles.reference — всегда
// prophet (см. build_site.py::_build_model_roles), поэтому «(Prophet)» — не догадка.
const ROLE_SHORT_LABELS = {
  reference: "эталон (Prophet)",
  naive: "наивная",
  recommended: "рекомендуемая панельная",
  best_mean: "лучшая по среднему",
  two_stage: "двухэтапная",
};

// role — строка через «+» (see: models[].role в докстринге build_site.py), а не
// список: одна модель может занимать несколько ролей сразу.
function roleShortLabel(roleStr) {
  return roleStr.split("+").map((r) => ROLE_SHORT_LABELS[r] || r).join(" + ");
}

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

function breaksMethodologyText(breaksInfo) {
  const det = CP_DETECTOR_LABELS[breaksInfo.detector] || breaksInfo.detector;
  const on = CP_MODE_LABELS_ON[breaksInfo.mode] || breaksInfo.mode;
  return "Изломы найдены задним числом по полному ряду — не сигнал в реальном времени: " +
    `детектор ${det} ${on}, штраф ${formatPenalty(breaksInfo.penalty)}.`;
}

// ---------------------------------------------------------------------------
// Поиск МО: без учёта регистра, «ё» = «е», по всем словам запроса в строке
// «название + регион»; сначала — совпадения с начала названия.
// ---------------------------------------------------------------------------

function normalizeSearchText(s) {
  return s.toLowerCase().replace(/ё/g, "е");
}

function buildSearchIndex(seriesRows) {
  // seriesRows — index.json.series: [[series_id, регион|null, ОКТМО|null, файл], ...]
  return seriesRows.map(([seriesId, region, oktmo, fileNumber]) => ({
    seriesId, region, oktmo, fileNumber,
    nameNorm: normalizeSearchText(seriesId),
    haystack: normalizeSearchText(region ? `${seriesId} ${region}` : seriesId),
  }));
}

function searchMunicipalities(query, rows, limit) {
  const q = normalizeSearchText(query.trim());
  if (!q) return { matches: [], total: 0 };
  const words = q.split(/\s+/).filter(Boolean);
  const tierA = [];
  const tierB = [];
  for (const row of rows) {
    if (!words.every((w) => row.haystack.includes(w))) continue;
    (row.nameNorm.startsWith(q) ? tierA : tierB).push(row);
  }
  const collator = new Intl.Collator("ru");
  const byName = (a, b) => collator.compare(a.seriesId, b.seriesId);
  tierA.sort(byName);
  tierB.sort(byName);
  const all = tierA.concat(tierB);
  return { matches: all.slice(0, limit), total: all.length };
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
    moFileCache.set(fileNumber, fetchJson(`data/mo/${fileNumber}.json`));
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
  const start = monthPrepYear(firstFold.test_from);
  const end = monthPrepYear(lastFold.test_to);
  return `Прогноз потребительских расходов выбранного муниципального образования на ${forecastYear} год ` +
    `и то, как разные модели ошибались на нём в ${start} — ${end}.`;
}

function searchHintText(index) {
  const total = index.n_series;
  const noRegion = index.n_no_region;
  const names = index.n_homonym_names;
  const series = index.n_homonym_series;
  return `Всего в панели ${rubFmt(total)} ${pluralRu(total, "муниципальное образование", "муниципальных образования", "муниципальных образований")}. ` +
    `У ${rubFmt(noRegion)} из них регион не определён — среди них ${rubFmt(series)} ${pluralRu(series, "ряд делит", "ряда делят", "рядов делят")} ` +
    `${rubFmt(names)} повторяющихся ${pluralRu(names, "название", "названия", "названий")}.`;
}

function invalidMoMessage(seriesId) {
  return `Ряда «${seriesId}» нет в данных стенда — показан муниципалитет по умолчанию.`;
}

function noRegionExplanationText(seriesId) {
  let text = "Регион не определён: в выгрузке СберИндекса у этого муниципального образования " +
    "есть только название, а его носят муниципалитеты нескольких регионов (либо этого названия " +
    "нет в справочнике СберИндекса) — приписывать регион наугад мы не стали.";
  if (/ #\d+$/.test(seriesId)) {
    text += " «#N» в конце названия — порядковый номер одноимённого ряда в выгрузке.";
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
  // Не «одна модель на всё» — сколько их на самом деле, видно по частям ниже.
  return `Прогноз собран по нескольким горизонтам, не всегда одной моделью: ${parts.join("; ")}. ` +
    `Муниципальный разрез ${forecastYear} года СберИндекс ещё не опубликовал — прогноз по этому МО фактом не проверен.`;
}

function breaksListText(mo, breaksInfo) {
  if (!mo.breaks.length) {
    return `Изломов при этом штрафе не найдено. ${breaksMethodologyText(breaksInfo)}`;
  }
  const months = mo.breaks.map((ym) => monthNomYear(ym)).join(", ");
  return `Изломы: ${months}. ${breaksMethodologyText(breaksInfo)}`;
}

function errorsCaptionText(index) {
  const testLen = monthsBetweenInclusive(index.folds[0].test_from, index.folds[0].test_to);
  return `MAE — средняя абсолютная ошибка за ${testLen} ${pluralRu(testLen, "месяц", "месяца", "месяцев")} теста, ` +
    `в ${index.unit}. Мельче и бледнее под числом этого МО — то же по всей панели, для масштаба.`;
}

function aggregateLeadText(agg) {
  const forecastYear = parseYM(agg.check.months[0]).y;
  return `Прогноз первого этапа двухэтапной модели от ${monthGenYear(agg.origin)} на ${forecastYear} год — ` +
    "единственный прогноз этой работы, который уже можно сверить с фактом. Он ошибается в среднем за год " +
    `на ${pctFmt(Math.abs(agg.mape.two_stage))}, против ${pctFmt(Math.abs(agg.mape.naive))} у правила ` +
    `«${agg.rule_names.naive}» и ${pctFmt(Math.abs(agg.mape.seasonal_naive))} у правила «${agg.rule_names.seasonal_naive}».`;
}

function aggregateCaptionText(agg) {
  // «по» в значении «включительно по» берёт тот же падеж, что «в» (винительный,
  // у месяцев он совпадает с именительным) — не родительный, как после «от».
  return `История федерального ряда по ${monthNomYear(agg.origin)}, факт ${parseYM(agg.check.months[0]).y} года ` +
    `поверх неё, прогноз первого этапа (модель «${agg.model_label}») от конца панели и два простых правила ` +
    `для сравнения. Единица — ${agg.unit}.`;
}

// ---------------------------------------------------------------------------
// График выбранного МО.
// ---------------------------------------------------------------------------

function buildMoChartSpec(index, seriesId, mo, legendEl) {
  const months = index.panel_months.concat(index.forecast_months);
  const nPanel = index.panel_months.length;

  const factValues = new Array(months.length).fill(null);
  mo.fact.forEach((v, i) => { factValues[i] = v; });

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
      { id: "forecast", label: "Прогноз (рекомендуемая модель)", color: "var(--chart-forecast)", values: forecastValues },
      {
        id: "known", label: "Если бы федеральный индекс за месяц уже был опубликован",
        color: "var(--chart-known)", dash: "dashed", values: knownValues,
      },
    ],
    markers,
    yFormat: rubFmt,
    yUnit: index.unit,
    ariaLabel: `Линейный график: факт расходов «${seriesId}» за ${monthNomYear(months[0])} — ` +
      `${monthNomYear(months[nPanel - 1])} и прогноз на ${monthNomYear(months[nPanel])} — ` +
      `${monthNomYear(months[months.length - 1])}, ${index.unit}.`,
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
  ["Месяц", "Факт", "Прогноз", "Разнос агрегата"].forEach((text) => {
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
  const foldLabels = index.folds.map((fold) => monthRangeAxis(fold.test_from, fold.test_to));

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const thModel = document.createElement("th");
  thModel.textContent = "Модель";
  headRow.appendChild(thModel);
  index.folds.forEach((fold, i) => {
    const th = document.createElement("th");
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
  thMean.textContent = "Среднее";
  headRow.appendChild(thMean);
  thead.appendChild(headRow);

  const means = modelMeansForMo(index, mo);
  const best = bestPerColumn(index, means);
  const panelMae = index.panel_mae;

  const tbody = document.createElement("tbody");
  index.models.forEach((model) => {
    const tr = document.createElement("tr");
    const rowTh = document.createElement("th");
    rowTh.setAttribute("scope", "row");
    // Короткая роль — первой, жирной строкой; полное название модели — второй,
    // приглушённой (на узком экране скрыта стилем, см. demo.css) — без прежнего
    // дублирования, когда роль повторялась и в названии модели, и в скобках.
    const primary = document.createElement("span");
    primary.className = "role-primary";
    primary.textContent = roleShortLabel(model.role);
    const secondary = document.createElement("span");
    secondary.className = "cell-sub role-desc";
    secondary.textContent = model.label;
    rowTh.append(primary, secondary);
    tr.appendChild(rowTh);

    const { perFold, mean } = means[model.id];
    const panelRow = panelMae[model.id] || [null, null, null, null];

    perFold.forEach((v, col) => {
      tr.appendChild(errorCell(v, panelRow[col], v !== null && v === best[col], foldLabels[col]));
    });
    tr.appendChild(errorCell(mean, panelRow[3], mean !== null && mean === best[nFolds], "Среднее"));
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
}

function errorCell(value, panelValue, isBest, narrowLabel) {
  const td = document.createElement("td");
  const classes = [];
  if (isBest) classes.push("cell-best");
  if (value === null) classes.push("cell-dash");
  if (classes.length) td.className = classes.join(" ");
  // На узком экране таблица становится карточками (demo.css): data-label — это
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

  const series = [
    { id: "fact", label: "Факт", color: "var(--chart-fact)", values: factLine },
    { id: "fact-check", label: "Факт", color: "var(--chart-fact)", mode: "dots", values: factDots, overlay: true },
    { id: "two_stage", label: "Прогноз первого этапа", color: "var(--chart-forecast)", values: ruleValues("two_stage") },
    {
      id: "naive", label: `Простое правило «${agg.rule_names.naive}»`,
      color: "var(--chart-known)", dash: "dashed", values: ruleValues("naive"),
    },
    {
      id: "seasonal_naive", label: `Простое правило «${agg.rule_names.seasonal_naive}»`,
      color: "var(--chart-third)", dash: "dotted", values: ruleValues("seasonal_naive"),
    },
  ];
  const markers = [{ atMonth: agg.origin, kind: "end", label: "конец панели" }];

  return {
    // На оси и в подсказке — целые (деления и так круглые числа, сотые ни к чему);
    // точные сотые — только в таблице ниже (renderAggregateTable), где это уже число, а не деление.
    months, series, markers, yFormat: rubFmt, yUnit: agg.unit,
    ariaLabel: `Линейный график: федеральный агрегат потребительских расходов, история по ` +
      `${monthGenYear(agg.origin)}, факт ${parseYM(agg.check.months[0]).y} года и прогноз первого этапа ` +
      `двухэтапной модели против двух простых правил, ${agg.unit}.`,
    legendEl,
  };
}

function renderAggregateTable(agg) {
  // Как в таблице отчёта (report/report.qmd, ~2148-2163): факт и прогноз целыми
  // (rub), ошибка — только в процентах (свой знак у каждой из трёх), без рублёвого
  // разноса и рублёвой средней ошибки — они здесь ничего не добавляют к процентам.
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
      rubFmt(agg.check.actual[i]),
      rubFmt(agg.check.forecast.two_stage[i]),
      pctFmt(agg.check.error_pct.two_stage[i], { sign: true }),
      pctFmt(agg.check.error_pct.naive[i], { sign: true }),
      pctFmt(agg.check.error_pct.seasonal_naive[i], { sign: true }),
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
    pctFmt(agg.mape.two_stage), pctFmt(agg.mape.naive), pctFmt(agg.mape.seasonal_naive),
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

async function selectMo(index, requestedSeriesId, { pushUrl }) {
  const invalidNote = document.getElementById("mo-invalid-note");
  const known = index.seriesById.has(requestedSeriesId);
  const seriesId = known ? requestedSeriesId : index.default_mo;
  if (known) {
    invalidNote.hidden = true;
  } else {
    invalidNote.textContent = invalidMoMessage(requestedSeriesId);
    invalidNote.hidden = false;
  }

  const entry = index.seriesById.get(seriesId);
  document.getElementById("mo-search").value = seriesId;
  document.getElementById("mo-name").textContent = seriesId;

  const metaEl = document.getElementById("mo-meta");
  const noteEl = document.getElementById("mo-no-region-note");
  if (entry.region) {
    metaEl.textContent = entry.oktmo ? `${entry.region}, ОКТМО ${entry.oktmo}` : entry.region;
    noteEl.hidden = true;
  } else {
    metaEl.textContent = "Регион не определён";
    noteEl.textContent = noRegionExplanationText(seriesId);
    noteEl.hidden = false;
  }

  const file = await fetchMoFile(entry.fileNumber);
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

  if (pushUrl) pushMoParam(seriesId);
}

function selectMoSafe(index, seriesId, opts) {
  selectMo(index, seriesId, opts).catch((error) => {
    console.error("Не удалось показать муниципальное образование", error);
  });
}

// ---------------------------------------------------------------------------
// Поиск: комбобокс + listbox по образцу ARIA (WAI-ARIA APG combobox).
// ---------------------------------------------------------------------------

function setupCombobox(index) {
  const input = document.getElementById("mo-search");
  const listbox = document.getElementById("mo-listbox");
  const RESULT_LIMIT = 8;
  let currentOptions = [];
  let activeIndex = -1;

  function openListbox() {
    listbox.hidden = false;
    input.setAttribute("aria-expanded", "true");
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

  function renderOptions(query) {
    listbox.innerHTML = "";
    if (!query.trim()) { closeListbox(); currentOptions = []; return; }

    const { matches, total } = searchMunicipalities(query, index.searchRows, RESULT_LIMIT);
    currentOptions = matches;
    activeIndex = -1;

    if (!matches.length) {
      const li = document.createElement("li");
      li.className = "listbox-status";
      li.textContent = "Ничего не найдено";
      listbox.appendChild(li);
      openListbox();
      return;
    }

    matches.forEach((row, i) => {
      const li = document.createElement("li");
      li.id = `mo-option-${i}`;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", "false");
      li.className = "listbox-option";
      const name = document.createElement("span");
      name.textContent = row.seriesId;
      const region = document.createElement("span");
      region.className = "option-region";
      region.textContent = row.region || "регион не определён";
      li.append(name, region);
      li.addEventListener("click", () => commitSelection(row.seriesId));
      listbox.appendChild(li);
    });

    if (total > matches.length) {
      const rest = total - matches.length;
      const li = document.createElement("li");
      li.className = "listbox-status";
      li.textContent = `И ещё ${rubFmt(rest)} ${pluralRu(rest, "совпадение", "совпадения", "совпадений")}`;
      listbox.appendChild(li);
    }
    openListbox();
  }

  // mousedown до click — иначе фокус успевает уйти с поля до выбора пункта.
  listbox.addEventListener("mousedown", (event) => event.preventDefault());

  input.addEventListener("input", () => renderOptions(input.value));
  input.addEventListener("focus", () => { if (input.value.trim()) renderOptions(input.value); });

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
    if (!event.target.closest(".combobox")) closeListbox();
  });
}

// ---------------------------------------------------------------------------
// Инициализация.
// ---------------------------------------------------------------------------

async function init() {
  const index = await fetchJson("data/index.json");
  index.searchRows = buildSearchIndex(index.series);
  index.seriesById = new Map(index.searchRows.map((row) => [row.seriesId, row]));

  document.getElementById("intro-text").textContent = introText(index);
  document.getElementById("search-hint").textContent = searchHintText(index);
  document.getElementById("footer-built").textContent = `Собрано ${dateWordsFromISO(index.built)}.`;

  setupCombobox(index);

  window.addEventListener("popstate", (event) => {
    const seriesId = (event.state && event.state.mo) || getMoParam() || index.default_mo;
    selectMoSafe(index, seriesId, { pushUrl: false });
  });

  await selectMo(index, getMoParam() || index.default_mo, { pushUrl: false });
  renderAggregateBlock().catch((error) => {
    console.error("Не удалось построить блок федерального агрегата", error);
  });
}

init().catch((error) => {
  console.error("Не удалось инициализировать стенд", error);
  const note = document.getElementById("mo-invalid-note");
  if (note) {
    note.textContent = "Не удалось загрузить данные стенда. Проверьте консоль браузера.";
    note.hidden = false;
  }
});
