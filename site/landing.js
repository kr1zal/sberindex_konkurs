/* site/landing.js — главная: графики, счётчики чисел, переключатели и поиск поверх данных
 * из <script id="landing-data">. Данные и все числа приходят от генератора
 * (scripts/build_site.py, формат — его докстринг); здесь только отрисовка, форматирование
 * и подписи. Ни одно число данных в этом файле не пишется: ни ряды, ни доли, ни проценты.
 *
 * Библиотек нет: SVG собирается вручную. Положение линий — в единицах viewBox 0…1000,
 * подписи и точки — в процентах блока графика (раскладка — site/landing.css).
 * При prefers-reduced-motion всё показывается сразу, без анимаций.
 *
 * Всё, что не касается страницы (шкалы, подписи оси, разбор чисел, ранжирование поиска), живёт
 * в site/landing-lib.js и подключается перед этим файлом: там его проверяют тесты через node.
 */
(function () {
  "use strict";

  const {
    formatInt, pluralRu, niceScale, formatTick, percentTick, createTickSets, historyKeep,
    numberTokens, countedText, rowsFromIndex, searchRows, pickTarget,
  } = LandingLib;

  let data;
  try {
    data = JSON.parse(document.getElementById("landing-data").textContent);
  } catch (error) {
    console.error("Не удалось прочитать данные главной", error);
    return;
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  const NBSP = "\u00a0";
  const VIEW = 1000; // сторона viewBox графиков
  const MONTH_ABBR = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
  const MONTH_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];

  // ---------------------------------------------------------------------------
  // Общие помощники
  // ---------------------------------------------------------------------------

  const byId = (id) => document.getElementById(id);

  function reducedMotion() {
    return Boolean(window.matchMedia) && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function node(tag, className, text, parent) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== null && text !== undefined) element.textContent = text;
    if (parent) parent.appendChild(element);
    return element;
  }

  function svgNode(tag, attributes, parent) {
    const element = document.createElementNS(SVG_NS, tag);
    Object.keys(attributes || {}).forEach((name) => element.setAttribute(name, attributes[name]));
    if (parent) parent.appendChild(element);
    return element;
  }

  // «ГГГГ-ММ» → «янв ГГГГ».
  function monthLabel(ym) {
    return `${MONTH_ABBR[Number(ym.slice(5, 7)) - 1]} ${ym.slice(0, 4)}`;
  }

  function extent(rows) {
    let min = Infinity;
    let max = -Infinity;
    rows.forEach((row) => row.forEach((value) => {
      if (value === null || value === undefined) return;
      if (value < min) min = value;
      if (value > max) max = value;
    }));
    return { min, max };
  }

  const spacedX = (count) => Array.from({ length: count }, (_, i) => (i / (count - 1)) * VIEW);
  const yOf = (value, scale) => (1 - (value - scale.lo) / (scale.hi - scale.lo)) * VIEW;
  const percentTop = (value, scale) => ((scale.hi - value) / (scale.hi - scale.lo)) * 100;

  function linePath(ys, xs) {
    let d = "";
    for (let i = 0; i < ys.length; i += 1) {
      d += `${i ? "L" : "M"}${xs[i].toFixed(1)},${ys[i].toFixed(1)}`;
    }
    return d;
  }

  const easeOut = (p, power) => 1 - Math.pow(1 - p, power);

  // Покадровая анимация: step(p) получает прогресс 0…1. Без анимаций (системная настройка)
  // сразу один кадр с p = 1.
  function animate(duration, step, done) {
    if (reducedMotion() || duration <= 0) {
      step(1);
      if (done) done();
      return { cancel() {} };
    }
    let frame = 0;
    let start = null;
    let cancelled = false;
    function tick(now) {
      if (cancelled) return;
      if (start === null) start = now;
      const progress = Math.min(1, (now - start) / duration);
      step(progress);
      if (progress < 1) frame = requestAnimationFrame(tick);
      else if (done) done();
    }
    frame = requestAnimationFrame(tick);
    return { cancel() { cancelled = true; cancelAnimationFrame(frame); } };
  }

  // Подписи оси: новый набор проявляется, все прежние гаснут и убираются. Какие наборы живые,
  // какие уходят, помнит модель (LandingLib.createTickSets): так быстрые клики подряд не оставляют
  // на оси второй шкалы, а элементы находятся по номеру набора, а не по порядку в разметке.
  const tickModels = new WeakMap();
  const TICK_FADE_MS = 450;

  function setTicks(box, scale, label, swap) {
    if (!tickModels.has(box)) tickModels.set(box, createTickSets());
    const model = tickModels.get(box);
    const animated = Boolean(swap) && !reducedMotion();
    const plan = model.show(animated);
    const setOf = (id) => box.querySelector(`.tick-set[data-set="${id}"]`);
    const drop = (id) => {
      const old = setOf(id);
      if (old && old.parentNode) old.parentNode.removeChild(old);
    };

    const set = node("div", "tick-set", null, box);
    set.dataset.set = String(plan.added);
    scale.ticks.forEach((value) => {
      const tick = node("div", "tick", null, set);
      tick.style.top = `${percentTop(value, scale).toFixed(2)}%`;
      node("span", null, label(value), tick);
    });
    plan.removed.forEach(drop);
    if (!animated) return;

    set.classList.add("is-entering");
    // Два кадра: стартовое состояние должно успеть примениться, иначе проявления не будет.
    requestAnimationFrame(() => requestAnimationFrame(() => set.classList.remove("is-entering")));
    plan.leaving.forEach((id) => {
      const old = setOf(id);
      if (old) old.classList.add("is-leaving");
      setTimeout(() => {
        model.finish(id);
        drop(id);
      }, TICK_FADE_MS);
    });
  }

  // Подписи месяцев по оси X: у левого края, в заданной точке и у правого.
  function setAxisLabels(box, labels, count) {
    box.textContent = "";
    labels.forEach((item) => {
      const span = node("span", null, item.text, box);
      if (item.index <= 0) {
        span.style.left = "0";
      } else if (item.index >= count - 1) {
        span.style.right = "0";
      } else {
        span.style.left = `${((item.index / (count - 1)) * 100).toFixed(2)}%`;
        span.style.transform = "translateX(-50%)";
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Шапка: на телефоне меню свёрнуто, на широком экране — всегда раскрыто
  // ---------------------------------------------------------------------------

  function initNav() {
    const menu = byId("nav-menu");
    if (!menu || !window.matchMedia) return;
    const narrow = window.matchMedia("(max-width: 719px)");
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
  // Обложка: 30 линий прорисовываются со сдвигом, затем общее движение
  // ---------------------------------------------------------------------------

  function initHero(story) {
    const lines = byId("hero-lines");
    const median = byId("hero-median");
    if (!lines || !median) return;
    const count = story.months.length;
    const xs = spacedX(count);
    const { min, max } = extent(story.series.concat([story.median]));
    const pad = (max - min) * 0.06;
    const scale = { lo: min - pad, hi: max + pad };
    story.series.forEach((row, index) => {
      const path = svgNode("path", { class: "hero-line", d: linePath(row.map((v) => yOf(v, scale)), xs) }, lines);
      path.style.setProperty("--i", index);
    });
    median.setAttribute("d", linePath(story.median.map((v) => yOf(v, scale)), xs));

    const axis = byId("hero-axis");
    if (axis) {
      setAxisLabels(axis, [
        { index: 0, text: monthLabel(story.months[0]) },
        { index: story.base_months, text: monthLabel(story.months[story.base_months]) },
        { index: count - 1, text: monthLabel(story.months[count - 1]) },
      ], count);
    }
  }

  // ---------------------------------------------------------------------------
  // Четыре числа в карточках: при появлении в окне — счёт от нуля, в конце — исходный текст
  // ---------------------------------------------------------------------------

  function countUp(element) {
    const original = element.textContent;
    const tokens = numberTokens(original);
    if (!tokens.length) return;
    animate(1500, (progress) => {
      element.textContent = countedText(original, tokens, easeOut(progress, 3));
    }, () => {
      element.textContent = original;
    });
  }

  // Строка панели («N рядов × M месяцев») не считается: на полпути формы слов не пересчитываются,
  // и выходило «946 рядов × 11 месяца». Считаются числа карточек.
  function initCounters() {
    const numbers = document.querySelectorAll(".stat:not(.stat-panel) .stat-number");
    if (!numbers.length || reducedMotion() || !("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        countUp(entry.target);
      });
    }, { threshold: 0.4 });
    numbers.forEach((element) => observer.observe(element));
  }

  // ---------------------------------------------------------------------------
  // 01 · Почему это прогноз одного числа: три шага над одними и теми же рядами
  // ---------------------------------------------------------------------------

  function initStory(story) {
    const plot = byId("story-plot");
    const linesLayer = byId("story-lines");
    const medianLayer = byId("story-median-layer");
    const medianPath = byId("story-median");
    const ticksBox = byId("story-ticks");
    const caption = byId("story-caption");
    const text = byId("story-text");
    const buttons = Array.from(document.querySelectorAll(".step"));
    if (!plot || !linesLayer || !medianPath || !ticksBox || !buttons.length) return;

    const count = story.months.length;
    const xs = spacedX(count);
    const paths = story.series.map(() => svgNode("path", {}, linesLayer));

    // Шаг 1 — ряды, шаг 2 — те же ряды бледнеют и видна медиана, шаг 3 — каждый ряд,
    // делённый на медиану, и ровная линия на единице: на ней все ряды были бы, если бы
    // двигались только вместе. Шкала у каждого шага своя, по его значениям.
    function buildState(step) {
      const rows = story.series.map((row) => (step === 2 ? row.map((v, i) => v / story.median[i]) : row));
      const flat = story.median.map((v) => (step === 2 ? 1 : v));
      const { min, max } = extent(rows.concat([flat]));
      const scale = niceScale(min, max, 5);
      return {
        ys: rows.map((row) => row.map((v) => yOf(v, scale))),
        median: flat.map((v) => yOf(v, scale)),
        scale,
        faded: step === 1,
        medianVisible: step !== 0,
      };
    }

    const states = [0, 1, 2].map(buildState);
    let shown = states[0];
    let active = 0;
    let running = null;

    function draw(state) {
      state.ys.forEach((ys, index) => paths[index].setAttribute("d", linePath(ys, xs)));
      medianPath.setAttribute("d", linePath(state.median, xs));
      shown = state;
    }

    // Координаты точек между двумя состояниями: линии перетекают, а не перескакивают.
    function mix(from, to, k) {
      return {
        ys: from.ys.map((row, j) => row.map((y, i) => y + (to.ys[j][i] - y) * k)),
        median: from.median.map((y, i) => y + (to.median[i] - y) * k),
      };
    }

    function stepText(step) {
      return buttons[step].querySelector(".step-text").textContent.replace(/\s+/g, " ").trim();
    }

    function select(step, animateMove) {
      const target = states[step];
      const from = shown;
      if (running) running.cancel();
      active = step;
      buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === step)));
      const captionText = buttons[step].getAttribute("data-caption");
      caption.textContent = captionText;
      plot.setAttribute("aria-label", captionText);
      text.textContent = stepText(step);
      linesLayer.classList.toggle("is-faded", target.faded);
      medianLayer.style.opacity = target.medianVisible ? "1" : "0";
      setTicks(ticksBox, target.scale, (value) => percentTick(value, target.scale.step), animateMove);
      running = animate(animateMove ? 800 : 0, (progress) => {
        draw(mix(from, target, easeOut(progress, 4)));
      }, () => {
        draw(target);
        running = null;
      });
    }

    buttons.forEach((button, index) => button.addEventListener("click", () => {
      if (index !== active) select(index, true);
    }));

    const axis = byId("story-axis");
    if (axis) {
      setAxisLabels(axis, [
        { index: 0, text: monthLabel(story.months[0]) },
        { index: story.base_months, text: monthLabel(story.months[story.base_months]) },
        { index: count - 1, text: monthLabel(story.months[count - 1]) },
      ], count);
    }
    draw(states[0]);
    select(0, false);
  }

  // ---------------------------------------------------------------------------
  // 02 · Горизонты: полосы MAE четырёх моделей
  // ---------------------------------------------------------------------------

  function initHorizons(horizons) {
    const segmented = byId("h-seg");
    const list = byId("h-bars");
    const note = byId("h-note");
    if (!segmented || !list || !note) return;

    const rows = horizons.models.map((model) => {
      const item = node("li", "bar-row", null, list);
      const name = node("div", "bar-name", null, item);
      node("span", "bar-label", model.label, name);
      node("span", "bar-sub", model.note, name);
      const track = node("div", "bar-track", null, item);
      track.setAttribute("aria-hidden", "true");
      const fill = node("div", "bar-fill", null, track);
      const value = node("span", "bar-value", null, item);
      return { item, fill, value };
    });

    const buttons = horizons.labels.map((label, index) => {
      const button = node("button", null, label, segmented);
      button.type = "button";
      button.addEventListener("click", () => select(index));
      return button;
    });

    function select(index) {
      const values = horizons.models.map((model) => model.mae[index]);
      const present = values.filter((v) => v !== null);
      const max = Math.max.apply(null, present);
      const min = Math.min.apply(null, present);
      rows.forEach((row, k) => {
        const value = values[k];
        const best = value !== null && value === min;
        row.item.classList.toggle("is-best", best);
        row.item.classList.toggle("is-empty", value === null);
        row.fill.style.width = value === null ? "0%" : `${((value / max) * 100).toFixed(1)}%`;
        row.value.textContent = value === null ? "не обучить" : `${formatInt(value)}${NBSP}₽`;
        if (best) node("span", "visually-hidden", " — лучшая", row.value);
      });
      note.textContent = horizons.notes[index];
      buttons.forEach((button, k) => button.setAttribute("aria-pressed", String(k === index)));
    }

    select(Math.max(0, horizons.list.indexOf(horizons.main)));
  }

  // ---------------------------------------------------------------------------
  // 03 · Проверка фактом: история, прогноз первого этапа, два правила, факт по месяцам
  // ---------------------------------------------------------------------------

  // Шаг между кружками месяцев проверки, px, не меньше этого: кружок — 11 px, и при меньшем шаге
  // на телефоне они наползали друг на друга. Чтобы шаг был таким, история на узком графике короче
  // (LandingLib.historyKeep); на широком остаётся вся.
  const FACT_MIN_DOT_STEP = 14;
  // Зазор между подписями оси лет, px.
  const FACT_LABEL_GAP = 10;

  function initFact(fact) {
    const plot = byId("fact-plot");
    const slider = byId("fact-k");
    if (!plot || !slider) return;

    const check = fact.check;
    const nCheck = check.months.length;
    const nFull = fact.history.values.length;
    const rules = [
      { key: "naive", line: byId("fact-r1-line"), toggle: byId("fact-r1"), text: byId("fact-r1-text") },
      { key: "seasonal_naive", line: byId("fact-r2-line"), toggle: byId("fact-r2"), text: byId("fact-r2-text") },
    ];
    const dotsBox = byId("fact-dots");
    const cursor = byId("fact-cursor");
    const readout = { month: byId("ro-month"), fc: byId("ro-fc"), act: byId("ro-act"), err: byId("ro-err") };
    const unit = `${NBSP}млрд${NBSP}₽`;

    let view = null; // раскладка под текущую ширину графика
    let shownMonth = nCheck;

    function showMonth(k) {
      shownMonth = k;
      const i = k - 1;
      cursor.style.left = view.left(view.nHistory + i);
      view.dots.forEach((dot, index) => dot.classList.toggle("is-off", index >= k));
      const month = check.months[i];
      const name = `${MONTH_NOM[Number(month.slice(5, 7)) - 1]} ${month.slice(0, 4)}`;
      const fc = check.forecast_rub.two_stage[i] + unit;
      const act = check.actual_rub[i] + unit;
      const err = `${check.error_pct.two_stage[i]}%`;
      readout.month.textContent = name;
      readout.fc.textContent = fc;
      readout.act.textContent = act;
      readout.err.textContent = err;
      slider.setAttribute("aria-valuetext", `${name}: прогноз ${fc}, факт ${act}, ошибка месяца ${err}`);
    }

    // Линии, шкала, подписи и кружки для keep последних месяцев истории. Данных генератора это
    // не меняет: на узком экране показывается их конец.
    function layout(keep) {
      const skip = nFull - keep;
      const history = { months: fact.history.months.slice(skip), values: fact.history.values.slice(skip) };
      const nHistory = keep;
      const total = nHistory + nCheck;
      const months = history.months.concat(check.months);
      const xs = spacedX(total);

      const { min, max } = extent([
        history.values, check.actual, check.forecast.two_stage, check.forecast.naive, check.forecast.seasonal_naive,
      ]);
      const scale = niceScale(min, max, 5);
      const left = (index) => `${((index / (total - 1)) * 100).toFixed(2)}%`;
      const yPoints = (values) => values.map((v) => yOf(v, scale));

      // Прогноз и правила продолжают историю от её последней точки.
      function tail(values) {
        const lastX = xs[nHistory - 1];
        const lastY = yOf(history.values[nHistory - 1], scale);
        return `M${lastX.toFixed(1)},${lastY.toFixed(1)}${linePath(yPoints(values), xs.slice(nHistory)).replace(/^M/, "L")}`;
      }

      setTicks(byId("fact-ticks"), scale, (value) => formatTick(value, scale.step), false);
      byId("fact-hist").setAttribute("d", linePath(yPoints(history.values), xs.slice(0, nHistory)));
      byId("fact-fc").setAttribute("d", tail(check.forecast.two_stage));
      byId("fact-r1-line").setAttribute("d", tail(check.forecast.naive));
      byId("fact-r2-line").setAttribute("d", tail(check.forecast.seasonal_naive));
      byId("fact-end").style.left = left(nHistory - 1);

      const years = byId("fact-years");
      years.textContent = "";
      months.forEach((month, index) => {
        if (month.slice(5, 7) !== "01") return;
        const span = node("span", null, month.slice(0, 4), years);
        span.style.left = left(index);
      });
      // История на узком графике начинается не с января: левый край подписан месяцем, иначе не
      // видно, с чего она начинается. Подпись, которой не хватает места до ближайшей, не ставится.
      if (history.months[0].slice(5, 7) !== "01") {
        const first = node("span", null, monthLabel(history.months[0]), years);
        first.style.left = "0";
        const next = Array.from(years.children).filter((span) => span !== first).map((span) => span.offsetLeft);
        if (next.length && first.offsetWidth + FACT_LABEL_GAP > Math.min.apply(null, next)) years.removeChild(first);
      }

      plot.setAttribute("aria-label",
        `График федерального агрегата: история с ${monthLabel(history.months[0])} по ${monthLabel(history.months[nHistory - 1])}, ` +
        `прогноз первого этапа и факт за ${nCheck} ${pluralRu(nCheck, "месяц", "месяца", "месяцев")} ${check.months[0].slice(0, 4)} года`);

      dotsBox.textContent = "";
      const dots = check.actual.map((value, index) => {
        const dot = node("div", "dot is-off", null, dotsBox);
        dot.style.left = left(nHistory + index);
        dot.style.top = `${percentTop(value, scale).toFixed(2)}%`;
        return dot;
      });

      view = { keep, nHistory, left, dots };
      showMonth(shownMonth);
    }

    const keepFor = () => historyKeep(nFull, nCheck, plot.clientWidth, FACT_MIN_DOT_STEP);

    byId("fact-unit").textContent = `Федеральный агрегат, ${fact.unit}`;
    slider.min = "1";
    slider.max = String(nCheck);
    slider.step = "1";
    slider.value = String(nCheck);
    slider.addEventListener("input", () => showMonth(Number(slider.value)));
    layout(keepFor());
    // Ширина графика меняется при повороте телефона и изменении окна: число точек истории — вместе с ней.
    if ("ResizeObserver" in window) {
      new ResizeObserver(() => {
        const keep = keepFor();
        if (keep !== view.keep) layout(keep);
      }).observe(plot);
    }

    rules.forEach((rule) => {
      const name = `«${fact.rule_names[rule.key]}»`;
      rule.text.textContent = `правило ${name}`;
      rule.toggle.addEventListener("change", () => {
        rule.line.style.opacity = rule.toggle.checked ? "1" : "0";
      });
    });
    byId("mean-own").textContent = `${fact.mape.two_stage}%`;
    byId("mean-r1-name").textContent = `«${fact.rule_names.naive}»`;
    byId("mean-r1").textContent = `${fact.mape.naive}%`;
    byId("mean-r2-name").textContent = `«${fact.rule_names.seasonal_naive}»`;
    byId("mean-r2").textContent = `${fact.mape.seasonal_naive}%`;
  }

  // ---------------------------------------------------------------------------
  // 04 · Стенд: три муниципалитета, мини-график, ссылка на стенд с выбранным
  // ---------------------------------------------------------------------------

  const NO_REGION = "регион не определён";
  const KNOWN_LABEL = "если федеральный индекс за месяц уже опубликован";

  function standUrl(seriesId) {
    return `demo/?mo=${encodeURIComponent(seriesId)}`;
  }

  function initTeaser(teaser) {
    const plot = byId("t-plot");
    const chips = byId("t-chips");
    if (!plot || !chips) return;

    const months = teaser.months;
    const count = months.length;
    const xs = spacedX(count);
    const nFact = teaser.items[0].fact.length;
    const paths = { fact: byId("t-fact"), known: byId("t-known"), fc: byId("t-fc") };
    const breaksBox = byId("t-breaks");
    const panelEnd = byId("t-panel-end");
    const go = byId("t-go");
    let active = -1;
    let shown = null;
    let running = null;

    panelEnd.style.left = `${(((nFact - 0.5) / (count - 1)) * 100).toFixed(2)}%`;

    // Для каждого МО — своя шкала по его значениям, с запасом сверху и снизу.
    function buildState(item) {
      const { min, max } = extent([item.fact, item.forecast, item.known]);
      const pad = (max - min) * 0.1;
      const scale = { lo: min - pad, hi: max + pad };
      const fromLast = (values) => [yOf(item.fact[nFact - 1], scale)].concat(values.map((v) => yOf(v, scale)));
      return {
        fact: item.fact.map((v) => yOf(v, scale)),
        known: fromLast(item.known),
        fc: fromLast(item.forecast),
      };
    }

    const states = teaser.items.map(buildState);

    function draw(state) {
      paths.fact.setAttribute("d", linePath(state.fact, xs.slice(0, nFact)));
      // Прогноз и пунктир начинаются от последней точки факта: их первый узел — она же.
      paths.known.setAttribute("d", linePath(state.known, xs.slice(nFact - 1)));
      paths.fc.setAttribute("d", linePath(state.fc, xs.slice(nFact - 1)));
      shown = state;
    }

    function mix(from, to, k) {
      const blend = (a, b) => a.map((y, i) => y + (b[i] - y) * k);
      return { fact: blend(from.fact, to.fact), known: blend(from.known, to.known), fc: blend(from.fc, to.fc) };
    }

    function legend(item) {
      const box = byId("t-legend");
      box.textContent = "";
      const factYears = `${months[0].slice(0, 4)}–${months[nFact - 1].slice(0, 4)}`;
      const items = [
        ["key key-fact", `факт ${factYears}`],
        ["key key-fc", `прогноз ${months[nFact].slice(0, 4)}`],
        ["key key-r1", KNOWN_LABEL],
      ];
      if (item.breaks.length) items.push(["key key-break", "изломы"]);
      items.forEach((entry) => {
        const li = node("li", null, null, box);
        node("span", entry[0], null, li);
        li.appendChild(document.createTextNode(entry[1]));
      });
    }

    function renderBreaks(item) {
      breaksBox.textContent = "";
      let previous = -10;
      let row = 0;
      item.breaks.forEach((month) => {
        const index = months.indexOf(month);
        if (index < 0) return;
        const line = node("div", "vline vline-break", null, breaksBox);
        line.style.left = `${((index / (count - 1)) * 100).toFixed(2)}%`;
        // Близкие изломы — подписи в два ряда, чтобы не наезжали друг на друга.
        row = index - previous < 5 ? row + 1 : 0;
        previous = index;
        const label = node("span", null, monthLabel(month), line);
        label.style.top = `${(row % 2) * 14 - 2}px`;
      });
    }

    function describe(item) {
      const parts = [
        `${item.id}: расходы на человека в месяц, факт ${monthLabel(months[0])} — ${monthLabel(months[nFact - 1])}`,
        `прогноз ${monthLabel(months[nFact])} — ${monthLabel(months[count - 1])}`,
      ];
      if (item.breaks.length) parts.push(`изломы: ${item.breaks.map(monthLabel).join(", ")}`);
      return parts.join("; ");
    }

    function select(index, animateMove) {
      const item = teaser.items[index];
      const from = shown || states[index];
      if (running) running.cancel();
      active = index;
      Array.from(chips.children).forEach((chip, k) => chip.setAttribute("aria-pressed", String(k === index)));
      byId("t-name").textContent = item.id;
      byId("t-region").textContent = item.region || NO_REGION;
      go.href = standUrl(item.id);
      plot.setAttribute("aria-label", describe(item));
      renderBreaks(item);
      legend(item);
      running = animate(animateMove ? 500 : 0, (progress) => {
        draw(mix(from, states[index], easeOut(progress, 3)));
      }, () => {
        draw(states[index]);
        running = null;
      });
    }

    teaser.items.forEach((item, index) => {
      const chip = node("button", "chip", item.short, chips);
      chip.type = "button";
      chip.addEventListener("click", () => { if (index !== active) select(index, true); });
    });
    select(0, false);
  }

  // ---------------------------------------------------------------------------
  // Поиск «Покажите мой город»: список рядов из demo/data/index.json — при первом фокусе.
  // Ранжирование — LandingLib.searchRows, то же, что на стенде.
  // ---------------------------------------------------------------------------

  function initSearch() {
    const input = byId("mo-q");
    const list = byId("mo-list");
    const go = byId("mo-go");
    const status = byId("mo-status");
    const note = byId("mo-note");
    if (!input || !list || !go || !note) return;

    const LIMIT = 6;
    const STAND = "demo/";
    const defaultNote = note.textContent;
    let rows = null;
    let loading = null;
    let failed = false;
    let failureShown = false;
    let selected = null;
    let options = [];
    let active = -1;

    function load() {
      if (rows) return Promise.resolve(rows);
      if (!loading) {
        loading = fetch("demo/data/index.json")
          .then((response) => {
            if (!response.ok) throw new Error(`demo/data/index.json: ${response.status}`);
            return response.json();
          })
          .then((index) => {
            rows = rowsFromIndex(index.series);
            failed = false;
            // Подсказка о сбое устарела: список загрузился при повторной попытке.
            if (failureShown) resetNote();
            return rows;
          })
          .catch((error) => {
            // Не запоминаем отказ: следующий фокус попробует загрузить заново.
            loading = null;
            failed = true;
            showFailure();
            throw error;
          });
      }
      return loading;
    }

    // Куда ведёт кнопка «Открыть прогноз»: выбранный в списке ряд, иначе подсвеченная подсказка,
    // иначе первое совпадение набранного запроса; пустой запрос и запрос без совпадений — стенд
    // без выбора, поиск там покажет сам стенд.
    function targetUrl() {
      const row = selected || pickTarget(rows, input.value, active >= 0 ? options[active] : null);
      return row ? standUrl(row.id) : STAND;
    }

    // Адрес кнопки держится в ссылке заранее: так он верен и при открытии в новой вкладке.
    function syncGo() {
      go.href = targetUrl();
    }

    // Переход по кнопке или клавише Enter. Запрос уже набран, а список ещё грузится (быстрый
    // набор, поле заполнено браузером после возврата на страницу): ждём список и идём по нему.
    function openStand() {
      if (rows || selected || !input.value.trim()) {
        window.location.href = targetUrl();
        return;
      }
      load().then(() => { window.location.href = targetUrl(); }, () => { window.location.href = STAND; });
    }

    function showFailure() {
      closeList();
      failureShown = true;
      note.textContent = "";
      note.appendChild(document.createTextNode("Список муниципалитетов не загрузился. Найти свой город можно "));
      const link = node("a", null, "на стенде", note);
      link.href = STAND;
      note.appendChild(document.createTextNode("."));
    }

    function resetNote() {
      failureShown = false;
      note.textContent = defaultNote;
    }

    function openList() {
      list.hidden = false;
      input.setAttribute("aria-expanded", "true");
      // Поле стоит низко, и список выходит за нижний край окна: докручиваем ровно настолько, чтобы он влез.
      list.scrollIntoView({ block: "nearest" });
    }

    function closeList() {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
      active = -1;
      syncGo();
    }

    function setActive(index) {
      const items = list.querySelectorAll('[role="option"]:not([aria-disabled])');
      items.forEach((item) => item.setAttribute("aria-selected", "false"));
      active = index;
      if (index >= 0 && index < items.length) {
        items[index].setAttribute("aria-selected", "true");
        input.setAttribute("aria-activedescendant", items[index].id);
        items[index].scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
      syncGo();
    }

    function statusRow(text) {
      list.textContent = "";
      const item = node("li", "suggestion-status", text, list);
      item.setAttribute("role", "option");
      item.setAttribute("aria-disabled", "true");
      openList();
    }

    function choose(row) {
      selected = row;
      failureShown = false;
      input.value = row.id;
      closeList();
      note.textContent = "";
      note.appendChild(document.createTextNode("Выбран: "));
      node("span", "picked", row.id, note);
      note.appendChild(document.createTextNode(` · ${row.region || NO_REGION} · `));
      const link = node("a", null, "открыть на стенде →", note);
      link.href = standUrl(row.id);
    }

    function renderList() {
      const query = input.value;
      if (!query.trim()) {
        list.textContent = "";
        options = [];
        status.textContent = "";
        closeList();
        return;
      }
      if (!rows) {
        statusRow("Загружаю список…");
        return;
      }
      const { matches, total } = searchRows(rows, query, LIMIT);
      options = matches;
      status.textContent = total
        ? `Найдено ${formatInt(total)} ${pluralRu(total, "совпадение", "совпадения", "совпадений")}`
        : "Ничего не нашлось";
      if (!matches.length) {
        statusRow("Ничего не нашлось — попробуйте часть названия или регион.");
        return;
      }
      list.textContent = "";
      matches.forEach((row, index) => {
        const item = node("li", "suggestion", null, list);
        item.id = `mo-option-${index}`;
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", "false");
        node("span", "suggestion-name", row.id, item);
        node("span", "suggestion-region", row.region || NO_REGION, item);
        item.addEventListener("click", () => choose(row));
      });
      if (total > matches.length) {
        const more = total - matches.length;
        const item = node("li", "suggestion-status",
          `И ещё ${formatInt(more)} ${pluralRu(more, "совпадение", "совпадения", "совпадений")} — уточните запрос`, list);
        item.setAttribute("role", "option");
        item.setAttribute("aria-disabled", "true");
      }
      active = -1;
      input.removeAttribute("aria-activedescendant");
      openList();
    }

    function render() {
      renderList();
      syncGo();
    }

    // Список загрузился: раскрывается он, только если поле всё ещё в фокусе — иначе медленная сеть
    // открыла бы подсказки у человека, который уже ушёл с поля.
    function afterLoad() {
      if (document.activeElement === input) render();
      else syncGo();
    }

    // mousedown до click: иначе фокус успевает уйти с поля раньше, чем сработает выбор.
    list.addEventListener("mousedown", (event) => event.preventDefault());

    input.addEventListener("focus", () => {
      // Первый фокус загружает список; после отказа каждый следующий пробует снова.
      load().then(afterLoad, () => {});
    });

    // Уход с поля (Tab) закрывает список; выбор мышью поле не покидает — см. mousedown выше.
    input.addEventListener("blur", closeList);

    input.addEventListener("input", () => {
      if (selected) {
        selected = null;
        resetNote();
      }
      if (!rows && failed) {
        showFailure();
        return;
      }
      render();
      if (!rows) load().then(afterLoad, () => {});
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        if (list.hidden) render();
        else if (options.length) setActive(Math.min(options.length - 1, active + 1));
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        if (!list.hidden && options.length) setActive(Math.max(0, active - 1));
      } else if (event.key === "Enter") {
        if (!list.hidden && options.length) {
          event.preventDefault();
          choose(options[active >= 0 ? active : 0]);
        } else if (input.value.trim()) {
          // Список закрыт или подсказок нет: то же, что кнопка «Открыть прогноз».
          event.preventDefault();
          openStand();
        }
      }
    });

    go.addEventListener("click", (event) => {
      if (rows || selected || !input.value.trim()) {
        syncGo();
        return;
      }
      event.preventDefault();
      openStand();
    });

    // Escape закрывает список всегда, где бы ни стоял фокус.
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !list.hidden) closeList();
    });

    document.addEventListener("click", (event) => {
      if (!event.target.closest(".search")) closeList();
    });
  }

  // ---------------------------------------------------------------------------
  // Запуск: сбой одного блока не должен останавливать остальные
  // ---------------------------------------------------------------------------

  [
    ["шапка", initNav],
    ["обложка", () => initHero(data.story)],
    ["числа", initCounters],
    ["история", () => initStory(data.story)],
    ["горизонты", () => initHorizons(data.horizons)],
    ["факт", () => initFact(data.fact)],
    ["стенд", () => initTeaser(data.teaser)],
    ["поиск", initSearch],
  ].forEach((block) => {
    try {
      block[1]();
    } catch (error) {
      console.error(`Не удалось запустить блок «${block[0]}»`, error);
    }
  });
}());
