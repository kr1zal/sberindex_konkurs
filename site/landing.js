/* site/landing.js — главная: графики, счёт чисел и переключатели поверх данных из
 * <script id="landing-data">. Данные и все числа приходят от генератора
 * (scripts/build_site.py, формат — его докстринг); здесь только отрисовка, форматирование
 * и подписи. Ни одно число данных в этом файле не пишется: ни ряды, ни доли, ни проценты.
 *
 * Библиотек нет: SVG собирается вручную. Положение линий — в единицах viewBox 0…1000,
 * подписи и точки — в процентах блока графика (раскладка — site/landing.css).
 * При prefers-reduced-motion всё показывается сразу, без анимаций.
 *
 * Всё, что не касается страницы (шкалы, подписи оси, разбор чисел), живёт в site/landing-lib.js
 * и подключается перед этим файлом: там его проверяют тесты через node.
 */
(function () {
  "use strict";

  const {
    formatInt, pluralRu, niceScale, formatTick, createTickSets, historyKeep,
    numberTokens, countedText, NO_REGION,
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
  // Обложка: 30 линий прорисовываются со сдвигом, затем общее движение. График — иллюстрация:
  // без осей и подписей, скрыт от программ чтения с экрана
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
  }

  // ---------------------------------------------------------------------------
  // Четыре числа полосы на обложке: при появлении в окне — счёт от нуля, в конце — исходный текст
  // ---------------------------------------------------------------------------

  // Счёт короткий: на полпути число — не то, что в итоге, и беглый взгляд не должен успеть его прочесть.
  const COUNT_MS = 600;

  function countUp(element) {
    const original = element.textContent;
    const tokens = numberTokens(original);
    if (!tokens.length) return;
    animate(COUNT_MS, (progress) => {
      element.textContent = countedText(original, tokens, easeOut(progress, 3));
    }, () => {
      element.textContent = original;
    });
  }

  // Считаются только числа полосы: слова рядом с числом на полпути не пересчитываются
  // (так вышло бы «946 рядов × 11 месяца»), и такие строки в полосе не стоят.
  function initCounters() {
    const numbers = document.querySelectorAll(".stat .stat-number");
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
  // 01 · Почему это прогноз одного числа: три шага над одними и теми же рядами и четвёртый —
  // разложение ошибки полосами. Шаги проходят сами один раз, когда раздел появляется на экране;
  // клик по шагу останавливает показ и открывает выбранный.
  // ---------------------------------------------------------------------------

  const STORY_PLAY_FIRST_MS = 900; // пауза до второго шага после появления раздела
  const STORY_PLAY_STEP_MS = 2300; // пауза между шагами показа
  const STORY_BARS_STEP = 3; // номер шага с полосами (с нуля)

  function initStory(story) {
    const plot = byId("story-plot");
    const card = plot ? plot.closest(".story-card") : null;
    const linesLayer = byId("story-lines");
    const medianLayer = byId("story-median-layer");
    const medianPath = byId("story-median");
    const band = byId("story-band");
    const bandLabel = byId("story-band-label");
    const caption = byId("story-caption");
    const text = byId("story-text");
    const barsBox = byId("story-bars");
    const barsLegend = byId("story-bars-legend");
    const unitLines = byId("story-unit-lines");
    const unitBars = byId("story-unit-bars");
    const buttons = Array.from(document.querySelectorAll(".step"));
    if (!plot || !card || !linesLayer || !medianPath || !band || !barsBox || !buttons.length) return;

    const count = story.months.length;
    const xs = spacedX(count);
    const paths = story.series.map(() => svgNode("path", {}, linesLayer));
    const unitLinesDefault = unitLines.textContent;

    // Шаг 1 — ряды, шаг 2 — те же ряды бледнеют и видна медиана, шаг 3 — каждый ряд,
    // делённый на медиану, ровная линия на единице и полоса ±M% вокруг неё: в ней лежит
    // не меньше заявленной доли значений. Отметок оси нет — это картина, а не измерение;
    // шкала у каждого шага своя, по его значениям.
    const spread = story.spread_pct / 100;
    function buildState(step) {
      const rows = story.series.map((row) => (step === 2 ? row.map((v, i) => v / story.median[i]) : row));
      const flat = story.median.map((v) => (step === 2 ? 1 : v));
      const { min, max } = extent(rows.concat([flat]));
      const scale = niceScale(min, max, 5);
      return {
        ys: rows.map((row) => row.map((v) => yOf(v, scale))),
        median: flat.map((v) => yOf(v, scale)),
        bandTop: yOf(1 + spread, scale),
        bandBottom: yOf(1 - spread, scale),
        faded: step === 1,
        medianVisible: step !== 0,
        bandVisible: step === 2,
      };
    }
    bandLabel.textContent = `±${story.spread_pct}%`;

    const states = [0, 1, 2].map(buildState);
    let shown = states[0];
    let active = 0;
    let running = null;

    function draw(state) {
      state.ys.forEach((ys, index) => paths[index].setAttribute("d", linePath(ys, xs)));
      medianPath.setAttribute("d", linePath(state.median, xs));
      band.setAttribute("y", state.bandTop.toFixed(1));
      band.setAttribute("height", Math.max(0, state.bandBottom - state.bandTop).toFixed(1));
      bandLabel.style.top = `${(state.bandTop / VIEW * 100).toFixed(2)}%`;
      shown = state;
    }

    // Координаты точек между двумя состояниями: линии перетекают, а не перескакивают.
    function mix(from, to, k) {
      return {
        ys: from.ys.map((row, j) => row.map((y, i) => y + (to.ys[j][i] - y) * k)),
        median: from.median.map((y, i) => y + (to.median[i] - y) * k),
        bandTop: from.bandTop + (to.bandTop - from.bandTop) * k,
        bandBottom: from.bandBottom + (to.bandBottom - from.bandBottom) * k,
      };
    }

    // Шаг 4: у каждой модели ошибка — разброс между муниципалитетами плюс промах мимо общего
    // движения; ширина полосы — доля от самой большой ошибки.
    const decomposition = story.decomposition;
    const maxTotal = Math.max.apply(null, decomposition.models.map((m) => m.bias + m.spread));
    decomposition.models.forEach((model) => {
      const row = node("div", "story-bar-row", null, barsBox);
      node("span", "story-bar-label", model.label, row);
      const track = node("span", "story-bar-track", null, row);
      node("span", "story-bar-spread", null, track).style.width = `${((model.spread / maxTotal) * 100).toFixed(2)}%`;
      node("span", "story-bar-bias", null, track).style.width = `${((model.bias / maxTotal) * 100).toFixed(2)}%`;
      node("span", "story-bar-value", `${formatInt(model.bias + model.spread)}${NBSP}₽`, row);
    });
    unitBars.textContent =
      `Средняя ошибка, ₽ на человека в месяц, горизонт ${decomposition.horizon}${NBSP}мес.: из чего она состоит`;

    function stepText(step) {
      return buttons[step].querySelector(".step-text").textContent.replace(/\s+/g, " ").trim();
    }

    // Видимой подписи под графиком нет: название шага и его строка — подпись графика для программ чтения
    // с экрана и текст скрытой живой области, которая объявляет смену шага.
    function stepLabel(step) {
      const title = buttons[step].querySelector(".step-title").textContent.replace(/\s+/g, " ").trim();
      return `${title}: ${stepText(step)}`;
    }

    function select(step, animateMove) {
      const bars = step === STORY_BARS_STEP;
      if (running) running.cancel();
      active = step;
      buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === step)));
      const label = stepLabel(step);
      caption.textContent = label;
      plot.setAttribute("aria-label", label);
      text.textContent = stepText(step);
      card.classList.toggle("is-bars", bars);
      barsBox.setAttribute("aria-hidden", String(!bars));
      barsLegend.hidden = !bars;
      unitLines.hidden = bars;
      unitBars.hidden = !bars;
      if (bars) return; // линии остаются как были и гаснут под полосами (стили)
      unitLines.textContent = step === 2 ? "Ряды, делённые на общее движение, %" : unitLinesDefault;
      const target = states[step];
      const from = shown;
      linesLayer.classList.toggle("is-faded", target.faded);
      medianLayer.style.opacity = target.medianVisible ? "1" : "0";
      card.classList.toggle("is-band", target.bandVisible);
      running = animate(animateMove ? 800 : 0, (progress) => {
        draw(mix(from, target, easeOut(progress, 4)));
      }, () => {
        draw(target);
        running = null;
      });
    }

    // Показ шагов один раз, когда раздел виден; любой клик по шагу его останавливает.
    let timers = [];
    let played = false;
    function stopPlay() {
      timers.forEach(clearTimeout);
      timers = [];
    }
    function play() {
      if (played) return;
      played = true;
      if (reducedMotion()) {
        select(STORY_BARS_STEP, false);
        return;
      }
      [1, 2, STORY_BARS_STEP].forEach((step, k) => {
        timers.push(setTimeout(() => select(step, true), STORY_PLAY_FIRST_MS + k * STORY_PLAY_STEP_MS));
      });
    }

    buttons.forEach((button, index) => button.addEventListener("click", () => {
      stopPlay();
      played = true;
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

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          observer.disconnect();
          play();
        });
      }, { threshold: 0.6 });
      observer.observe(plot);
    }
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
  // 03 · Проверка фактом: история, прогноз первого этапа, два правила, факт по месяцам.
  // Факт проверки — та же линия, что и история, с кружками в месяцах; ползунок под месяцами
  // проверки открывает её по месяцу, бегунок стоит под кружком выбранного месяца.
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
    const reveal = byId("fact-act-reveal"); // прямоугольник, которым обрезана линия факта проверки
    const scrubTrack = byId("fact-scrub");
    const scrubTicks = byId("fact-scrub-ticks");
    const scrubValue = byId("fact-k-value");
    const readout = { month: byId("ro-month"), fc: byId("ro-fc"), act: byId("ro-act"), err: byId("ro-err") };
    const unit = `${NBSP}млрд${NBSP}₽`;

    let view = null; // раскладка под текущую ширину графика
    let shownMonth = nCheck;

    function showMonth(k) {
      shownMonth = k;
      const i = k - 1;
      cursor.style.left = view.left(view.nHistory + i);
      // Линия факта проверки нарисована целиком и открыта до выбранного месяца: обрезка по x
      // в единицах viewBox, как у точек линии.
      reveal.style.width = `${view.xs[view.nHistory + i].toFixed(1)}px`;
      view.dots.forEach((dot, index) => dot.classList.toggle("is-off", index >= k));
      view.ticks.forEach((tick, index) => tick.classList.toggle("is-on", index < k));
      const month = check.months[i];
      const name = `${MONTH_NOM[Number(month.slice(5, 7)) - 1]} ${month.slice(0, 4)}`;
      const fc = check.forecast_rub.two_stage[i] + unit;
      const act = check.actual_rub[i] + unit;
      const err = `${check.error_pct.two_stage[i]}%`;
      readout.month.textContent = name;
      readout.fc.textContent = fc;
      readout.act.textContent = act;
      readout.err.textContent = err;
      scrubValue.textContent = MONTH_NOM[Number(month.slice(5, 7)) - 1];
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
      byId("fact-act").setAttribute("d", tail(check.actual));
      byId("fact-fc").setAttribute("d", tail(check.forecast.two_stage));
      byId("fact-r1-line").setAttribute("d", tail(check.forecast.naive));
      byId("fact-r2-line").setAttribute("d", tail(check.forecast.seasonal_naive));
      byId("fact-end").style.left = left(nHistory - 1);

      // Ползунок стоит под месяцами проверки: его бегунок — на одной вертикали с кружком месяца и
      // линией курсора. Ширина дорожки — от первого до последнего месяца плюс бегунок (его центр
      // не доходит до краёв поля на половину своей ширины — поле на столько же шире).
      scrubTrack.style.left = `calc(${left(nHistory)} - var(--scrub-thumb) / 2)`;
      scrubTrack.style.width = `calc(${((nCheck - 1) / (total - 1) * 100).toFixed(2)}% + var(--scrub-thumb))`;
      scrubTicks.textContent = "";
      const ticks = check.months.map((_, index) => {
        const tick = node("span", "scrub-tick", null, scrubTicks);
        tick.style.left = `${((index / (nCheck - 1)) * 100).toFixed(2)}%`;
        return tick;
      });

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

      view = { keep, nHistory, left, xs, dots, ticks };
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
  // 04 · Изломы: доля муниципалитетов с изломом по месяцам — найдено задним числом, по полному ряду
  // ---------------------------------------------------------------------------

  function initBreaks(breaks) {
    const plot = byId("breaks-plot");
    const cols = byId("breaks-cols");
    const ticks = byId("breaks-ticks");
    const axis = byId("breaks-axis");
    if (!breaks || !plot || !cols || !ticks || !axis) return;

    const count = breaks.months.length;
    const scale = niceScale(0, Math.max.apply(null, breaks.share), 4);
    setTicks(ticks, scale, (value) => `${formatTick(value, scale.step)}%`, false);

    // Столбец месяца — доля от верха шкалы; месяцы массового согласия подсвечены и подписаны долей.
    // Подпись стоит над серединой столбца, поэтому положение в процентах считается по центру ячейки.
    breaks.share.forEach((value, index) => {
      const height = `${((value / scale.hi) * 100).toFixed(2)}%`;
      const col = node("div", "breaks-col", null, cols);
      col.style.height = height;
      const rank = breaks.top.indexOf(breaks.months[index]);
      if (rank < 0) return;
      col.classList.add("is-top");
      const label = node("span", "breaks-value", breaks.top_labels[rank], cols);
      label.style.left = `${(((index + 0.5) / count) * 100).toFixed(2)}%`;
      label.style.bottom = `calc(${height} + 4px)`;
    });

    // Подписи месяцев: у левого края, у каждого января и у правого.
    breaks.months.forEach((month, index) => {
      const first = index === 0;
      const last = index === count - 1;
      if (!(first || last || month.slice(5, 7) === "01")) return;
      const span = node("span", null, monthLabel(month), axis);
      if (first) {
        span.style.left = "0";
      } else if (last) {
        span.style.right = "0";
      } else {
        span.style.left = `${(((index + 0.5) / count) * 100).toFixed(2)}%`;
        span.style.transform = "translateX(-50%)";
      }
    });

    const monthName = (ym) => `${MONTH_NOM[Number(ym.slice(5, 7)) - 1]} ${ym.slice(0, 4)}`;
    const ranked = breaks.top
      .map((month, rank) => ({ month, label: breaks.top_labels[rank], share: breaks.share[breaks.months.indexOf(month)] }))
      .sort((a, b) => b.share - a.share);
    plot.setAttribute("aria-label",
      `Доля муниципалитетов, у которых найден излом, по месяцам с ${monthLabel(breaks.months[0])} ` +
      `по ${monthLabel(breaks.months[count - 1])}; больше всего — ` +
      ranked.map((item) => `${monthName(item.month)}, ${item.label}`).join("; "));
  }

  // ---------------------------------------------------------------------------
  // 05 · Прогноз по муниципалитету: один из рядов крупно — факт, прогноз рекомендуемой модели
  // и эталона конкурса, точки структурных изменений, числа раздела; три примера переключаются
  // кнопками, ссылка «Открыть этот муниципалитет» ведёт на страницу прогноза с выбранным рядом.
  // ---------------------------------------------------------------------------

  function standUrl(seriesId) {
    return `demo/?mo=${encodeURIComponent(seriesId)}`;
  }

  // Правая часть графика (доля ширины) отдана подписи на конце линии прогноза: ряд там не рисуется.
  const TEASER_LABEL_SHARE = 0.13;

  function initTeaser(teaser) {
    const plot = byId("t-plot");
    const chips = byId("t-chips");
    if (!plot || !chips) return;

    const months = teaser.months;
    const count = months.length;
    const span = VIEW * (1 - TEASER_LABEL_SHARE);
    const xs = Array.from({ length: count }, (_, i) => (i / (count - 1)) * span);
    const left = (index) => `${((index / (count - 1)) * (1 - TEASER_LABEL_SHARE) * 100).toFixed(2)}%`;
    const nFact = teaser.items[0].fact.length;
    const paths = { fact: byId("t-fact"), base: byId("t-base"), fc: byId("t-fc") };
    const ends = { fact: byId("t-end-fact"), base: byId("t-end-base"), fc: byId("t-end-fc") };
    const breaksBox = byId("t-breaks");
    const go = byId("t-go");
    const rubles = (value) => `${formatInt(value)}${NBSP}₽`;
    // Числа раздела приходят из данных уже строками в формате отчёта — к ним только знак рубля.
    const withRub = (text) => `${text}${NBSP}₽`;
    const factYears = `${months[0].slice(0, 4)}–${months[nFact - 1].slice(0, 4)}`;
    const lastFactYear = months[nFact - 1].slice(0, 4);
    const forecastYear = months[nFact].slice(0, 4);
    const reference = teaser.reference;
    let active = -1;
    let shown = null;
    let running = null;

    byId("t-panel-end").style.left = left(nFact - 0.5);
    const years = byId("t-years");
    months.forEach((month, index) => {
      if (month.slice(5, 7) !== "01") return;
      node("span", null, month.slice(0, 4), years).style.left = left(index);
    });
    const referenceName = `${reference.name[0].toLowerCase()}${reference.name.slice(1)}`;
    byId("t-legend-fact").textContent = `факт ${factYears}`;
    byId("t-legend-fc").textContent = `прогноз ${forecastYear}`;
    byId("t-legend-base").textContent = `${referenceName} (${reference.note})`;
    byId("t-stat-fc-label").textContent = `Прогноз на ${forecastYear}, в среднем за месяц`;
    byId("t-stat-check-label").textContent = `Точнее всех на истории ${lastFactYear}`;

    // Для каждого МО — своя «красивая» шкала по его значениям, с отметками оси, как у графика
    // раздела о проверке фактом; при смене примера отметки меняются вместе с линиями.
    function buildState(item) {
      const { min, max } = extent([item.fact, item.forecast, item.baseline]);
      const scale = niceScale(min, max, 4);
      const fromLast = (values) => [yOf(item.fact[nFact - 1], scale)].concat(values.map((v) => yOf(v, scale)));
      return {
        scale,
        fact: item.fact.map((v) => yOf(v, scale)),
        base: fromLast(item.baseline),
        fc: fromLast(item.forecast),
      };
    }

    const states = teaser.items.map(buildState);

    function draw(state) {
      paths.fact.setAttribute("d", linePath(state.fact, xs.slice(0, nFact)));
      // Прогноз и эталон начинаются от последней точки факта: их первый узел — она же.
      paths.base.setAttribute("d", linePath(state.base, xs.slice(nFact - 1)));
      paths.fc.setAttribute("d", linePath(state.fc, xs.slice(nFact - 1)));
      // Концы линий — точки с подписью: положение в процентах от высоты графика, y — в единицах viewBox.
      ends.fact.style.top = `${(state.fact[nFact - 1] / VIEW * 100).toFixed(2)}%`;
      ends.base.style.top = `${(state.base[state.base.length - 1] / VIEW * 100).toFixed(2)}%`;
      ends.fc.style.top = `${(state.fc[state.fc.length - 1] / VIEW * 100).toFixed(2)}%`;
      shown = state;
    }

    function mix(from, to, k) {
      const blend = (a, b) => a.map((y, i) => y + (b[i] - y) * k);
      return { fact: blend(from.fact, to.fact), base: blend(from.base, to.base), fc: blend(from.fc, to.fc) };
    }

    function renderBreaks(item) {
      breaksBox.textContent = "";
      let previous = -10;
      let row = 0;
      item.breaks.forEach((month) => {
        const index = months.indexOf(month);
        if (index < 0) return;
        const line = node("div", "vline vline-break", null, breaksBox);
        line.style.left = left(index);
        // Близкие изломы — подписи в два ряда, чтобы не наезжали друг на друга.
        row = index - previous < 5 ? row + 1 : 0;
        previous = index;
        const label = node("span", null, monthLabel(month), line);
        label.style.top = `${(row % 2) * 14 - 2}px`;
      });
    }

    function renderStats(item) {
      const summary = item.summary;
      byId("t-stat-fc").textContent = withRub(summary.forecast_mean);
      byId("t-stat-fc-note").textContent =
        `${summary.growth} к ${lastFactYear} году · ${referenceName}: ${withRub(summary.baseline_mean)}`;
      byId("t-stat-breaks").textContent = item.breaks.length ? item.breaks.map(monthLabel).join(" · ") : "не найдены";
      const best = summary.best;
      const ownBest = best.id !== reference.id;
      byId("t-stat-best").textContent = ownBest ? best.name : reference.name;
      byId("t-stat-best-note").textContent = ownBest
        ? `ошибка ${withRub(best.mae)} против ${withRub(summary.reference_mae)} у ${referenceName.replace(/^эталон/, "эталона")}`
        : `ошибка ${withRub(best.mae)}: остальные модели здесь ошибались сильнее`;
    }

    function describe(item) {
      const parts = [
        `${item.id}: расходы на человека в месяц, факт ${monthLabel(months[0])} — ${monthLabel(months[nFact - 1])}`,
        `прогноз ${monthLabel(months[nFact])} — ${monthLabel(months[count - 1])} рекомендуемой модели и эталона конкурса`,
      ];
      if (item.breaks.length) parts.push(`точки структурных изменений: ${item.breaks.map(monthLabel).join(", ")}`);
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
      byId("t-end-fact-value").textContent = rubles(item.fact[nFact - 1]);
      byId("t-end-fact-month").textContent = monthLabel(months[nFact - 1]);
      byId("t-end-fc-value").textContent = rubles(item.forecast[item.forecast.length - 1]);
      byId("t-end-fc-month").textContent = monthLabel(months[count - 1]);
      setTicks(byId("t-ticks"), states[index].scale, (value) => formatTick(value, states[index].scale.step), animateMove);
      renderBreaks(item);
      renderStats(item);
      running = animate(animateMove ? 500 : 0, (progress) => {
        draw(mix(from, states[index], easeOut(progress, 3)));
      }, () => {
        draw(states[index]);
        running = null;
      });
    }

    ends.fact.style.left = left(nFact - 1);
    ends.base.style.left = left(count - 1);
    ends.fc.style.left = left(count - 1);

    teaser.items.forEach((item, index) => {
      const chip = node("button", "chip", null, chips);
      chip.type = "button";
      node("span", "chip-name", item.short, chip);
      node("span", "visually-hidden", ", ", chip);
      node("span", "chip-kind", item.kind, chip);
      chip.addEventListener("click", () => { if (index !== active) select(index, true); });
    });
    select(0, false);
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
    ["изломы", () => initBreaks(data.breaks)],
    ["прогноз по муниципалитету", () => initTeaser(data.teaser)],
  ].forEach((block) => {
    try {
      block[1]();
    } catch (error) {
      console.error(`Не удалось запустить блок «${block[0]}»`, error);
    }
  });
}());
