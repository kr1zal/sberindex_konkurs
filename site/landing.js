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
  // 02 · Горизонты: четыре модели на всех горизонтах одним графиком, между эталоном и лучшей из
  // наших — лента выигрыша с процентом на каждом горизонте; выбранный горизонт — панель справа.
  // Горизонты листаются сами, когда раздел виден; клик по переключателю, столбцу графика или
  // стрелке останавливает показ.
  // ---------------------------------------------------------------------------

  const HZ_PLAY_FIRST_MS = 900; // пауза до первого шага показа после появления раздела
  const HZ_PLAY_STEP_MS = 3400; // пауза между горизонтами при показе
  const HZ_MAX_STEPS = 8; // сколько горизонтов показ листает сам, потом останавливается на основном
  const HZ_LABEL_GAP = 11; // подписи справа от линий не ближе этого друг к другу, % высоты графика
  const HZ_GRID_STEP = 1000; // шаг линий сетки, ₽
  const HZ_NOT_OURS = ["reference", "naive"]; // роли не наших моделей: выигрыш считается против эталона

  function initHorizons(horizons) {
    const plot = byId("hz-plot");
    const axis = byId("hz-axis");
    const cols = byId("hz-cols");
    const rowsBox = byId("hz-rows");
    const note = byId("h-note");
    const hint = byId("hz-hint");
    if (!plot || !axis || !cols || !rowsBox || !note || !hint) return;
    const models = horizons.models;
    const count = horizons.list.length;
    const mainIndex = Math.max(0, horizons.list.indexOf(horizons.main));
    const rolesOf = (model) => model.role.split("+");
    const primaryRole = (model) => rolesOf(model)[0];
    const reference = models.find((model) => rolesOf(model).includes("reference"));
    const ours = models.filter((model) => !rolesOf(model).some((role) => HZ_NOT_OURS.includes(role)));
    if (!reference || !ours.length) return;
    const present = (model) => model.mae.filter((value) => value !== null);
    const yMax = Math.ceil(Math.max(...models.flatMap(present)) / HZ_GRID_STEP) * HZ_GRID_STEP;
    const xOf = (index) => (index + 0.5) / count * VIEW;
    const yOf = (value) => (1 - value / yMax) * VIEW;
    const pctX = (index) => `${(xOf(index) / VIEW * 100).toFixed(2)}%`;
    const pctYValue = (value) => (1 - value / yMax) * 100;
    const pctY = (value) => `${pctYValue(value).toFixed(2)}%`;
    const lower = (text) => text.charAt(0).toLowerCase() + text.slice(1);
    const lastIndex = (model) => {
      let last = -1;
      model.mae.forEach((value, index) => { if (value !== null) last = index; });
      return last;
    };

    // Лучшая из наших моделей на горизонте и её выигрыш у эталона: на сколько процентов меньше ошибка.
    function bestOurs(index) {
      let best = null;
      ours.forEach((model) => {
        const value = model.mae[index];
        if (value !== null && (best === null || value < best.value)) best = { model, value };
      });
      return best;
    }
    function gainPct(index) {
      const best = bestOurs(index);
      return best ? Math.round((1 - best.value / reference.mae[index]) * 100) : null;
    }

    byId("hz-unit").textContent = `Средняя ошибка, ${horizons.unit}`;

    const ticks = byId("hz-ticks");
    for (let value = 0; value <= yMax; value += HZ_GRID_STEP) {
      const tick = node("div", "tick", null, ticks);
      tick.style.top = pctY(value);
      node("span", null, value ? formatInt(value) : "0", tick);
    }

    // Лента между эталоном и лучшей из наших, потом линии: наши поверх остальных, лучшая — сверху.
    let band = "";
    for (let index = 0; index < count; index++) {
      band += `${index ? "L" : "M"}${xOf(index).toFixed(1)},${yOf(reference.mae[index]).toFixed(1)}`;
    }
    for (let index = count - 1; index >= 0; index--) {
      const best = bestOurs(index);
      band += `L${xOf(index).toFixed(1)},${yOf(best ? best.value : reference.mae[index]).toFixed(1)}`;
    }
    byId("hz-gain-band").setAttribute("d", `${band}Z`);
    const lines = byId("hz-lines");
    const drawOrder = models.slice().sort((a, b) => Number(ours.includes(a)) - Number(ours.includes(b))
      || Number(primaryRole(a) === "best_mean") - Number(primaryRole(b) === "best_mean"));
    drawOrder.forEach((model) => {
      let d = "";
      let first = true;
      model.mae.forEach((value, index) => {
        if (value === null) return;
        d += `${first ? "M" : "L"}${xOf(index).toFixed(1)},${yOf(value).toFixed(1)}`;
        first = false;
      });
      svgNode("path", { class: `hz-line hz-line-${primaryRole(model)}`, d }, lines);
      const last = lastIndex(model);
      if (last >= 0 && last < count - 1) {
        const y = yOf(model.mae[last]).toFixed(1);
        svgNode("path", { class: "hz-line-gap", d: `M${xOf(last).toFixed(1)},${y} L${VIEW},${y}` }, lines);
      }
    });

    const marks = byId("hz-marks");
    drawOrder.forEach((model) => model.mae.forEach((value, index) => {
      if (value === null) return;
      const dot = node("div", `hz-dot hz-dot-${primaryRole(model)}`, null, marks);
      dot.style.left = pctX(index);
      dot.style.top = pctY(value);
    }));
    const pills = horizons.list.map((_, index) => {
      const best = bestOurs(index);
      const pill = node("div", "hz-pill", best ? `−${gainPct(index)}${NBSP}%` : "", marks);
      pill.style.left = pctX(index);
      pill.style.top = pctY((reference.mae[index] + (best ? best.value : reference.mae[index])) / 2);
      pill.hidden = !best;
      return pill;
    });

    // Подписи линий справа — у последней точки каждой; тесные раздвигаются сверху вниз.
    const labelsBox = byId("hz-labels");
    const labelItems = models.map((model) => {
      const last = lastIndex(model);
      return {
        model,
        top: pctYValue(model.mae[last]),
        sub: last < count - 1 ? `на ${lower(horizons.labels[count - 1])} не обучить` : "",
      };
    }).sort((a, b) => a.top - b.top);
    labelItems.forEach((item, k) => {
      if (k) item.top = Math.max(item.top, labelItems[k - 1].top + HZ_LABEL_GAP);
    });
    labelItems.forEach((item) => {
      const label = node("div", `hz-label hz-label-${primaryRole(item.model)}`, null, labelsBox);
      label.style.top = `${item.top.toFixed(2)}%`;
      node("span", "hz-label-name", item.model.label, label);
      if (item.sub) node("span", "hz-label-sub", item.sub, label);
    });

    // Столбцы — кнопки для мыши во всю высоту графика; переключатели под осью — для всех.
    const colButtons = horizons.list.map((_, index) => {
      const button = node("button", "hz-col", null, cols);
      button.type = "button";
      button.tabIndex = -1;
      button.style.left = `${(index / count * 100).toFixed(2)}%`;
      button.style.width = `${(100 / count).toFixed(2)}%`;
      button.addEventListener("click", () => pick(index));
      return button;
    });
    axis.style.setProperty("--hz-step", `${HZ_PLAY_STEP_MS}ms`);
    const chips = horizons.list.map((_, index) => {
      const windows = horizons.windows[index];
      const wrap = node("div", "hz-tick-chip", null, axis);
      wrap.style.left = pctX(index);
      const chip = node("button", "hz-chip", horizons.labels[index], wrap);
      chip.type = "button";
      chip.setAttribute("aria-label",
        `${horizons.labels[index]}: ${formatInt(windows)} ${pluralRu(windows, "окно", "окна", "окон")} проверки`);
      node("span", "hz-chip-prog", null, chip).setAttribute("aria-hidden", "true");
      node("div", "hz-windows", "●".repeat(windows), wrap).setAttribute("aria-hidden", "true");
      chip.addEventListener("click", () => pick(index));
      return { wrap, chip };
    });

    const legend = byId("hz-legend");
    models.forEach((model) => {
      const item = node("li", null, null, legend);
      node("span", `key hz-key-${primaryRole(model)}`, null, item);
      item.append(lower(model.label));
    });
    const bandItem = node("li", null, null, legend);
    node("span", "key hz-key-band", null, bandItem);
    bandItem.append("выигрыш наших моделей у эталона");
    const dotsItem = node("li", null, null, legend);
    node("span", "key hz-key-dots", "●●●", dotsItem);
    dotsItem.append("окна проверки на горизонте");

    const rows = models.map(() => {
      const item = node("li", "hz-row", null, rowsBox);
      const name = node("div", null, null, item);
      const text = node("span", "hz-row-name", null, name);
      const sub = node("span", "hz-row-sub", null, name);
      const fill = node("div", "hz-row-fill", null, node("div", "hz-row-track", null, name));
      const value = node("span", "hz-row-value", null, item);
      return { item, text, sub, fill, value };
    });

    let active = -1;
    let playing = false;
    let played = false;
    let timers = [];

    function setHint(auto) {
      hint.classList.toggle("is-auto", auto);
      byId("hz-hint-text").textContent = auto
        ? "горизонты листаются сами — нажмите на любой, чтобы остановить"
        : "нажимайте горизонты — или стрелки справа";
    }

    function select(index) {
      active = index;
      const best = bestOurs(index);
      byId("hz-sel").textContent = horizons.labels[index];
      byId("hz-gain").textContent = best ? `−${gainPct(index)}${NBSP}%` : "—";
      byId("hz-gain-sub").textContent = best ? `точнее эталона конкурса — ${lower(best.model.label)}` : "";
      const values = models.map((model) => model.mae[index]);
      const known = values.filter((value) => value !== null);
      const max = Math.max(...known);
      const min = Math.min(...known);
      const order = models.map((model, k) => ({ model, k, value: values[k] }))
        .sort((a, b) => (a.value === null ? Infinity : a.value) - (b.value === null ? Infinity : b.value));
      order.forEach(({ model, k, value }, position) => {
        const row = rows[k];
        row.item.style.order = String(position);
        row.item.classList.toggle("is-best", value !== null && value === min);
        row.item.classList.toggle("is-empty", value === null);
        row.text.textContent = model.label;
        row.sub.textContent = model.note;
        row.fill.style.width = value === null ? "0%" : `${((value / max) * 100).toFixed(1)}%`;
        row.value.textContent = value === null ? "не обучить" : `${formatInt(value)}${NBSP}₽`;
      });
      note.textContent = horizons.notes[index];
      byId("hz-band").style.left = pctX(index);
      pills.forEach((pill, k) => pill.classList.toggle("is-on", k === index));
      colButtons.forEach((button, k) => button.setAttribute("aria-pressed", String(k === index)));
      chips.forEach(({ wrap, chip }, k) => {
        chip.setAttribute("aria-pressed", String(k === index));
        wrap.classList.toggle("is-on", k === index);
        chip.classList.toggle("is-timing", playing && k === index);
      });
      setHint(playing || !played);
    }

    // Показ один раз, когда раздел виден: от первого горизонта по кругу, не больше HZ_MAX_STEPS шагов,
    // потом — основной горизонт. Любой выбор рукой останавливает показ.
    function stopPlay() {
      timers.forEach(clearTimeout);
      timers = [];
      playing = false;
      played = true;
    }
    function pick(index) {
      stopPlay();
      select((index + count) % count);
    }
    function play() {
      if (played) return;
      played = true;
      if (reducedMotion()) {
        select(active);
        return;
      }
      playing = true;
      let next = 0;
      let steps = 0;
      const step = () => {
        select(next);
        steps += 1;
        next = (next + 1) % count;
        const last = steps >= HZ_MAX_STEPS;
        timers.push(setTimeout(() => {
          if (!last) { step(); return; }
          playing = false;
          select(mainIndex);
        }, HZ_PLAY_STEP_MS));
      };
      timers.push(setTimeout(step, HZ_PLAY_FIRST_MS));
    }

    // Строка «точнее эталона…» и оговорка у горизонтов разной длины: их высота — по самому длинному
    // тексту из всех горизонтов, иначе панель прыгала бы при переключении. Пересчёт при смене ширины.
    // Пробник сохраняет отступы элемента: высота задаётся вместе с ними (box-sizing: border-box).
    const side = plot.closest(".hz").querySelector(".hz-side");
    const gainSub = byId("hz-gain-sub");
    const subTexts = horizons.list.map((_, index) => {
      const best = bestOurs(index);
      return best ? `точнее эталона конкурса — ${lower(best.model.label)}` : "";
    });
    function reserveHeights() {
      [[gainSub, subTexts], [note, horizons.notes]].forEach(([element, texts]) => {
        const probe = node(element.tagName.toLowerCase(), element.className, null, side);
        probe.style.position = "absolute";
        probe.style.visibility = "hidden";
        probe.style.width = `${element.clientWidth}px`;
        probe.style.margin = "0";
        probe.style.minHeight = "0";
        let tallest = 0;
        texts.forEach((text) => {
          probe.textContent = text;
          tallest = Math.max(tallest, probe.offsetHeight);
        });
        probe.remove();
        element.style.minHeight = `${tallest}px`;
      });
    }

    byId("hz-prev").addEventListener("click", () => pick(active - 1));
    byId("hz-next").addEventListener("click", () => pick(active + 1));
    select(mainIndex);
    reserveHeights();
    if ("ResizeObserver" in window) new ResizeObserver(reserveHeights).observe(side);

    if ("IntersectionObserver" in window && !reducedMotion()) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          observer.disconnect();
          play();
        });
      }, { threshold: 0.5 });
      observer.observe(plot);
    } else {
      played = true;
      setHint(false);
    }
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
  // 06 · Метод: карта решения. Блоки стоят сеткой в разметке, связи между ними рисуются по их
  // положению на странице и перерисовываются при изменении ширины; наведение на блок (и фокус
  // с клавиатуры) подсвечивает его связи и соседей. На узком экране связей нет — колонки идут
  // одна под другой, порядок читается по заголовкам колонок.
  // ---------------------------------------------------------------------------

  // Связи карты: откуда, куда, подпись на линии. Идентификаторы — узлов разметки (id="mn-…").
  const METHOD_EDGES = [
    ["federal", "agg", ""], ["panel", "gbm", ""], ["categories", "gbm", "обучающие ряды"],
    ["agg", "gbm", "общий фактор"], ["panel", "two", ""], ["agg", "two", "разнос долями"],
    ["baselines", "origin", ""], ["gbm", "origin", ""], ["two", "origin", ""], ["origin", "metrics", ""],
    ["panel", "cp", ""], ["bench", "cp", ""], ["cp", "signal", ""],
  ];
  // Короткие имена блоков для строки связей на узком экране; у блока без имени берётся его заголовок.
  const METHOD_SHORT = {
    panel: "панель", federal: "длинные ряды", categories: "категории трат", baselines: "эталоны",
    bench: "стенд с врезками", agg: "прогноз фед. ряда", gbm: "бустинг", two: "двухэтапная",
    cp: "детекторы", origin: "скользящий origin", metrics: "метрики", signal: "сигнал по панели",
  };
  const METHOD_LOOP = 52; // вынос связи внутри одной колонки вправо от блоков, px: в пределах промежутка колонок
  const METHOD_ELBOW = 64; // связь через колонку: до этого расстояния от цели идёт прямо по своей строке, дальше — изгиб
  const METHOD_LOOP_LABEL_T = 0.35; // где на петле стоит островок-подпись: ещё до следующего блока, в свободном месте
  const METHOD_START = 2; // точка начала связи — сразу за краем блока-источника, px
  const METHOD_END = 3; // остриё наконечника — перед рамкой блока-цели, px
  const METHOD_DROP_GAP = 10; // низ опущенных блоков — на столько выше связи «эталоны → origin», px
  const METHOD_DOWN_MAX = 120; // соседи по колонке ближе этого — связь прямо вниз, дальше — петля

  function initMethod() {
    const map = byId("method-map");
    const svg = byId("method-links");
    if (!map || !svg) return;
    const nodes = Array.from(map.querySelectorAll(".method-node"));
    const nodeOf = (id) => byId(`mn-${id}`);
    const edges = METHOD_EDGES.filter(([from, to]) => nodeOf(from) && nodeOf(to)).map(([from, to, label]) => ({
      from, to, label,
      path: svgNode("path", {
        class: "method-link", "marker-start": "url(#method-dot)", "marker-end": "url(#method-arrow)",
      }, svg),
      text: label ? node("span", "method-link-label", label, map) : null,
    }));
    const bezier = (p0, p1, p2, p3, t) => {
      const u = 1 - t;
      return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
    };

    // Строка связей блока для узкого экрана (на широком скрыта стилями): «← откуда · → куда» по тому же
    // списку рёбер, что рисует линии, короткими именами; подпись ребра — в скобках.
    const shortName = (id) => METHOD_SHORT[id] || nodeOf(id).querySelector(".method-node-title").textContent;
    const relText = (list) => list.map(([other, label]) => shortName(other) + (label ? ` (${label})` : "")).join(" · ");
    nodes.forEach((element) => {
      const id = element.id.slice(3);
      const inbound = edges.filter((edge) => edge.to === id).map((edge) => [edge.from, edge.label]);
      const outbound = edges.filter((edge) => edge.from === id).map((edge) => [edge.to, edge.label]);
      if (!inbound.length && !outbound.length) return;
      const rel = node("span", "method-node-rel", null, element);
      [["←", inbound], ["→", outbound]].forEach(([arrow, list]) => {
        if (!list.length) return;
        const part = node("span", "", null, rel);
        node("b", "", arrow, part);
        part.appendChild(document.createTextNode(` ${relText(list)}`));
      });
    });

    // Блоки моделей и проверки опускаются так, чтобы двухэтапная модель стояла прямо над связью
    // «эталоны → origin» (она идёт по строке эталонов): сдвиг — от положения блоков без сдвига.
    function settleDrop() {
      map.style.setProperty("--method-drop", "0px");
      const two = nodeOf("two");
      const baselines = nodeOf("baselines");
      if (!two || !baselines) return;
      const line = baselines.getBoundingClientRect();
      const drop = line.top + line.height / 2 - METHOD_DROP_GAP - two.getBoundingClientRect().bottom;
      map.style.setProperty("--method-drop", `${Math.max(0, drop).toFixed(1)}px`);
    }

    function draw() {
      if (getComputedStyle(svg).display === "none") {
        map.style.setProperty("--method-drop", "0px");
        return;
      }
      settleDrop();
      const box = map.getBoundingClientRect();
      svg.setAttribute("viewBox", `0 0 ${box.width.toFixed(1)} ${box.height.toFixed(1)}`);
      edges.forEach((edge) => {
        const a = nodeOf(edge.from).getBoundingClientRect();
        const b = nodeOf(edge.to).getBoundingClientRect();
        const sameColumn = Math.abs(a.left - b.left) < 1;
        let x1 = a.right - box.left + METHOD_START;
        let y1 = a.top + a.height / 2 - box.top;
        let x2 = b.left - box.left - METHOD_END;
        let y2 = b.top + b.height / 2 - box.top;
        let d;
        let mx;
        let my;
        if (sameColumn && b.top - a.bottom < METHOD_DOWN_MAX) {
          // Соседи по колонке: прямо вниз, подпись — посередине.
          x1 = a.left + a.width / 2 - box.left; y1 = a.bottom - box.top + METHOD_START;
          x2 = x1; y2 = b.top - box.top - METHOD_END;
          d = `M${x1},${y1} L${x2},${y2}`;
          mx = x1; my = (y1 + y2) / 2;
        } else if (sameColumn) {
          // Через блок: петля справа от колонки, входит в блок сбоку; подпись — в начале петли, где свободно.
          x2 = b.right - box.left + METHOD_END;
          d = `M${x1},${y1} C${x1 + METHOD_LOOP},${y1} ${x2 + METHOD_LOOP},${y2} ${x2},${y2}`;
          mx = bezier(x1, x1 + METHOD_LOOP, x2 + METHOD_LOOP, x2, METHOD_LOOP_LABEL_T);
          my = bezier(y1, y1, y2, y2, METHOD_LOOP_LABEL_T);
        } else if (x2 - x1 > b.width) {
          // Через колонку: прямо по своей строке (там, где у промежуточной колонки нет блока), и уже в
          // последнем промежутке — изгиб к цели: линия вся на виду, а не под чужими блоками.
          const xh = x2 - METHOD_ELBOW;
          d = `M${x1},${y1} L${xh},${y1} C${xh + METHOD_ELBOW * 0.6},${y1} ${xh + METHOD_ELBOW * 0.4},${y2} ${x2},${y2}`;
          mx = (x1 + xh) / 2; my = y1;
        } else {
          const cx = (x2 - x1) * 0.5;
          d = `M${x1},${y1} C${x1 + cx},${y1} ${x2 - cx},${y2} ${x2},${y2}`;
          mx = (x1 + x2) / 2; my = (y1 + y2) / 2;
        }
        edge.path.setAttribute("d", d);
        if (edge.text) {
          // Островок-подпись сидит посреди своей линии.
          edge.text.style.left = `${mx.toFixed(1)}px`;
          edge.text.style.top = `${my.toFixed(1)}px`;
        }
      });
    }

    function highlight(id) {
      edges.forEach((edge) => {
        const on = id !== null && (edge.from === id || edge.to === id);
        edge.path.classList.toggle("is-on", on);
        edge.path.classList.toggle("is-off", id !== null && !on);
        edge.path.setAttribute("marker-end", on ? "url(#method-arrow-on)" : "url(#method-arrow)");
        edge.path.setAttribute("marker-start", on ? "url(#method-dot-on)" : "url(#method-dot)");
        if (edge.text) {
          edge.text.classList.toggle("is-on", on);
          edge.text.classList.toggle("is-off", id !== null && !on);
        }
      });
      nodes.forEach((element) => {
        const own = element.id.slice(3);
        const near = id !== null && own !== id
          && edges.some((edge) => (edge.from === id && edge.to === own) || (edge.to === id && edge.from === own));
        element.classList.toggle("is-near", near);
      });
    }

    nodes.forEach((element) => {
      const id = element.id.slice(3);
      ["mouseenter", "focus"].forEach((type) => element.addEventListener(type, () => highlight(id)));
      ["mouseleave", "blur"].forEach((type) => element.addEventListener(type, () => highlight(null)));
    });

    draw();
    if ("ResizeObserver" in window) new ResizeObserver(draw).observe(map);
    // Шрифты подгружаются после первого кадра: высота блоков меняется, связи — вместе с ней.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
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
    ["метод", initMethod],
  ].forEach((block) => {
    try {
      block[1]();
    } catch (error) {
      console.error(`Не удалось запустить блок «${block[0]}»`, error);
    }
  });
}());
