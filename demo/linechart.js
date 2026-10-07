/* demo/linechart.js — линейный график в SVG, свой, без библиотек и CDN.
 *
 * Ничего не знает о муниципалитетах, моделях или формате данных стенда: получает
 * готовые ряды (числа по месяцам) и подписи, а форматирует и довычисляет только то,
 * что относится к самому графику (шкалы, деления, подсказка). Смысл данных и тексты
 * страницы — в demo.js.
 *
 * Использование:
 *   const chart = LineChart.create(containerEl, {
 *     months: ["2023-01", ..., "2025-12"],       // ось X целиком, по порядку
 *     series: [{
 *       id, label, color,        // color — CSS-значение, обычно var(--chart-...)
 *       dash,                    // null | "dashed" | "dotted"
 *       mode,                   // "line" (по умолчанию) | "dots" — только точки
 *       overlay,                 // true — не показывать отдельной строкой в легенде
 *                                 //   и в подсказке (используется для наложенных точек
 *                                 //   факта поверх уже показанной линии факта)
 *       values,                  // number|null по одному на месяц, длина = months.length
 *     }, ...],
 *     markers: [{ atMonth, kind: "end" | "break", label }],
 *     markerKeys: [{ kind, label }],   // ключи отметок в легенде: только у тех видов, что есть
 *                                       //   среди markers (например, { kind: "break", label: "изломы" })
 *     yFormat: (value) => string,
 *     yUnit: "руб. на человека в месяц",
 *     ariaLabel: "...",
 *     legendEl: HTMLElement | null,   // куда вывести HTML-легенду (может быть null)
 *   });
 *   chart.update(newSpec);   // пересобрать с новыми данными на том же контейнере
 *   chart.destroy();         // снять ResizeObserver и обработчики
 *   LineChart.assignMarkerRows(boxes, rows, gap);  // строки подписей отметок; вынесено,
 *                            // чтобы проверять раскладку без браузера
 *
 * Числа графика — не единственный способ их прочитать: подсказка дополняет таблицу
 * под графиком, а не заменяет её (таблицу строит demo.js из тех же данных).
 */
"use strict";

const LineChart = (() => {
  const SVG_NS = "http://www.w3.org/2000/svg";

  const MONTH_SHORT = ["янв", "фев", "мар", "апр", "май", "июн",
                        "июл", "авг", "сен", "окт", "ноя", "дек"];

  // Высота фигуры целиком, включая полосу подписей оси X (иначе контейнер с
  // фиксированной высотой обрезает последнюю строку подписей — частая ошибка).
  const HEIGHT_WIDE = 340;
  const HEIGHT_NARROW = 240;
  const NARROW_BREAKPOINT = 480;

  const DASH = { dashed: "8 5", dotted: "2 4" };

  // Подписи отметок: первая строка — на 10 px ниже верхнего края поля графика, следующие
  // — через 12 px, строк не больше трёх (дальше подпись остаётся в последней).
  const MARKER_LABEL_TOP = 10;
  const MARKER_LABEL_STEP = 12;
  const MARKER_LABEL_ROWS = 3;

  function parseYM(ym) {
    const [y, m] = ym.split("-").map(Number);
    return { y, m };
  }

  /** «2025-01» → «янв 2025» (withYear) или «янв» — подписи оси X. */
  function formatMonthShort(ym, withYear) {
    const { y, m } = parseYM(ym);
    return withYear ? `${MONTH_SHORT[m - 1]} ${y}` : MONTH_SHORT[m - 1];
  }

  /**
   * Строки подписей отметок. boxes — горизонтальные границы подписей по порядку X
   * ({ left, right }). Подпись встаёт в первую из rows строк, где она не ближе gap к уже
   * поставленным; если свободной строки нет — остаётся в последней. Возвращает номера
   * строк, по одному на подпись.
   */
  function assignMarkerRows(boxes, rows, gap) {
    const taken = [];
    return boxes.map(({ left, right }) => {
      let row = 0;
      while (row < rows - 1
        && taken.some((t) => t.row === row && left < t.right + gap && t.left < right + gap)) {
        row += 1;
      }
      taken.push({ row, left, right });
      return row;
    });
  }

  function svgEl(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const key in attrs) {
        if (attrs[key] !== null && attrs[key] !== undefined) node.setAttribute(key, attrs[key]);
      }
    }
    return node;
  }

  function isNum(v) {
    return v !== null && v !== undefined && Number.isFinite(v);
  }

  /** Индексы подряд идущих ненулевых точек — отдельный путь на каждую пробежку,
   * чтобы пропуск в данных не рисовался соединяющей линией. */
  function buildRuns(values) {
    const runs = [];
    let current = null;
    values.forEach((v, i) => {
      if (isNum(v)) {
        if (!current) { current = []; runs.push(current); }
        current.push(i);
      } else {
        current = null;
      }
    });
    return runs;
  }

  /** «Круглые» шаги 1-2-5 — деления оси, которые не удивляют глаз. */
  function niceStep(raw) {
    if (!Number.isFinite(raw) || raw <= 0) return 1;
    const exp = Math.floor(Math.log10(raw));
    const base = Math.pow(10, exp);
    const frac = raw / base;
    const niceFrac = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
    return niceFrac * base;
  }

  function computeYTicks(min, max, targetCount) {
    if (min === max) { min -= 1; max += 1; }
    const step = niceStep((max - min) / (targetCount || 5));
    const niceMin = Math.floor(min / step) * step;
    const niceMax = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + step / 2; v += step) {
      ticks.push(Math.round(v * 1e6) / 1e6);
    }
    return ticks;
  }

  /** Деления оси X: сперва кварталы (янв/апр/июл/окт, год — у января), если не
   * помещаются подписи — только январи, если и они не помещаются — прореживаем
   * года. Тики и подписи выбираются вместе: не подписанных делений не рисуем. */
  function computeXTicks(months, plotWidthPx) {
    const n = months.length;
    if (n < 2) {
      return months.map((m, i) => ({ index: i, label: formatMonthShort(m, true) }));
    }
    const pxPerStep = plotWidthPx / (n - 1);
    const MIN_GAP_SHORT = 40;
    const MIN_GAP_YEAR = 56;

    const quarterIdx = [];
    const januaryIdx = [];
    months.forEach((m, i) => {
      const mm = parseYM(m).m;
      if (mm === 1) { januaryIdx.push(i); quarterIdx.push(i); }
      else if (mm === 4 || mm === 7 || mm === 10) quarterIdx.push(i);
    });

    const gapOf = (idx) => idx.length > 1
      ? (idx[idx.length - 1] - idx[0]) / (idx.length - 1) * pxPerStep
      : Infinity;

    if (quarterIdx.length && gapOf(quarterIdx) >= MIN_GAP_SHORT) {
      return quarterIdx.map(i => ({
        index: i, label: formatMonthShort(months[i], parseYM(months[i]).m === 1),
      }));
    }
    if (januaryIdx.length && gapOf(januaryIdx) >= MIN_GAP_YEAR) {
      return januaryIdx.map(i => ({ index: i, label: formatMonthShort(months[i], true) }));
    }
    // Прореживаем январи по годам, пока подписи не перестанут спорить за место.
    for (let step = 2; step <= januaryIdx.length; step += 1) {
      const thinned = januaryIdx.filter((_, k) => k % step === 0);
      if (gapOf(thinned) >= MIN_GAP_YEAR || thinned.length <= 1) {
        return thinned.map(i => ({ index: i, label: formatMonthShort(months[i], true) }));
      }
    }
    return [0, n - 1].map(i => ({ index: i, label: formatMonthShort(months[i], true) }));
  }

  function create(container, spec) {
    let currentSpec = spec;
    let width = 0;
    let height = HEIGHT_WIDE;
    let margin = { top: 28, right: 18, bottom: 34, left: 64 };
    let scaleX = (i) => i;
    let scaleY = (v) => v;
    // Ссылка на элемент курсора, а не id: на странице стенда два графика создаются
    // этой же функцией, и id="chart-cursor" повторялся бы в обоих SVG — невалидный
    // HTML, который до сих пор работал только потому, что querySelector ниже был
    // ограничен своим svg.
    let cursorEl = null;

    container.innerHTML = "";
    const svg = svgEl("svg", { role: "img", "aria-label": spec.ariaLabel || "" });
    const titleNode = document.createElementNS(SVG_NS, "title");
    svg.appendChild(titleNode);
    container.appendChild(svg);

    const tooltip = document.createElement("div");
    // night — ночные токены site.css: плашка тёмная на любой теме, а ключи рядов в ней
    // берут цвета, подобранные под тёмный фон.
    tooltip.className = "chart-tooltip night";
    tooltip.hidden = true;
    container.appendChild(tooltip);

    function plotWidth() { return Math.max(1, width - margin.left - margin.right); }
    function plotHeight() { return Math.max(1, height - margin.top - margin.bottom); }

    function visibleSeries() {
      return currentSpec.series.filter((s) => !s.overlay);
    }

    function renderLegend() {
      const el = currentSpec.legendEl;
      if (!el) return;
      el.innerHTML = "";
      const list = document.createElement("ul");
      list.className = "chart-legend-list";
      currentSpec.series.forEach((s) => {
        if (s.overlay) return;
        const item = document.createElement("li");
        item.className = "chart-legend-item";
        const key = document.createElement("span");
        // Класс по значению dash ("dashed"/"dotted"), а не один общий "-dashed" на любой
        // штрих: «как год назад» рисуется точками и должна выглядеть точками и в легенде,
        // а не как ещё одна штриховая линия.
        key.className = "chart-legend-key" + (s.dash ? ` chart-legend-key-${s.dash}` : "");
        key.style.setProperty("--legend-color", s.color);
        const text = document.createElement("span");
        text.textContent = s.label;
        item.append(key, text);
        list.appendChild(item);
      });
      // Ключ отметки — тем же штрихом и цветом, что сама отметка: вертикальная линия без ключа в
      // легенде ничего не говорит. Ключ — только у вида отметки, который на графике нарисован.
      (currentSpec.markerKeys || []).forEach(({ kind, label }) => {
        if (!(currentSpec.markers || []).some((marker) => marker.kind === kind)) return;
        const item = document.createElement("li");
        item.className = "chart-legend-item";
        const key = document.createElement("span");
        key.className = `chart-legend-key chart-legend-key-marker-${kind}`;
        const text = document.createElement("span");
        text.textContent = label;
        item.append(key, text);
        list.appendChild(item);
      });
      el.appendChild(list);
      if (currentSpec.yUnit) {
        const unit = document.createElement("span");
        unit.className = "chart-legend-unit";
        unit.textContent = currentSpec.yUnit;
        el.appendChild(unit);
      }
    }

    function nearestIndex(clientX) {
      const rect = svg.getBoundingClientRect();
      const xInSvg = (clientX - rect.left) * (width / rect.width) - margin.left;
      const n = currentSpec.months.length;
      const idx = Math.round(xInSvg / (plotWidth() / (n - 1)));
      return Math.min(n - 1, Math.max(0, idx));
    }

    function showTooltip(index, clientX) {
      const month = currentSpec.months[index];
      const rows = visibleSeries()
        // joinAt — точка стыка факта с началом линии прогноза: число там настоящее,
        // но это факт, а не прогноз, и подсказка его как прогноз/пунктир не подписывает —
        // только линия проходит через него.
        .map((s) => ({ s, v: s.values[index] }))
        .filter(({ s, v }) => isNum(v) && s.joinAt !== index);
      if (!rows.length) { hideTooltip(); return; }

      tooltip.innerHTML = "";
      const head = document.createElement("div");
      head.className = "chart-tooltip-month";
      head.textContent = formatMonthShort(month, true);
      tooltip.appendChild(head);
      // Единица — под месяцем: число без неё («Факт 21 722») читается как «чего?».
      if (currentSpec.yUnit) {
        const unit = document.createElement("div");
        unit.className = "chart-tooltip-unit";
        unit.textContent = currentSpec.yUnit;
        tooltip.appendChild(unit);
      }
      rows.forEach(({ s, v }) => {
        const row = document.createElement("div");
        row.className = "chart-tooltip-row";
        const key = document.createElement("span");
        key.className = "chart-tooltip-key" + (s.dash ? ` chart-tooltip-key-${s.dash}` : "");
        key.style.setProperty("--legend-color", s.color);
        const label = document.createElement("span");
        label.className = "chart-tooltip-label";
        label.textContent = s.label;
        const value = document.createElement("span");
        value.className = "chart-tooltip-value";
        value.textContent = currentSpec.yFormat(v);
        row.append(key, label, value);
        tooltip.appendChild(row);
      });
      tooltip.hidden = false;

      const rect = container.getBoundingClientRect();
      let left = scaleX(index) + 12;
      const maxLeft = rect.width - tooltip.offsetWidth - 4;
      if (left > maxLeft) left = scaleX(index) - tooltip.offsetWidth - 12;
      tooltip.style.left = `${Math.max(4, left)}px`;
      tooltip.style.top = `${margin.top}px`;

      if (cursorEl) {
        cursorEl.setAttribute("x1", scaleX(index));
        cursorEl.setAttribute("x2", scaleX(index));
        cursorEl.setAttribute("visibility", "visible");
      }
    }

    function hideTooltip() {
      tooltip.hidden = true;
      if (cursorEl) cursorEl.setAttribute("visibility", "hidden");
    }

    function onPointerMove(event) {
      showTooltip(nearestIndex(event.clientX), event.clientX);
    }
    function onDocPointerDown(event) {
      if (!container.contains(event.target)) hideTooltip();
    }
    // Escape закрывает подсказку, не двигая указателя: содержимое, появившееся при наведении, должно
    // закрываться без движения мыши (WCAG 1.4.13).
    function onDocKeyDown(event) {
      if (event.key === "Escape") hideTooltip();
    }

    svg.addEventListener("pointermove", onPointerMove);
    svg.addEventListener("pointerdown", onPointerMove);
    svg.addEventListener("pointerleave", (event) => {
      // Указатель мыши покидает график — уходим; палец, наоборот, обычно
      // «поднимается» в другом месте, поэтому касание закрывает подсказку
      // отдельным document-обработчиком, а не здесь.
      if (event.pointerType !== "touch") hideTooltip();
    });
    document.addEventListener("pointerdown", onDocPointerDown);
    document.addEventListener("keydown", onDocKeyDown);

    function draw() {
      width = container.clientWidth;
      if (width <= 0) return;
      const narrow = width < NARROW_BREAKPOINT;
      height = narrow ? HEIGHT_NARROW : HEIGHT_WIDE;
      // Подписи оси — моноширинные: «50 000» занимает ширину шести знаков, слева нужен запас.
      margin = narrow
        ? { top: 24, right: 12, bottom: 30, left: 52 }
        : { top: 28, right: 18, bottom: 34, left: 64 };

      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
      titleNode.textContent = currentSpec.ariaLabel || "";
      svg.setAttribute("aria-label", currentSpec.ariaLabel || "");
      while (svg.lastChild !== titleNode) svg.removeChild(svg.lastChild);

      const months = currentSpec.months;
      const n = months.length;
      const pw = plotWidth();
      const ph = plotHeight();
      scaleX = (i) => margin.left + (n > 1 ? (i * pw) / (n - 1) : pw / 2);

      let min = Infinity;
      let max = -Infinity;
      currentSpec.series.forEach((s) => s.values.forEach((v) => {
        if (isNum(v)) { if (v < min) min = v; if (v > max) max = v; }
      }));
      if (!Number.isFinite(min)) { min = 0; max = 1; }
      const pad = (max - min) * 0.08 || Math.abs(max || 1) * 0.08;
      const yTicks = computeYTicks(min - pad, max + pad, narrow ? 4 : 5);
      const yMin = yTicks[0];
      const yMax = yTicks[yTicks.length - 1];
      scaleY = (v) => margin.top + ph - ((v - yMin) / (yMax - yMin)) * ph;

      const gridGroup = svgEl("g", { class: "chart-grid" });
      yTicks.forEach((t) => {
        gridGroup.appendChild(svgEl("line", {
          x1: margin.left, x2: margin.left + pw, y1: scaleY(t), y2: scaleY(t),
          class: "chart-gridline",
        }));
        const label = svgEl("text", {
          x: margin.left - 8, y: scaleY(t), class: "chart-axis-label", "text-anchor": "end",
          "dominant-baseline": "middle",
        });
        label.textContent = currentSpec.yFormat(t);
        gridGroup.appendChild(label);
      });
      svg.appendChild(gridGroup);

      const xTicks = computeXTicks(months, pw);
      const axisGroup = svgEl("g", { class: "chart-axis" });
      axisGroup.appendChild(svgEl("line", {
        x1: margin.left, x2: margin.left + pw, y1: margin.top + ph, y2: margin.top + ph,
        class: "chart-axis-line",
      }));
      xTicks.forEach(({ index, label }) => {
        const x = scaleX(index);
        const anchor = index === 0 ? "start" : index === n - 1 ? "end" : "middle";
        const text = svgEl("text", {
          x, y: margin.top + ph + 18, class: "chart-axis-label", "text-anchor": anchor,
        });
        text.textContent = label;
        axisGroup.appendChild(text);
      });
      svg.appendChild(axisGroup);

      // «Январь» подписан с годом и потому шире соседних коротких месяцев — крайняя
      // (сама широкая и по краю) подпись изредка задевает следующую, если между ними
      // всего один шаг деления. Реальная ширина известна только после рендера, поэтому
      // расчёт деления (computeXTicks) её не учитывает — здесь чистим по факту:
      // подпись, задевающая уже оставленную соседнюю, снимается целиком (тик остаётся).
      let lastLabelRight = -Infinity;
      axisGroup.querySelectorAll("text").forEach((node) => {
        const box = node.getBBox();
        if (box.x < lastLabelRight + 4) {
          node.remove();
        } else {
          lastLabelRight = box.x + box.width;
        }
      });

      // Отметки: конец панели (тонкая сплошная) и изломы (штрих, с подписью месяца).
      // Сначала все линии, затем все подписи: подпись лежит над линиями соседних отметок,
      // а не под ними.
      const markerGroup = svgEl("g", { class: "chart-markers" });
      const orderedMarkers = (currentSpec.markers || [])
        .map((marker) => ({ marker, idx: months.indexOf(marker.atMonth) }))
        .filter(({ idx }) => idx >= 0)
        .sort((a, b) => a.idx - b.idx);
      orderedMarkers.forEach(({ marker, idx }) => {
        const x = scaleX(idx);
        markerGroup.appendChild(svgEl("line", {
          x1: x, x2: x, y1: margin.top, y2: margin.top + ph,
          class: `chart-marker chart-marker-${marker.kind}`,
        }));
      });
      const markerLabels = orderedMarkers.map(({ marker, idx }) => {
        const x = scaleX(idx);
        const anchor = x > margin.left + pw * 0.75 ? "end" : "start";
        // Вид отметки — в классе подписи: у изломов подпись своего цвета, как и сама линия.
        const labelText = svgEl("text", {
          x: x + (anchor === "end" ? -4 : 4), y: margin.top + MARKER_LABEL_TOP,
          class: `chart-marker-label chart-marker-label-${marker.kind}`, "text-anchor": anchor,
        });
        labelText.textContent = marker.label;
        markerGroup.appendChild(labelText);
        return labelText;
      });
      svg.appendChild(markerGroup);

      // Ширина подписи известна только после рендера (моноширинный шрифт шире обычного),
      // поэтому раскладка — по факту. Подпись, которой не хватает места справа от отметки
      // («конец панели» стоит близко к правому краю), переходит налево от неё: иначе правый
      // край графика обрезал бы её на середине. Затем подпись встаёт в первую строку, где не
      // задевает уже поставленные: изломы идут с шагом в несколько месяцев, а на телефоне
      // график втрое уже, и подписи соседних отметок сходятся в одной строке.
      const labelBoxes = markerLabels.map((node) => {
        let box = node.getBBox();
        if (node.getAttribute("text-anchor") === "start" && box.x + box.width > width - 4) {
          node.setAttribute("text-anchor", "end");
          node.setAttribute("x", Number(node.getAttribute("x")) - 8);
          box = node.getBBox();
        }
        return { left: box.x, right: box.x + box.width };
      });
      assignMarkerRows(labelBoxes, MARKER_LABEL_ROWS, 4).forEach((row, i) => {
        markerLabels[i].setAttribute("y", margin.top + MARKER_LABEL_TOP + row * MARKER_LABEL_STEP);
      });

      // Ряды: путь на каждую непрерывную пробежку точек, чтобы пропуск не рисовал линию.
      const seriesGroup = svgEl("g", { class: "chart-series" });
      currentSpec.series.forEach((s) => {
        const mode = s.mode || "line";
        const runs = buildRuns(s.values);
        if (mode === "dots") {
          s.values.forEach((v, i) => {
            if (!isNum(v)) return;
            const dot = svgEl("circle", {
              cx: scaleX(i), cy: scaleY(v), r: 3.5, class: "chart-dot",
            });
            dot.style.fill = s.color;
            seriesGroup.appendChild(dot);
          });
          return;
        }
        runs.forEach((run) => {
          const d = run.map((i, k) => `${k === 0 ? "M" : "L"} ${scaleX(i)} ${scaleY(s.values[i])}`).join(" ");
          // id ряда — в классе: толщину линии задаёт стиль страницы, а не сам график.
          const path = svgEl("path", { d, class: `chart-line chart-line-${s.id}`, fill: "none" });
          path.style.stroke = s.color;
          if (s.dash && DASH[s.dash]) path.style.strokeDasharray = DASH[s.dash];
          seriesGroup.appendChild(path);
          if (run.length > 1) {
            const last = run[run.length - 1];
            const endDot = svgEl("circle", { cx: scaleX(last), cy: scaleY(s.values[last]), r: 4, class: "chart-end-dot" });
            endDot.style.fill = s.color;
            seriesGroup.appendChild(endDot);
          }
        });
      });
      svg.appendChild(seriesGroup);

      // Курсор подсказки — поверх рядов, скрыт до наведения/касания. Ссылка сохраняется
      // в cursorEl (без id — см. комментарий в create()), draw() пересоздаёт её при
      // каждой перерисовке (resize, update), поэтому переприсваивание, а не const.
      cursorEl = svgEl("line", {
        x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + ph,
        class: "chart-cursor", visibility: "hidden",
      });
      svg.appendChild(cursorEl);

      // Прозрачная область поверх графика — единая цель для наведения/касания.
      const hit = svgEl("rect", {
        x: margin.left, y: margin.top, width: pw, height: ph, class: "chart-hit-area", fill: "transparent",
      });
      svg.appendChild(hit);

      renderLegend();
    }

    const resizeObserver = new ResizeObserver(() => draw());
    resizeObserver.observe(container);
    draw();

    // Лишние подписи оси снимаются по измеренной ширине (getBBox), а моноширинный шрифт
    // может догрузиться позже первой отрисовки и стать шире запасного: когда шрифты
    // загрузились — график перерисовывается с настоящими размерами.
    let alive = true;
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => { if (alive) draw(); });
    }

    return {
      update(nextSpec) {
        currentSpec = nextSpec;
        hideTooltip();
        draw();
      },
      destroy() {
        alive = false;
        resizeObserver.disconnect();
        document.removeEventListener("pointerdown", onDocPointerDown);
        document.removeEventListener("keydown", onDocKeyDown);
        container.innerHTML = "";
      },
    };
  }

  return { create, formatMonthShort, assignMarkerRows };
})();
