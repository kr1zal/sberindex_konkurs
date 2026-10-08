/* site/landing-lib.js — чистые помощники главной: подписи чисел и шкал, учёт наборов подписей оси,
 * разбор чисел для счёта, раскладка истории на графике. Страницы здесь нет:
 * ни document, ни window, — поэтому файл целиком запускает node в тестах
 * (tests/test_landing_lib.py). Со страницей работает site/landing.js. Данные и числа сюда не
 * пишутся: всё приходит аргументами.
 */
"use strict";

const LandingLib = (() => {
  const NBSP = " ";

  // ---------------------------------------------------------------------------
  // Числа и слова
  // ---------------------------------------------------------------------------

  // Неразрывный пробел в разрядах, как в отчёте.
  function groupDigits(digits) {
    return digits.replace(/\B(?=(\d{3})+(?!\d))/g, NBSP);
  }

  function formatInt(value) {
    return groupDigits(String(Math.round(value)));
  }

  function pluralRu(n, one, few, many) {
    const m = Math.abs(Math.trunc(n)) % 100;
    if (m >= 11 && m <= 14) return many;
    const last = m % 10;
    if (last === 1) return one;
    if (last >= 2 && last <= 4) return few;
    return many;
  }

  // ---------------------------------------------------------------------------
  // Шкала и подписи оси
  // ---------------------------------------------------------------------------

  // «Красивая» шкала: шаг из ряда 1, 2, 2,5, 5 × 10^k, границы кратны шагу и охватывают
  // все значения; ticks — отметки оси внутри границ.
  function niceScale(min, max, wanted) {
    const span = max - min || Math.abs(max) || 1;
    const raw = span / wanted;
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const fraction = raw / power;
    const step = (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10) * power;
    const lo = Math.floor(min / step + 1e-9) * step;
    const hi = Math.ceil(max / step - 1e-9) * step;
    const ticks = [];
    for (let tick = lo; tick <= hi + step * 1e-6; tick += step) {
      ticks.push(Math.round(tick * 1e6) / 1e6);
    }
    return { lo, hi, step, ticks };
  }

  // Знаков после запятой, которых подписи отметки хватает, чтобы шаг шкалы читался без
  // округления: шаг 10 — ни одного, шаг 2,5 или 0,5 — один, шаг 0,25 — два.
  function stepDecimals(step) {
    for (let digits = 0; digits < 3; digits += 1) {
      const scaled = Math.abs(step) * Math.pow(10, digits);
      if (Math.abs(scaled - Math.round(scaled)) < 1e-6) return digits;
    }
    return 2;
  }

  // Подпись отметки: разряды через неразрывный пробел, десятичная запятая, минус «−»; знаков
  // после запятой столько, сколько нужно шагу шкалы: при шаге 2,5 подпись «97,5», а не «98».
  function formatTick(value, step) {
    const parts = Math.abs(value).toFixed(stepDecimals(step)).split(".");
    const negative = value < 0 && Number(parts.join(".")) !== 0;
    return (negative ? "−" : "") + groupDigits(parts[0]) + (parts.length > 1 ? `,${parts[1]}` : "");
  }


  // Учёт наборов подписей оси. Набор — все подписи одной шкалы. При смене шкалы новый набор
  // встаёт поверх, а все прежние — и те, что ещё не убраны после прошлой смены, — уходят: живой
  // набор в любой момент ровно один. Модель ничего не знает о странице: show() возвращает
  // номера, а страница рисует и убирает элементы сама.
  //   show(animated) → { added, leaving, removed }
  //     added    — номер нового набора;
  //     leaving  — номера наборов, которые начали уходить (до сих пор были живыми);
  //     removed  — номера наборов, которые надо убрать сразу (без анимации).
  //   finish(id) — набор убран со страницы.
  function createTickSets() {
    let counter = 0;
    let sets = []; // [{ id, leaving }] в порядке появления
    return {
      show(animated) {
        counter += 1;
        const added = counter;
        let leaving = [];
        let removed = [];
        if (animated) {
          leaving = sets.filter((set) => !set.leaving).map((set) => set.id);
          sets.forEach((set) => { set.leaving = true; });
        } else {
          removed = sets.map((set) => set.id);
          sets = [];
        }
        sets.push({ id: added, leaving: false });
        return { added, leaving, removed };
      },
      finish(id) {
        sets = sets.filter((set) => set.id !== id);
      },
      ids() {
        return sets.map((set) => set.id);
      },
      liveIds() {
        return sets.filter((set) => !set.leaving).map((set) => set.id);
      },
    };
  }

  // ---------------------------------------------------------------------------
  // График проверки фактом на узком экране
  // ---------------------------------------------------------------------------

  // Сколько последних точек истории оставить на графике шириной width (px), чтобы шаг между
  // точками был не меньше minStep: точки проверки нарисованы кружками, и при меньшем шаге они
  // наползают друг на друга. Остаётся хотя бы одна точка истории; на широком графике — вся.
  function historyKeep(nHistory, nCheck, width, minStep) {
    if (!(width > 0) || !(minStep > 0)) return nHistory;
    const maxPoints = Math.floor(width / minStep) + 1;
    return Math.max(1, Math.min(nHistory, maxPoints - nCheck));
  }

  // ---------------------------------------------------------------------------
  // Счёт чисел: разбор текста и текст на каждом кадре
  // ---------------------------------------------------------------------------

  // Число в тексте: разряды через неразрывный пробел, десятичная запятая. Диапазон «от–до%»
  // даёт два числа.
  const NUMBER_PATTERN = /\d+(?: \d{3})*(?:,\d+)?/g;

  function numberTokens(text) {
    return Array.from(text.matchAll(NUMBER_PATTERN), (match) => {
      const raw = match[0];
      const comma = raw.indexOf(",");
      return {
        start: match.index,
        end: match.index + raw.length,
        value: parseFloat(raw.replace(/ /g, "").replace(",", ".")),
        digits: comma >= 0 ? raw.length - comma - 1 : 0,
        grouped: raw.indexOf(NBSP) >= 0,
      };
    });
  }

  // Текст с числами, умноженными на progress (0…1): знаки после запятой и разряды те же, что
  // в исходном; всё, что не число, остаётся как было. При progress = 1 — исходный текст.
  function countedText(original, tokens, progress) {
    let out = "";
    let position = 0;
    tokens.forEach((token) => {
      const parts = (token.value * progress).toFixed(token.digits).split(".");
      out += original.slice(position, token.start);
      out += (token.grouped ? groupDigits(parts[0]) : parts[0]) + (parts.length > 1 ? `,${parts[1]}` : "");
      position = token.end;
    });
    return out + original.slice(position);
  }

  // ---------------------------------------------------------------------------
  // Подпись региона у муниципалитета без региона
  // ---------------------------------------------------------------------------

  const NO_REGION = "регион не определён";

  return {
    groupDigits, formatInt, pluralRu,
    niceScale, stepDecimals, formatTick, createTickSets, historyKeep,
    numberTokens, countedText,
    NO_REGION,
  };
})();
