"""Стенд разладок: шесть детекторов на синтетике и потоковый сигнал на реальной панели.

Шесть детекторов (`src/changepoints.py`) гоняются по потоковому протоколу
(`src/cp_bench.py`) на возмущениях известной величины, внесённых в реальные ряды:
все — при неизвестном числе изломов, методы со штрафом — на каждом штрафе сетки.
Затем штраф для реальных данных выбирается правилом `selection` — до того, как
посчитаны реальные данные, — и потоковый сигнал считается по всем рядам панели.
Протокол задан в конфиге и больше нигде: действующий — `configs/changepoints.yaml` (v2),
прежний — `configs/changepoints_v1.yaml`, он пишет в `results/cp_v1/`.

Файлы в `output.dir` каждый раз переписываются целиком — стенд не сливается
партиями, как `src/run.py`. Первая колонка каждого файла — `protocol`, версия протокола
(`v2`): файлы разных версий иначе не отличить.

    cp_calibration.csv           штраф ядра, при котором доля ложных тревог как у PELT
    cp_bench.csv                 все строки стенда
    cp_summary.csv               детектор × режим × штраф: обнаружено, задержка, ложные, J Юдена
    cp_summary_by_magnitude.csv  то же по величине возмущения при выбранном штрафе
    cp_summary_by_position.csv   то же по позиции врезки при выбранном штрафе
    cp_realtime.csv              потоковый сигнал на панели по месяцам, на каждом штрафе
    cp_realtime_events.csv       события: месяц перехода порога, задержка, доли в окне события
    cp_realtime_rank.csv         ранговый вид: доля месяца, ранг, медиана фона — при realtime.rank_view
    cp_offline.csv               изломы по полному ряду — прежний офлайновый расчёт, для сравнения
    cp_offline_series.csv        те же изломы по рядам, строка на излом: пары изломов считаются из файла

    .venv/bin/python -u scripts/changepoints.py [--config configs/changepoints.yaml]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cp_bench import (  # noqa: E402
    RANK_COLUMNS, bench_sample, calibrate_kernel_penalty, detector_with_penalty, effective_penalty,
    month_offset, panel_realtime, preprocess, realtime_rank, run_bench, summarise,
)
from src.data import build_matrix, load_panel  # noqa: E402


def select_penalty(summary: pd.DataFrame, detector: str, mode: str) -> float:
    """Штраф для реальных данных по правилу `selection.rule` конфига.

    Среди строк детектора и режима — штраф с наибольшим J Юдена, при равенстве меньший.
    J сравнивается округлённым: разность долей с разными знаменателями даёт равные
    по смыслу значения, которые расходятся в последнем разряде, и ничью решал бы
    шум округления, а не правило.
    """
    rows = summary[(summary["detector"] == detector) & (summary["mode"] == mode) & summary["penalty"].notna()]
    if rows.empty:
        raise ValueError(f"в сводке нет строк со штрафом для {detector} / {mode}")
    youden = rows["J Юдена"].round(9)
    return float(rows.loc[youden == youden.max(), "penalty"].min())


def offline_breaks(wide: pd.DataFrame, detector: str, mode: str, penalty: float) -> pd.DataFrame:
    """Изломы по полному ряду, строка на излом: ряд и месяц нового уровня (`month_offset`).

    Прежний расчёт отчёта. Детектор видит ряд целиком, поэтому о своевременности это
    ничего не говорит — только о том, где изломы в итоге оказались. По рядам, а не только
    долями — чтобы изломы одного ряда, например пару через три месяца, можно было считать
    из файла.
    """
    months = pd.DatetimeIndex(wide.index).strftime("%Y-%m")
    detect = detector_with_penalty(detector, penalty)
    offset = month_offset(mode)
    rows = []
    for col in wide.columns:
        for b in detect(preprocess(wide[col].to_numpy(dtype=float), mode)).breakpoints:
            if 0 <= b + offset < len(months):
                rows.append({"series_id": col, "month": months[b + offset]})
    return pd.DataFrame(rows, columns=["series_id", "month"])


def offline_shares(breaks: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Доля МО с изломом по полному ряду в каждом месяце панели; ряд с двумя изломами —
    в двух месяцах. Считается из `offline_breaks`, чтобы два файла не расходились."""
    months = pd.DatetimeIndex(wide.index).strftime("%Y-%m")
    counts = breaks["month"].value_counts().reindex(months, fill_value=0).to_numpy()
    return pd.DataFrame({"month": months, "share": counts / wide.shape[1] * 100, "n_breaks": counts})


def edge_warnings(calibration: pd.DataFrame, grid: np.ndarray) -> list[str]:
    """Предупреждения о калибровке, упёршейся в край сетки: ближайшая к PELT доля ложных
    тревог может лежать за её пределами, и штраф ядра тогда сопоставим лишь приблизительно."""
    edges = {float(np.min(grid)), float(np.max(grid))}
    return [
        f"ВНИМАНИЕ: калибровка на краю сетки — номинальный штраф {row.penalty:g}: штраф ядра "
        f"{row.kernel_pen:.4g}, ложных у ядра {row.fa_kernel:.1f}%, у PELT {row.fa_pelt:.1f}%"
        for row in calibration.itertuples(index=False) if row.kernel_pen in edges
    ]


def failure_warning(what: str, n_failed: int) -> list[str]:
    """Строка «ВНИМАНИЕ» о шагах, на которых детектор упал. Протокол считает их шагами
    без тревоги, и доли с ними занижены; без строки в логе это прошло бы молча."""
    if not n_failed:
        return []
    return [f"ВНИМАНИЕ: {what}: детектор упал на {int(n_failed)} шагах — они засчитаны как шаги без тревоги"]


def selection_edge_warning(chosen: float, penalties: Sequence[float]) -> list[str]:
    """Строка «ВНИМАНИЕ», если правило выбрало крайний штраф сетки: J Юдена мог бы расти
    и за её краем, и тогда лучший штраф лежит вне сетки. Сетку это не меняет — только видимость."""
    edges = {min(penalties): "наименьший", max(penalties): "наибольший"}
    if chosen not in edges:
        return []
    return [f"ВНИМАНИЕ: выбранный штраф на краю сетки — {chosen:g}, {edges[chosen]} из {sorted(penalties)}: "
            f"оптимум может лежать за её пределами"]


def position_lines(months: Sequence[str], positions: Sequence[int]) -> list[str]:
    """Позиции врезки месяцами панели и строка «ВНИМАНИЕ» для позиции в декабре или январе.

    Врезка в месяц сезонного скачка засчитывает обнаружением сам скачок: в v1 так вышло
    с позицией 11 — декабрём 2023. Протокол этим не проверяется и не меняется — только видимость.
    """
    lines = ["позиции врезки: " + ", ".join(f"{p} → {months[p]}" for p in positions)]
    for p in positions:
        name = {"12": "декабрь", "01": "январь"}.get(months[p][-2:])
        if name:
            lines.append(f"ВНИМАНИЕ: позиция врезки {p} → {months[p]}, {name}: сезонный скачок этого месяца "
                         f"совпадает с врезкой и может засчитываться её обнаружением")
    return lines


def save(frame: pd.DataFrame, path: Path, protocol: str, whole: Sequence[str] = ()) -> None:
    """CSV целиком, первой колонкой — версия протокола: файлы v1 и v2 иначе не отличить.
    Колонки `whole` — целые с пропусками: «14», а не «14.0», пусто — нет значения.
    `frame` не меняется: после записи он идёт в лог и дальше в расчёт."""
    out = frame.astype({column: "Int64" for column in whole})
    out.insert(0, "protocol", protocol)
    out.to_csv(path, index=False)


def table(frame: pd.DataFrame, digits: str = ",.1f") -> str:
    return frame.to_string(index=False, float_format=lambda v: format(v, digits))


def main() -> int:
    parser = argparse.ArgumentParser(description="Стенд разладок и потоковый сигнал на панели")
    parser.add_argument("--config", default="configs/changepoints.yaml")
    args = parser.parse_args()
    started = time.perf_counter()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    # Версия — первой строкой лога и первой колонкой каждого файла: результаты v1 и v2
    # лежат рядом, и по файлу или логу должно быть видно, чей он.
    protocol = f"v{cfg['protocol_version']}"
    print(f"ПРОТОКОЛ СТЕНДА: {protocol} ({args.config})")
    bench_cfg, realtime = cfg["bench"], cfg["realtime"]
    # Ключи протокола без умолчаний, и читаются сразу: без них прогон падает в начале,
    # а не через несколько минут. max_delay: null — без границы задержки, как в v1.
    max_delay, twin_rule, rank_view = bench_cfg["max_delay"], bench_cfg["twin_rule"], realtime["rank_view"]
    panel = load_panel(ROOT / cfg["data"]["path"])
    wide, report = build_matrix(panel, cfg["data"]["category"], max_gap=cfg["data"]["max_gap"])
    print(report.as_text(), end="\n\n")

    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    penalties = [float(p) for p in bench_cfg["penalties"]]
    sample = bench_sample(wide, bench_cfg["n_series"], bench_cfg["seed"])
    print(f"рядов в панели: {wide.shape[1]} | периодов: {wide.shape[0]} | в выборке стенда: {sample.shape[1]}")
    for line in position_lines(pd.DatetimeIndex(wide.index).strftime("%Y-%m"), bench_cfg["positions"]):
        print(line)

    stage = time.perf_counter()
    spec = bench_cfg["kernel_pen_grid"]
    grid = np.geomspace(spec["min"], spec["max"], spec["points"])
    calibration = calibrate_kernel_penalty(sample, penalties, realtime["mode"], grid=grid)
    save(calibration, out_dir / "cp_calibration.csv", protocol)
    print(f"\nКАЛИБРОВКА ШТРАФА ЯДРА: нетронутые ряды выборки, режим {realtime['mode']}, сетка "
          f"{spec['min']:g}–{spec['max']:g} из {spec['points']} точек; доля ложных тревог ядра, %, — "
          f"ближайшая к PELT при номинальном штрафе ({time.perf_counter() - stage:.0f} с)")
    print(table(calibration, ".4g"))
    for line in edge_warnings(calibration, grid):
        print(line)
    for row in calibration.itertuples(index=False):
        for line in (failure_warning(f"калибровка, PELT при штрафе {row.penalty:g}", row.n_failed_pelt)
                     + failure_warning(f"калибровка, ядро при штрафе {row.kernel_pen:.4g}", row.n_failed_kernel)):
            print(line)
    # сбои ядра по всей сетке — одно число на калибровку, в каждой строке таблицы то же
    grid_failed = int(calibration["n_failed_grid"].max()) if not calibration.empty else 0
    for line in failure_warning("калибровка, ядро на сетке штрафа", grid_failed):
        print(line)
    kernel_pens = dict(zip(calibration["penalty"], calibration["kernel_pen"]))

    stage = time.perf_counter()
    print(f"\nСТЕНД: {sample.shape[1]} рядов; позиции {bench_cfg['positions']}, величины "
          f"{bench_cfg['magnitudes']}, возмущения {bench_cfg['kinds']}, режимы {bench_cfg['modes']}, "
          f"штрафы {penalties}; зачёт: max_delay {max_delay}, twin_rule {twin_rule}")
    bench = run_bench(
        sample, positions=bench_cfg["positions"], magnitudes=bench_cfg["magnitudes"],
        kinds=bench_cfg["kinds"], modes=bench_cfg["modes"], detectors=bench_cfg["detectors"],
        penalties=penalties, seed=bench_cfg["seed"], max_delay=max_delay, twin_rule=twin_rule,
        kernel_penalties=kernel_pens, log=print,
    )
    save(bench, out_dir / "cp_bench.csv", protocol, whole=("position", "delay", "signal"))
    summary = summarise(bench)
    save(summary, out_dir / "cp_summary.csv", protocol)
    print(f"стенд: {len(bench)} строк, {time.perf_counter() - stage:.0f} с")
    print("\nСВОДКА: задержка в месяцах — среди обнаруженных; CUSUM — со своим порогом, штраф пуст")
    print(table(summary))
    for row in summary.itertuples(index=False):
        penalty = "—" if pd.isna(row.penalty) else f"{row.penalty:g}"  # у CUSUM штрафа нет
        for line in failure_warning(f"стенд, {row.detector} / {row.mode} / штраф {penalty}", row.n_failed):
            print(line)

    # Штраф выбирается по стенду до того, как посчитаны реальные данные: иначе правило
    # можно было бы подогнать под картину на них.
    detector, mode = realtime["detector"], realtime["mode"]
    chosen = select_penalty(summary, detector, mode)
    candidates = summary[(summary["detector"] == detector) & (summary["mode"] == mode)]
    # Текст правила собирается из конфига: детектор и режим те же, что у select_penalty.
    print(f"\nПРАВИЛО ВЫБОРА ШТРАФА: {cfg['selection']['rule'].format(detector=detector, mode=mode)}")
    print(f"  {detector} / {mode}: " + "; ".join(
        f"штраф {p:g} → J Юдена {j:.1f}" for p, j in zip(candidates["penalty"], candidates["J Юдена"])))
    print(f"  выбран штраф: {chosen:g}")
    for line in selection_edge_warning(chosen, penalties):
        print(line)

    at_chosen = bench[(bench["penalty"] == chosen) | bench["penalty"].isna()]  # CUSUM — со своим порогом
    by_magnitude = summarise(at_chosen, by=("magnitude",))
    save(by_magnitude, out_dir / "cp_summary_by_magnitude.csv", protocol)
    print(f"\nПО ВЕЛИЧИНЕ ВОЗМУЩЕНИЯ при штрафе {chosen:g}")
    print(table(by_magnitude))
    by_position = summarise(at_chosen, by=("position",))
    save(by_position, out_dir / "cp_summary_by_position.csv", protocol, whole=("position",))
    print(f"\nПО ПОЗИЦИИ ВРЕЗКИ при штрафе {chosen:g}")
    print(table(by_position.astype({"position": "Int64"})))

    stage = time.perf_counter()
    print(f"\nРЕАЛЬНЫЕ ДАННЫЕ: {detector} на {mode} в расширяющемся окне по {wide.shape[1]} рядам, "
          f"порог {realtime['threshold_share']}% МО; для сравнения — изломы по полному ряду")
    monthly_parts, event_parts, rank_parts, offline_parts, break_parts = [], [], [], [], []
    for penalty in penalties:
        effective = effective_penalty(detector, penalty, kernel_pens)  # как на стенде
        monthly, events = panel_realtime(
            wide, detector, mode, effective, realtime["threshold_share"], realtime["events"]
        )
        monthly_parts.append(monthly.assign(penalty=penalty))
        event_parts.append(events.assign(penalty=penalty, selected=penalty == chosen))
        if rank_view:
            ranked = realtime_rank(monthly, realtime["events"])
            rank_parts.append(ranked.assign(penalty=penalty, selected=penalty == chosen))
        breaks = offline_breaks(wide, detector, mode, effective)
        break_parts.append(breaks.assign(penalty=penalty))
        offline_parts.append(offline_shares(breaks, wide).assign(penalty=penalty))
        n_failed = int(monthly["n_failed"].sum())
        print(f"  штраф {penalty:g}: {time.perf_counter() - stage:.0f} с, шагов со сбоем детектора: {n_failed}")
        for line in failure_warning(f"реальные данные, {detector} / {mode} / штраф {penalty:g}", n_failed):
            print(line)
    shares = pd.concat(monthly_parts, ignore_index=True)[["penalty", "month", "share", "n_signals", "n_failed"]]
    events = pd.concat(event_parts, ignore_index=True)[
        ["penalty", "selected", "event", "crossed_month", "delay", "max_share", "share_in_window"]
    ]
    offline = pd.concat(offline_parts, ignore_index=True)[["penalty", "month", "share", "n_breaks"]]
    offline_series = pd.concat(break_parts, ignore_index=True)[["penalty", "series_id", "month"]]
    save(shares, out_dir / "cp_realtime.csv", protocol)
    save(events, out_dir / "cp_realtime_events.csv", protocol, whole=("delay",))
    save(offline, out_dir / "cp_offline.csv", protocol)
    save(offline_series, out_dir / "cp_offline_series.csv", protocol)

    print("\nСОБЫТИЯ: в окне события (до следующего) — первый месяц с долей не ниже порога, "
          "задержка в месяцах, наибольшая месячная доля и доля МО с началом эпизода за всё окно")
    print(table(events))
    if rank_view:
        rank = pd.concat(rank_parts, ignore_index=True)[["penalty", "selected", *RANK_COLUMNS]]
        save(rank, out_dir / "cp_realtime_rank.csv", protocol)
        print(f"\nРАНГОВЫЙ ВИД при штрафе {chosen:g} — описание, не второй порог: доля МО с началом эпизода "
              f"в месяц события, её ранг среди наблюдаемых месяцев (1 — наибольшая доля), медиана долей "
              f"месяцев без событий и отношение к ней")
        # два знака: один ряд панели — 0,05%, и фон «0.0%» при конечном отношении сбивал бы с толку
        for row in rank[rank["selected"] & rank["event"].notna()].itertuples(index=False):
            ratio = "—" if pd.isna(row.ratio_to_background) else f"{row.ratio_to_background:.2f}"
            print(f"  {row.event}: доля {row.share:.2f}%, ранг {row.rank:g} из {row.n_months}, "
                  f"медиана фона {row.background_median:.2f}%, отношение {ratio}")
    side_by_side = pd.concat({
        "потоково": shares.pivot(index="month", columns="penalty", values="share"),
        "по полному ряду": offline.pivot(index="month", columns="penalty", values="share"),
    }, axis=1)
    print("\nДОЛЯ МО ПО МЕСЯЦАМ, %: начала эпизодов потокового сигнала и изломы по полному ряду; "
          "во второй строке шапки — штраф")
    print(side_by_side.to_string(float_format=lambda v: f"{v:.1f}"))

    names = ["cp_calibration.csv", "cp_bench.csv", "cp_summary.csv", "cp_summary_by_magnitude.csv",
             "cp_summary_by_position.csv", "cp_realtime.csv", "cp_realtime_events.csv",
             *(["cp_realtime_rank.csv"] if rank_view else []), "cp_offline.csv", "cp_offline_series.csv"]
    print(f"\nсохранено в {out_dir}: " + ", ".join(names))
    print(f"время работы: {(time.perf_counter() - started) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
