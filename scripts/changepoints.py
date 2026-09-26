"""Стенд разладок: шесть детекторов на синтетике и потоковый сигнал на реальной панели.

Шесть детекторов (`src/changepoints.py`) гоняются по потоковому протоколу
(`src/cp_bench.py`) на возмущениях известной величины, внесённых в реальные ряды:
все — при неизвестном числе изломов, методы со штрафом — на каждом штрафе сетки.
Затем штраф для реальных данных выбирается правилом `selection` — до того, как
посчитаны реальные данные, — и потоковый сигнал считается по всем рядам панели.
Протокол задан в `configs/changepoints.yaml` и больше нигде.

Файлы в `output.dir` каждый раз переписываются целиком — стенд не сливается
партиями, как `src/run.py`:

    cp_calibration.csv           штраф ядра, при котором доля ложных тревог как у PELT
    cp_bench.csv                 все строки стенда
    cp_summary.csv               детектор × режим × штраф: обнаружено, задержка, ложные, J Юдена
    cp_summary_by_magnitude.csv  то же по величине возмущения при выбранном штрафе
    cp_realtime.csv              потоковый сигнал на панели по месяцам, на каждом штрафе
    cp_realtime_events.csv       события: месяц перехода порога, задержка, доли в окне события
    cp_offline.csv               изломы по полному ряду — прежний офлайновый расчёт, для сравнения

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
    bench_sample, calibrate_kernel_penalty, detector_with_penalty, month_offset,
    panel_realtime, preprocess, run_bench, summarise,
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


def offline_shares(wide: pd.DataFrame, detector: str, mode: str, penalty: float) -> pd.DataFrame:
    """Доля МО с изломом в месяце по полному ряду — прежний расчёт отчёта, для сравнения.

    Детектор видит ряд целиком, поэтому о своевременности это ничего не говорит —
    только о том, где изломы в итоге оказались. Ряд с двумя изломами попадает
    в два месяца. Месяц излома — месяц нового уровня (`month_offset`).
    """
    months = pd.DatetimeIndex(wide.index).strftime("%Y-%m")
    detect = detector_with_penalty(detector, penalty)
    offset = month_offset(mode)
    counts = np.zeros(len(months), dtype=int)
    for col in wide.columns:
        for b in detect(preprocess(wide[col].to_numpy(dtype=float), mode)).breakpoints:
            if 0 <= b + offset < len(months):
                counts[b + offset] += 1
    return pd.DataFrame({"month": months, "share": counts / wide.shape[1] * 100, "n_breaks": counts})


def save(frame: pd.DataFrame, path: Path, whole: Sequence[str] = ()) -> None:
    """CSV целиком; колонки `whole` — целые с пропусками: «14», а не «14.0», пусто — нет значения."""
    frame.astype({column: "Int64" for column in whole}).to_csv(path, index=False)


def table(frame: pd.DataFrame, digits: str = ",.1f") -> str:
    return frame.to_string(index=False, float_format=lambda v: format(v, digits))


def main() -> int:
    parser = argparse.ArgumentParser(description="Стенд разладок и потоковый сигнал на панели")
    parser.add_argument("--config", default="configs/changepoints.yaml")
    args = parser.parse_args()
    started = time.perf_counter()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bench_cfg, realtime = cfg["bench"], cfg["realtime"]
    panel = load_panel(ROOT / cfg["data"]["path"])
    wide, report = build_matrix(panel, cfg["data"]["category"], max_gap=cfg["data"]["max_gap"])
    print(report.as_text(), end="\n\n")

    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    penalties = [float(p) for p in bench_cfg["penalties"]]
    sample = bench_sample(wide, bench_cfg["n_series"], bench_cfg["seed"])
    print(f"рядов в панели: {wide.shape[1]} | периодов: {wide.shape[0]} | в выборке стенда: {sample.shape[1]}")

    stage = time.perf_counter()
    calibration = calibrate_kernel_penalty(sample, penalties, realtime["mode"])
    save(calibration, out_dir / "cp_calibration.csv")
    print(f"\nКАЛИБРОВКА ШТРАФА ЯДРА: нетронутые ряды выборки, режим {realtime['mode']}; доля ложных "
          f"тревог ядра, %, — ближайшая к PELT при номинальном штрафе ({time.perf_counter() - stage:.0f} с)")
    print(table(calibration, ".4g"))
    kernel_pens = dict(zip(calibration["penalty"], calibration["kernel_pen"]))

    stage = time.perf_counter()
    print(f"\nСТЕНД: {sample.shape[1]} рядов; позиции {bench_cfg['positions']}, величины "
          f"{bench_cfg['magnitudes']}, возмущения {bench_cfg['kinds']}, режимы {bench_cfg['modes']}, "
          f"штрафы {penalties}")
    bench = run_bench(
        sample, positions=bench_cfg["positions"], magnitudes=bench_cfg["magnitudes"],
        kinds=bench_cfg["kinds"], modes=bench_cfg["modes"], detectors=bench_cfg["detectors"],
        penalties=penalties, seed=bench_cfg["seed"], kernel_penalties=kernel_pens, log=print,
    )
    save(bench, out_dir / "cp_bench.csv", whole=("position", "delay", "signal"))
    summary = summarise(bench)
    save(summary, out_dir / "cp_summary.csv")
    print(f"стенд: {len(bench)} строк, {time.perf_counter() - stage:.0f} с")
    print("\nСВОДКА: задержка в месяцах — среди обнаруженных; CUSUM — со своим порогом, штраф пуст")
    print(table(summary))

    # Штраф выбирается по стенду до того, как посчитаны реальные данные: иначе правило
    # можно было бы подогнать под картину на них.
    detector, mode = realtime["detector"], realtime["mode"]
    chosen = select_penalty(summary, detector, mode)
    candidates = summary[(summary["detector"] == detector) & (summary["mode"] == mode)]
    print(f"\nПРАВИЛО ВЫБОРА ШТРАФА: {cfg['selection']['rule']}")
    print(f"  {detector} / {mode}: " + "; ".join(
        f"штраф {p:g} → J Юдена {j:.1f}" for p, j in zip(candidates["penalty"], candidates["J Юдена"])))
    print(f"  выбран штраф: {chosen:g}")

    by_magnitude = summarise(bench[(bench["penalty"] == chosen) | bench["penalty"].isna()], by=("magnitude",))
    save(by_magnitude, out_dir / "cp_summary_by_magnitude.csv")
    print(f"\nПО ВЕЛИЧИНЕ ВОЗМУЩЕНИЯ при штрафе {chosen:g}")
    print(table(by_magnitude))

    stage = time.perf_counter()
    print(f"\nРЕАЛЬНЫЕ ДАННЫЕ: {detector} на {mode} в расширяющемся окне по {wide.shape[1]} рядам, "
          f"порог {realtime['threshold_share']}% МО; для сравнения — изломы по полному ряду")
    monthly_parts, event_parts, offline_parts = [], [], []
    for penalty in penalties:
        effective = kernel_pens[penalty] if detector == "kernel_rbf" else penalty  # как на стенде
        monthly, events = panel_realtime(
            wide, detector, mode, effective, realtime["threshold_share"], realtime["events"]
        )
        monthly_parts.append(monthly.assign(penalty=penalty))
        event_parts.append(events.assign(penalty=penalty, selected=penalty == chosen))
        offline_parts.append(offline_shares(wide, detector, mode, effective).assign(penalty=penalty))
        print(f"  штраф {penalty:g}: {time.perf_counter() - stage:.0f} с")
    shares = pd.concat(monthly_parts, ignore_index=True)[["penalty", "month", "share", "n_signals"]]
    events = pd.concat(event_parts, ignore_index=True)[
        ["penalty", "selected", "event", "crossed_month", "delay", "max_share", "share_in_window"]
    ]
    offline = pd.concat(offline_parts, ignore_index=True)[["penalty", "month", "share", "n_breaks"]]
    save(shares, out_dir / "cp_realtime.csv")
    save(events, out_dir / "cp_realtime_events.csv", whole=("delay",))
    save(offline, out_dir / "cp_offline.csv")

    print("\nСОБЫТИЯ: в окне события (до следующего) — первый месяц с долей не ниже порога, "
          "задержка в месяцах, наибольшая месячная доля и доля МО с началом эпизода за всё окно")
    print(table(events))
    side_by_side = pd.concat({
        "потоково": shares.pivot(index="month", columns="penalty", values="share"),
        "по полному ряду": offline.pivot(index="month", columns="penalty", values="share"),
    }, axis=1)
    print("\nДОЛЯ МО ПО МЕСЯЦАМ, %: начала эпизодов потокового сигнала и изломы по полному ряду; "
          "во второй строке шапки — штраф")
    print(side_by_side.to_string(float_format=lambda v: f"{v:.1f}"))

    names = ["cp_calibration.csv", "cp_bench.csv", "cp_summary.csv", "cp_summary_by_magnitude.csv",
             "cp_realtime.csv", "cp_realtime_events.csv", "cp_offline.csv"]
    print(f"\nсохранено в {out_dir}: " + ", ".join(names))
    print(f"время работы: {(time.perf_counter() - started) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
