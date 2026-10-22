"""Baut abscross_dict.pkl speicherschonend und fortsetzbar neu auf.

Jedes Gas wird in Teilbereichen (Chunks) berechnet und als .npy in einem
Arbeitsordner abgelegt. Erst am Ende wird die Pickle-Datei zusammengesetzt.
Bei Abbruch einfach erneut starten: fertige Gase werden uebersprungen.
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import tomexplorer_core as core


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def build_gas(gas: str, axis: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[list[float]]]:
    xsc_ok = core._local_xsc_can_build(gas, args.temp, args.pressure, float(axis[0]), float(axis[-1]))
    if not xsc_ok:
        core._ensure_species_available(gas, float(axis[0]), float(axis[-1]))
    ranges = core._coverage_ranges_for_gas_request(gas, args.temp, args.pressure, float(axis[0]), float(axis[-1]))
    if not ranges:
        raise ValueError("keine Daten im Bereich")
    sigma = np.zeros_like(axis)
    chunk_points = max(2, int(round(args.chunk / args.step)))
    for start in range(0, axis.size, chunk_points):
        stop = min(axis.size, start + chunk_points)
        if stop - start < 2:
            continue
        sub_axis = axis[start:stop]
        mask = core._axis_mask_from_intervals(sub_axis, ranges)
        if not np.any(mask):
            continue
        for attempt in (1, 2, 3):
            try:
                nu, sig = core._sigma_from_local_db(
                    gas, args.temp, args.pressure, float(sub_axis[0]), float(sub_axis[-1]), args.step
                )
                break
            except MemoryError:
                core.hp.LOCAL_TABLE_CACHE.clear()
                gc.collect()
                if attempt == 3:
                    raise
                time.sleep(5)
        sigma[start:stop][mask] = np.interp(sub_axis[mask], nu, sig)
        del nu, sig
        log(f"  {gas}: {stop / axis.size:.0%}")
    return sigma, [[float(a), float(b)] for a, b in ranges]


def write_timings_from_log(log_path: Path, step: float, span: float) -> None:
    import json
    import re

    seconds: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"\] (\S+): OK in (\d+)s", line)
        if match:
            seconds[match.group(1)] = float(match.group(2))
    core.OFFLINE_TIMINGS_PATH.write_text(
        json.dumps({"step_cm1": step, "span_cm1": span, "seconds": seconds}, indent=1), encoding="utf-8"
    )
    log(f"Rechenzeiten fuer {len(seconds)} Gase in {core.OFFLINE_TIMINGS_PATH.name} geschrieben")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timings-from-log", help="nur Rechenzeiten aus einem Log extrahieren")
    parser.add_argument("--log-file", help="Logdatei dieses Laufs; Rechenzeiten werden fortlaufend daraus gespeichert")
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--temp", type=float, default=35.0)
    parser.add_argument("--pressure", type=float, default=1000.0)
    parser.add_argument("--min", type=float, default=1000.0, dest="nu_min")
    parser.add_argument("--max", type=float, default=10000.0, dest="nu_max")
    parser.add_argument("--chunk", type=float, default=250.0)
    parser.add_argument("--work", default="offline_build")
    parser.add_argument("--gases", nargs="*")
    parser.add_argument("--no-assemble", action="store_true")
    args = parser.parse_args()

    if args.timings_from_log:
        write_timings_from_log(Path(args.timings_from_log), args.step, args.nu_max - args.nu_min)
        return

    work = core.BASE_DIR / args.work
    work.mkdir(exist_ok=True)
    npoints = int(round((args.nu_max - args.nu_min) / args.step)) + 1
    axis = args.nu_min + np.arange(npoints, dtype=float) * args.step
    gases = sorted(args.gases or core.GAS_LIBRARY.keys())
    log(f"Start: {len(gases)} Gase, {npoints} Punkte, step={args.step}, T={args.temp} C, p={args.pressure} hPa")
    core._ensure_database_started()

    started = time.time()
    for index, gas in enumerate(gases, 1):
        sigma_path = work / f"{gas}.npy"
        meta_path = work / f"{gas}.ranges"
        fail_path = work / f"{gas}.failed"
        if sigma_path.exists() or fail_path.exists():
            log(f"[{index}/{len(gases)}] {gas}: bereits erledigt")
            continue
        t0 = time.time()
        try:
            sigma, ranges = build_gas(gas, axis, args)
            np.save(sigma_path.with_suffix(".tmp.npy"), sigma)
            os.replace(sigma_path.with_suffix(".tmp.npy"), sigma_path)
            meta_path.write_text(repr(ranges), encoding="utf-8")
            log(f"[{index}/{len(gases)}] {gas}: OK in {time.time() - t0:.0f}s, max={sigma.max():.3e}, Bereich={ranges}")
        except Exception as exc:
            traceback.print_exc()
            fail_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            log(f"[{index}/{len(gases)}] {gas}: FEHLER {type(exc).__name__}: {exc}")
        finally:
            core.hp.LOCAL_TABLE_CACHE.clear()
            core._clear_runtime_caches()
            gc.collect()
    log(f"Berechnung fertig nach {(time.time() - started) / 3600:.2f} h")

    if args.no_assemble:
        return
    library: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    coverage: dict[str, list[list[float]]] = {}
    for gas in gases:
        path = work / f"{gas}.npy"
        if path.exists():
            library[gas] = (axis, np.load(path))
            coverage[gas] = eval((work / f"{gas}.ranges").read_text(encoding="utf-8"))
    payload = {
        "library": library,
        "metadata": {
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "reference_temperature_c": args.temp,
            "reference_pressure_hpa": args.pressure,
            "native_step_cm1": args.step,
        },
        "coverage_ranges": coverage,
    }
    target = core.OFFLINE_SPECTRA_PATH
    new_path = target.with_suffix(".new")
    with new_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if target.exists():
        backup = target.with_name(f"abscross_dict_backup_{datetime.now():%Y%m%d_%H%M%S}.pkl")
        os.replace(target, backup)
        log(f"Altes Pickle gesichert als {backup.name}")
    os.replace(new_path, target)
    log(f"Neues Pickle geschrieben: {len(library)} Gase, {target.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
