from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from itertools import combinations
import math
import json
from pathlib import Path
import pickle
import re
import shutil
from typing import Any

import numpy as np
from scipy.signal import find_peaks

import hapi as hp


PA_PER_ATM = 101325.0
PA_PER_HPA = 100.0
TORR_PER_HPA = 0.750061683
BOLTZMANN = 1.380649e-23
CM3_PER_M3 = 1_000_000.0
DEFAULT_MANUAL_STEP_CM1 = 0.01
DEFAULT_SEARCH_STEP_CM1 = 0.02
DEFAULT_MAX_PLOT_POINTS = 6000
FETCH_MARGIN_CM1 = 5.0
PICKLE_REBUILD_TEMPERATURE_C = 35.0
PICKLE_REBUILD_PRESSURE_HPA = 1013.25
LOCAL_CACHE_MAX_GASES = 10
INTERFERENCE_WORST_CASE_SWEEP_FACTORS = (0.0, 0.5, 1.0)
TARGET_WORST_CASE_SWEEP_FACTORS = (1.0, 2.0, 5.0)
MIN_ACCEPTABLE_SIGNAL_TO_INTERFERENCE = 1.0
MIN_ACCEPTABLE_DELTA_ALPHA_SELECTIVITY = 1.3
MIN_ACCEPTABLE_WMS2F_SELECTIVITY = 1.0
MIN_ACCEPTABLE_WMS2F_SHAPE_SIMILARITY = 0.72
MIN_CANDIDATE_PROMINENCE_RATIO = 0.04
MIN_MULTI_TARGET_CANDIDATE_SCORE = -220.0
MAX_LOCAL_PEAKS_PER_WINDOW_GAS = 12

BASE_DIR = Path(__file__).resolve().parent
LEGACY_DATA_DIR = BASE_DIR / "data"
DATA_DIR = BASE_DIR / "hitran_cache"
XSC_CACHE_DIR = DATA_DIR / "xsc"
OFFLINE_SPECTRA_PATH = BASE_DIR / "abscross_dict.pkl"
OFFLINE_DB_MODE = "offline"
LIVE_DB_MODE = "live"

PREFERRED_GAS_COLORS = {
    "H2O": "#1d9bf0",
    "CO2": "#9a6700",
    "O3": "#8b5cf6",
    "N2O": "#16a34a",
    "CO": "#ef4444",
    "CH4": "#22c55e",
    "NO": "#64748b",
    "SO2": "#fb7185",
    "NO2": "#f97316",
    "C2H6": "#2563eb",
}
CATALOG_FALLBACK_COLORS = (
    "#0f766e",
    "#7c3aed",
    "#b45309",
    "#c2410c",
    "#0f766e",
    "#1d4ed8",
    "#be123c",
    "#4d7c0f",
    "#4338ca",
    "#0369a1",
)
DISTINCT_PLOT_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be123c",
    "#65a30d",
    "#c2410c",
    "#0f766e",
)

HITRAN_MOLECULE_NAMES = {
    1: "Water",
    2: "Carbon dioxide",
    3: "Ozone",
    4: "Nitrous oxide",
    5: "Carbon monoxide",
    6: "Methane",
    7: "Oxygen",
    8: "Nitric oxide",
    9: "Sulfur dioxide",
    10: "Nitrogen dioxide",
    11: "Ammonia",
    12: "Nitric acid",
    13: "Hydroxyl",
    14: "Hydrogen fluoride",
    15: "Hydrogen chloride",
    16: "Hydrogen bromide",
    17: "Hydrogen iodide",
    18: "Chlorine monoxide",
    19: "Carbonyl sulfide",
    20: "Formaldehyde",
    21: "Hypochlorous acid",
    22: "Nitrogen",
    23: "Hydrogen cyanide",
    24: "Methyl chloride",
    25: "Hydrogen peroxide",
    26: "Acetylene",
    27: "Ethane",
    28: "Phosphine",
    29: "Carbonyl fluoride",
    30: "Sulfur hexafluoride",
    31: "Hydrogen sulfide",
    32: "Formic acid",
    33: "Hydroperoxyl",
    34: "Atomic oxygen",
    35: "Chlorine nitrate",
    36: "Nitric oxide cation",
    37: "Hypobromous acid",
    38: "Ethylene",
    39: "Methanol",
    40: "Methyl bromide",
    41: "Acetonitrile",
    42: "Tetrafluoromethane",
    43: "Diacetylene",
    44: "Cyanoacetylene",
    45: "Hydrogen",
    46: "Carbon monosulfide",
    47: "Sulfur trioxide",
    48: "Cyanogen",
    49: "Phosgene",
    50: "Sulfur monoxide",
    51: "Fluoromethane",
    52: "Germane",
    53: "Carbon disulfide",
    54: "Methyl iodide",
    55: "Nitrogen trifluoride",
    56: "Trihydrogen cation",
    57: "Methyl radical",
    58: "Disulfur",
    59: "Carbonyl chlorofluoride",
    60: "Nitrous acid",
    61: "Nitryl chloride",
}

XSC_GAS_ALIASES: dict[str, tuple[str, ...]] = {
    "C2F6": ("CF3CF3",),
    "CF3CF3": ("C2F6",),
}


def _gas_lookup_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)).upper()


def _gas_lookup_tokens(gas: str) -> set[str]:
    base_token = _gas_lookup_token(gas)
    tokens: set[str] = {base_token} if base_token else set()
    for alias in XSC_GAS_ALIASES.get(base_token, tuple()):
        alias_token = _gas_lookup_token(alias)
        if alias_token:
            tokens.add(alias_token)
    return tokens


def _safe_gas_key(label: str, molecule_id: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    if not cleaned:
        return f"M{molecule_id}"
    if cleaned[0].isdigit():
        return f"M{molecule_id}_{cleaned}"
    return cleaned


def _dropdown_label(formula: str, molecule_id: int | None) -> str:
    if molecule_id is None:
        return f"{formula} - lokale XSC"
    molecule_name = HITRAN_MOLECULE_NAMES.get(molecule_id)
    if not molecule_name:
        return formula
    return f"{formula} - {molecule_name}"


def _discover_local_xsc_gases() -> tuple[str, ...]:
    if not XSC_CACHE_DIR.exists():
        return tuple()

    discovered: set[str] = set()
    for directory in sorted(path for path in XSC_CACHE_DIR.iterdir() if path.is_dir()):
        name = directory.name.strip()
        if not name:
            continue
        expected_tokens = _gas_lookup_tokens(name)
        if not expected_tokens:
            continue
        has_matching_xsc = any(
            _gas_lookup_token(path.stem.split("_", 1)[0]) in expected_tokens
            for path in directory.rglob("*.xsc")
        )
        if has_matching_xsc:
            discovered.add(name)

    for path in sorted(XSC_CACHE_DIR.rglob("*.xsc")):
        stem = path.stem.strip()
        if not stem:
            continue
        gas = stem.split("_", 1)[0].strip()
        if gas:
            discovered.add(gas)

    return tuple(sorted(discovered))


def _component_color_map(gases: tuple[str, ...]) -> dict[str, str]:
    return {
        gas: DISTINCT_PLOT_COLORS[index % len(DISTINCT_PLOT_COLORS)]
        for index, gas in enumerate(gases)
    }


def _build_gas_library() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    used_keys: set[str] = set()
    fallback_index = 0
    mol_name_index = hp.ISO_INDEX["mol_name"]

    for (molecule_id, isotope_id), payload in sorted(hp.ISO.items()):
        if isotope_id != 1:
            continue

        label = str(payload[mol_name_index])
        key = _safe_gas_key(label, molecule_id)
        if key in used_keys:
            key = f"M{molecule_id}_{key}"
        used_keys.add(key)

        color = PREFERRED_GAS_COLORS.get(label)
        if color is None:
            color = CATALOG_FALLBACK_COLORS[fallback_index % len(CATALOG_FALLBACK_COLORS)]
            fallback_index += 1

        catalog[key] = {
            "molecule_id": molecule_id,
            "isotope_id": isotope_id,
            "label": label,
            "dropdown_label": _dropdown_label(label, molecule_id),
            "color": color,
        }

    for gas in _discover_local_xsc_gases():
        if gas in catalog:
            continue
        color = CATALOG_FALLBACK_COLORS[fallback_index % len(CATALOG_FALLBACK_COLORS)]
        fallback_index += 1
        catalog[gas] = {
            "molecule_id": None,
            "isotope_id": None,
            "label": gas,
            "dropdown_label": _dropdown_label(gas, None),
            "color": color,
        }

    plot_colors = _component_color_map(tuple(sorted(catalog.keys())))
    for gas, plot_color in plot_colors.items():
        catalog[gas]["plot_color"] = plot_color

    return catalog


GAS_LIBRARY: dict[str, dict[str, Any]] = _build_gas_library()

_FETCHED_RANGES: dict[str, tuple[float, float]] = {}
_DB_STARTED = False
_DB_PATH: str | None = None


@dataclass(frozen=True)
class ComponentSpectrum:
    gas: str
    color: str
    concentration: float
    sigma_cm2_per_molecule: np.ndarray
    alpha_per_cm: np.ndarray
    peak_wavenumber_cm1: float
    peak_wavelength_um: float


@dataclass(frozen=True)
class ManualSpectrumResult:
    wavenumber_cm1: np.ndarray
    wavelength_um: np.ndarray
    total_sigma_cm2_per_molecule: np.ndarray
    total_alpha_per_cm: np.ndarray
    components: dict[str, ComponentSpectrum]
    temperature_c: float
    pressure_hpa: float
    step_cm1: float
    range_label: str
    coverage_ranges_cm1_by_gas: dict[str, tuple[tuple[float, float], ...]]
    missing_ranges_cm1_by_gas: dict[str, tuple[tuple[float, float], ...]]
    source_details_by_gas: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CrossSectionSegment:
    gas: str
    path: Path
    nu_min: float
    nu_max: float
    temperature_k: float | None
    pressure_torr: float | None
    common_name: str
    broadener: str
    reference: str
    resolution: float | None
    axis_cm1: np.ndarray
    sigma_cm2_per_molecule: np.ndarray


@dataclass(frozen=True)
class WindowGasMetric:
    gas: str
    peak_alpha_per_cm: float
    peak_sigma_cm2_per_molecule: float
    peak_wavelength_um: float
    peak_wavenumber_cm1: float
    peak_index: int
    interference_alpha_per_cm: float
    signal_to_interference: float
    prominence_ratio: float
    peak_region_target_contrast_per_cm: float = 0.0
    peak_region_interference_contrast_per_cm: float = 0.0
    peak_region_selectivity: float = 0.0
    peak_region_purity: float = 0.0
    peak_region_target_delta_alpha_per_cm: float = 0.0
    peak_region_interference_delta_alpha_per_cm: float = 0.0
    peak_region_delta_alpha_selectivity: float = 0.0
    peak_region_wms2f_selectivity: float = 0.0
    peak_region_wms2f_shape_similarity: float = 0.0


@dataclass(frozen=True)
class WindowGasDiagnostic:
    gas: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    metric: WindowGasMetric | None = None


@dataclass(frozen=True)
class LaserWindowCandidate:
    window_id: str
    wavelength_min_um: float
    wavelength_max_um: float
    center_wavelength_um: float
    tuning_span_nm: float
    coverage: tuple[str, ...]
    score: float
    gas_metrics: dict[str, WindowGasMetric]


@dataclass(frozen=True)
class LaserPlan:
    rank: int
    score: float
    covered_targets: tuple[str, ...]
    missing_targets: tuple[str, ...]
    windows: tuple[LaserWindowCandidate, ...]


@dataclass(frozen=True)
class WindowCandidateDiagnostic:
    wavelength_min_um: float
    wavelength_max_um: float
    tuning_span_nm: float
    coverage: tuple[str, ...]
    score: float
    gas_diagnostics: dict[str, WindowGasDiagnostic]


def gas_options() -> list[dict[str, str]]:
    return [
        {"label": GAS_LIBRARY[gas]["dropdown_label"], "value": gas}
        for gas in sorted(GAS_LIBRARY.keys())
    ]


def wavelength_um_to_wavenumber_cm1(wavelength_um: float | np.ndarray) -> float | np.ndarray:
    return 1.0e4 / np.asarray(wavelength_um)


def wavenumber_cm1_to_wavelength_um(wavenumber_cm1: float | np.ndarray) -> float | np.ndarray:
    return 1.0e4 / np.asarray(wavenumber_cm1)


def concentration_to_molar_fraction(value: float, unit: str) -> float:
    unit_key = unit.lower()
    if unit_key == "fraction":
        return value
    if unit_key == "%":
        return value / 100.0
    if unit_key == "ppm":
        return value * 1.0e-6
    if unit_key == "ppb":
        return value * 1.0e-9
    raise ValueError(f"Unsupported concentration unit: {unit}")


def format_concentration(molar_fraction: float) -> str:
    if molar_fraction >= 1.0e-2:
        return f"{molar_fraction * 100:.3g} %"
    if molar_fraction >= 1.0e-6:
        return f"{molar_fraction * 1.0e6:.3g} ppm"
    return f"{molar_fraction * 1.0e9:.3g} ppb"


def normalize_wavenumber_window(
    range_unit: str,
    range_min: float,
    range_max: float,
) -> tuple[float, float]:
    if range_min <= 0 or range_max <= 0:
        raise ValueError("Spectral range values must be positive.")

    if range_unit == "um":
        wl_min = min(range_min, range_max)
        wl_max = max(range_min, range_max)
        nu_min = float(wavelength_um_to_wavenumber_cm1(wl_max))
        nu_max = float(wavelength_um_to_wavenumber_cm1(wl_min))
        return nu_min, nu_max

    if range_unit == "cm-1":
        return float(min(range_min, range_max)), float(max(range_min, range_max))

    raise ValueError(f"Unsupported spectral range unit: {range_unit}")


def total_number_density_cm3(temperature_c: float, pressure_hpa: float) -> float:
    temperature_k = temperature_c + 273.15
    pressure_pa = pressure_hpa * PA_PER_HPA
    return pressure_pa / (BOLTZMANN * temperature_k) / CM3_PER_M3


def recommended_step_cm1(span_cm1: float, manual_mode: bool = True) -> float:
    if manual_mode:
        return max(DEFAULT_MANUAL_STEP_CM1, span_cm1 / 80_000.0)
    return max(DEFAULT_SEARCH_STEP_CM1, span_cm1 / 60_000.0)


def downsample_indices(length: int, max_points: int = DEFAULT_MAX_PLOT_POINTS) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, max_points, dtype=int)


def downsample_manual_result(
    result: ManualSpectrumResult,
    max_points: int = DEFAULT_MAX_PLOT_POINTS,
) -> ManualSpectrumResult:
    indices = downsample_indices(len(result.wavenumber_cm1), max_points=max_points)
    if len(indices) == len(result.wavenumber_cm1):
        return result

    components: dict[str, ComponentSpectrum] = {}
    for gas, component in result.components.items():
        peak_index = int(np.argmax(component.alpha_per_cm[indices]))
        components[gas] = ComponentSpectrum(
            gas=component.gas,
            color=component.color,
            concentration=component.concentration,
            sigma_cm2_per_molecule=component.sigma_cm2_per_molecule[indices],
            alpha_per_cm=component.alpha_per_cm[indices],
            peak_wavenumber_cm1=float(result.wavenumber_cm1[indices][peak_index]),
            peak_wavelength_um=float(result.wavelength_um[indices][peak_index]),
        )

    return ManualSpectrumResult(
        wavenumber_cm1=result.wavenumber_cm1[indices],
        wavelength_um=result.wavelength_um[indices],
        total_sigma_cm2_per_molecule=result.total_sigma_cm2_per_molecule[indices],
        total_alpha_per_cm=result.total_alpha_per_cm[indices],
        components=components,
        temperature_c=result.temperature_c,
        pressure_hpa=result.pressure_hpa,
        step_cm1=result.step_cm1,
        range_label=result.range_label,
        coverage_ranges_cm1_by_gas=result.coverage_ranges_cm1_by_gas,
        missing_ranges_cm1_by_gas=result.missing_ranges_cm1_by_gas,
        source_details_by_gas=result.source_details_by_gas,
    )


def _ensure_database_started() -> None:
    global _DB_STARTED, _DB_PATH
    DATA_DIR.mkdir(exist_ok=True)
    db_path = str(DATA_DIR)
    if _DB_STARTED and _DB_PATH == db_path:
        return
    hp.db_begin(db_path)
    _DB_STARTED = True
    _DB_PATH = db_path


def _clear_runtime_caches() -> None:
    _load_offline_sigma_library.cache_clear()
    offline_library_metadata.cache_clear()
    _cached_offline_sigma_bundle.cache_clear()
    _cached_sigma_bundle.cache_clear()
    _local_table_range.cache_clear()
    _load_local_xsc_segments.cache_clear()


def _has_local_table_cache(gas: str) -> bool:
    return (DATA_DIR / f"{gas}.data").exists() and (DATA_DIR / f"{gas}.header").exists()


def _xsc_search_directories() -> tuple[Path, ...]:
    return (XSC_CACHE_DIR, DATA_DIR)


def _iter_local_xsc_paths(gas: str) -> tuple[Path, ...]:
    expected_tokens = _gas_lookup_tokens(gas)
    if not expected_tokens:
        return tuple()
    seen: set[Path] = set()
    matches: list[Path] = []
    for directory in _xsc_search_directories():
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.xsc")):
            stem_token = _gas_lookup_token(path.stem.split("_", 1)[0])
            if stem_token not in expected_tokens:
                continue
            if path not in seen:
                seen.add(path)
                matches.append(path)
    return tuple(matches)


def _parse_xsc_optional_float(field: str) -> float | None:
    text = field.strip().replace("D", "E")
    if not text:
        return None
    return float(text)


def _parse_xsc_filename_metadata(path: Path) -> dict[str, float | str | None]:
    patterns = (
        r"^(?P<gas>[A-Za-z0-9]+)_(?P<temperature>-?\d+(?:\.\d+)?)K[_-](?P<pressure>-?\d+(?:\.\d+)?)Torr_(?P<nu_min>-?\d+(?:\.\d+)?)\-(?P<nu_max>-?\d+(?:\.\d+)?)_(?P<reference>[^.]+)$",
        r"^(?P<gas>[A-Za-z0-9]+)_(?P<temperature>-?\d+(?:\.\d+)?)_(?P<pressure>-?\d+(?:\.\d+)?)_(?P<nu_min>-?\d+(?:\.\d+)?)\-(?P<nu_max>-?\d+(?:\.\d+)?)_(?P<reference>[^.]+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, path.stem)
        if not match:
            continue
        return {
            "gas": match.group("gas"),
            "temperature_k": float(match.group("temperature")),
            "pressure_torr": float(match.group("pressure")),
            "nu_min": float(match.group("nu_min")),
            "nu_max": float(match.group("nu_max")),
            "reference": match.group("reference"),
        }
    return {
        "gas": None,
        "temperature_k": None,
        "pressure_torr": None,
        "nu_min": None,
        "nu_max": None,
        "reference": "",
    }


def _is_xsc_header_line(line: str) -> bool:
    if len(line) < 60:
        return False
    formula = line[:20].strip()
    if not formula or " " in formula:
        return False
    try:
        float(line[20:30].strip().replace("D", "E"))
        float(line[30:40].strip().replace("D", "E"))
        int(line[40:47].strip())
    except ValueError:
        return False
    return True


@lru_cache(maxsize=64)
def _load_local_xsc_segments(path_str: str, mtime_ns: int, size_bytes: int) -> tuple[CrossSectionSegment, ...]:
    del mtime_ns, size_bytes
    path = Path(path_str)
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    segments: list[CrossSectionSegment] = []

    first_non_empty = next((line for line in raw_lines if line.strip()), "")
    if first_non_empty and not _is_xsc_header_line(first_non_empty):
        axis, sigma = hp.read_xsect(str(path))
        axis_array = np.asarray(axis, dtype=float)
        sigma_array = np.maximum(np.asarray(sigma, dtype=float), 0.0)
        if axis_array.size == 0 or axis_array.size != sigma_array.size:
            return tuple()
        filename_metadata = _parse_xsc_filename_metadata(path)
        gas = str(filename_metadata.get("gas") or path.stem.split("_", 1)[0])
        return (
            CrossSectionSegment(
                gas=gas,
                path=path,
                nu_min=float(axis_array[0]),
                nu_max=float(axis_array[-1]),
                temperature_k=filename_metadata.get("temperature_k") if isinstance(filename_metadata.get("temperature_k"), float) else None,
                pressure_torr=filename_metadata.get("pressure_torr") if isinstance(filename_metadata.get("pressure_torr"), float) else None,
                common_name=gas,
                broadener="",
                reference=str(filename_metadata.get("reference") or ""),
                resolution=None,
                axis_cm1=axis_array,
                sigma_cm2_per_molecule=sigma_array,
            ),
        )

    line_index = 0
    while line_index < len(raw_lines):
        header_line = raw_lines[line_index]
        if not header_line.strip():
            line_index += 1
            continue
        if not _is_xsc_header_line(header_line):
            line_index += 1
            continue

        gas = header_line[:20].strip()
        nu_min = float(header_line[20:30].strip().replace("D", "E"))
        nu_max = float(header_line[30:40].strip().replace("D", "E"))
        point_count = int(header_line[40:47].strip())
        temperature_k = _parse_xsc_optional_float(header_line[47:54])
        pressure_torr = _parse_xsc_optional_float(header_line[54:60])
        resolution = _parse_xsc_optional_float(header_line[70:75])
        common_name = header_line[75:90].strip()
        broadener = header_line[94:97].strip()
        reference = header_line[97:100].strip()

        values: list[float] = []
        line_index += 1
        while line_index < len(raw_lines) and len(values) < point_count:
            value_line = raw_lines[line_index]
            if _is_xsc_header_line(value_line):
                break
            for offset in range(0, len(value_line), 10):
                field = value_line[offset:offset + 10].strip().replace("D", "E")
                if not field:
                    continue
                try:
                    values.append(max(float(field), 0.0))
                except ValueError:
                    continue
                if len(values) >= point_count:
                    break
            line_index += 1

        if len(values) < point_count:
            raise ValueError(f"Local XSC file {path.name} ended before {point_count} points were read.")

        segments.append(
            CrossSectionSegment(
                gas=gas,
                path=path,
                nu_min=nu_min,
                nu_max=nu_max,
                temperature_k=temperature_k,
                pressure_torr=pressure_torr,
                common_name=common_name,
                broadener=broadener,
                reference=reference,
                resolution=resolution,
                axis_cm1=np.linspace(nu_min, nu_max, point_count, dtype=float),
                sigma_cm2_per_molecule=np.asarray(values[:point_count], dtype=float),
            )
        )

    return tuple(segments)


def _local_xsc_segments_for_gas(gas: str) -> tuple[CrossSectionSegment, ...]:
    expected_tokens = _gas_lookup_tokens(gas)
    if not expected_tokens:
        return tuple()
    segments: list[CrossSectionSegment] = []
    load_errors: list[str] = []
    for path in _iter_local_xsc_paths(gas):
        filename_token = _gas_lookup_token(path.stem.split("_", 1)[0])
        try:
            stat = path.stat()
            parsed = _load_local_xsc_segments(str(path), stat.st_mtime_ns, stat.st_size)
        except Exception as exc:
            load_errors.append(f"{path.name}: {exc}")
            continue
        for segment in parsed:
            segment_token = _gas_lookup_token(segment.gas)
            if segment_token in expected_tokens or filename_token in expected_tokens:
                segments.append(segment)
    if segments:
        return tuple(segments)
    if load_errors:
        sample_errors = "; ".join(load_errors[:2])
        if len(load_errors) > 2:
            sample_errors += f"; +{len(load_errors) - 2} weitere"
        raise ValueError(
            f"Local HITRAN XSC files were found for {gas}, but none could be parsed. {sample_errors}"
        )
    return tuple(segments)


def _segment_overlap_cm1(segment: CrossSectionSegment, nu_min: float, nu_max: float) -> float:
    return max(0.0, min(segment.nu_max, nu_max) - max(segment.nu_min, nu_min))


def _merged_interval_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    current_start, current_end = ordered[0]
    total = 0.0
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    total += current_end - current_start
    return total


def _normalize_intervals(intervals: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    cleaned = sorted(
        (float(min(start, end)), float(max(start, end)))
        for start, end in intervals
        if max(start, end) > min(start, end)
    )
    if not cleaned:
        return tuple()
    merged: list[tuple[float, float]] = []
    current_start, current_end = cleaned[0]
    for start, end in cleaned[1:]:
        if start <= current_end + 1.0e-9:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return tuple(merged)


def _clip_intervals(
    intervals: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    lower_bound: float,
    upper_bound: float,
) -> tuple[tuple[float, float], ...]:
    clipped = [
        (max(float(start), lower_bound), min(float(end), upper_bound))
        for start, end in intervals
        if min(float(end), upper_bound) > max(float(start), lower_bound)
    ]
    return _normalize_intervals(clipped)


def _complement_intervals(
    covered_intervals: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    lower_bound: float,
    upper_bound: float,
) -> tuple[tuple[float, float], ...]:
    normalized = _clip_intervals(covered_intervals, lower_bound, upper_bound)
    if lower_bound >= upper_bound:
        return tuple()
    gaps: list[tuple[float, float]] = []
    cursor = lower_bound
    for start, end in normalized:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < upper_bound:
        gaps.append((cursor, upper_bound))
    return _normalize_intervals(gaps)


def _axis_mask_from_intervals(axis: np.ndarray, intervals: tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros_like(axis, dtype=bool)
    for start, end in intervals:
        mask |= (axis >= start) & (axis <= end)
    return mask


def _interpolate_with_zero_outside(
    target_axis: np.ndarray,
    source_axis: np.ndarray,
    source_sigma: np.ndarray,
) -> np.ndarray:
    interpolated = np.zeros_like(target_axis, dtype=float)
    in_bounds = (target_axis >= float(source_axis[0])) & (target_axis <= float(source_axis[-1]))
    if np.any(in_bounds):
        interpolated[in_bounds] = np.interp(target_axis[in_bounds], source_axis, source_sigma)
    return interpolated


def _select_local_xsc_segments(
    gas: str,
    temperature_c: float,
    pressure_hpa: float,
    nu_min: float,
    nu_max: float,
) -> tuple[CrossSectionSegment, ...]:
    segments = _local_xsc_segments_for_gas(gas)
    if not segments:
        raise FileNotFoundError(
            f"No local HITRAN XSC files were found for {gas}. Place .xsc files in {XSC_CACHE_DIR} or {DATA_DIR}."
        )

    grouped: dict[tuple[Any, ...], list[CrossSectionSegment]] = {}
    for segment in segments:
        key = (
            round(segment.temperature_k, 3) if segment.temperature_k is not None else None,
            round(segment.pressure_torr, 3) if segment.pressure_torr is not None else None,
            segment.common_name,
            segment.broadener,
            segment.reference,
        )
        grouped.setdefault(key, []).append(segment)

    requested_temperature_k = temperature_c + 273.15
    requested_pressure_torr = pressure_hpa * TORR_PER_HPA
    best_segments: tuple[CrossSectionSegment, ...] | None = None
    best_score: tuple[float, float, float] | None = None
    available_ranges: list[str] = []

    for group_segments in grouped.values():
        overlaps = [
            (max(segment.nu_min, nu_min), min(segment.nu_max, nu_max))
            for segment in group_segments
            if _segment_overlap_cm1(segment, nu_min, nu_max) > 0.0
        ]
        available_ranges.extend(f"{segment.nu_min:.2f}-{segment.nu_max:.2f} cm-1" for segment in group_segments)
        total_overlap = _merged_interval_length(overlaps)
        if total_overlap <= 0.0:
            continue

        group_temperature_k = next((segment.temperature_k for segment in group_segments if segment.temperature_k is not None), None)
        group_pressure_torr = next((segment.pressure_torr for segment in group_segments if segment.pressure_torr is not None), None)
        temp_penalty = abs(group_temperature_k - requested_temperature_k) if group_temperature_k is not None else 9999.0
        pressure_penalty = abs(group_pressure_torr - requested_pressure_torr) if group_pressure_torr is not None else 9999.0
        score = (total_overlap, -temp_penalty, -pressure_penalty)
        if best_score is None or score > best_score:
            best_score = score
            best_segments = tuple(sorted(group_segments, key=lambda segment: (segment.nu_min, segment.nu_max)))

    if best_segments is None:
        unique_ranges = sorted(set(available_ranges))
        range_preview = ", ".join(unique_ranges[:8])
        if len(unique_ranges) > 8:
            range_preview += f", +{len(unique_ranges) - 8} weitere"
        raise ValueError(
            f"Local HITRAN XSC files for {gas} do not overlap the requested range {nu_min:.2f}-{nu_max:.2f} cm-1. "
            + (f"Available XSC ranges: {range_preview}." if range_preview else "")
        )

    return best_segments


def _coverage_ranges_for_gas_request(
    gas: str,
    temperature_c: float,
    pressure_hpa: float,
    nu_min: float,
    nu_max: float,
) -> tuple[tuple[float, float], ...]:
    coverage_sources: list[tuple[float, float]] = []

    cached_table_range = _local_table_range(gas)
    if cached_table_range is not None:
        coverage_sources.append(cached_table_range)

    try:
        segments = _select_local_xsc_segments(gas, temperature_c, pressure_hpa, nu_min, nu_max)
    except Exception:
        segments = tuple()

    coverage_sources.extend(
        (max(segment.nu_min, nu_min), min(segment.nu_max, nu_max))
        for segment in segments
        if _segment_overlap_cm1(segment, nu_min, nu_max) > 0.0
    )

    return _clip_intervals(coverage_sources, nu_min, nu_max)


def _local_xsc_can_build(gas: str, temperature_c: float, pressure_hpa: float, nu_min: float, nu_max: float) -> bool:
    try:
        _select_local_xsc_segments(gas, temperature_c, pressure_hpa, nu_min, nu_max)
        return True
    except Exception:
        return False


def _sigma_from_local_xsc(
    gas: str,
    temperature_c: float,
    pressure_hpa: float,
    nu_min: float,
    nu_max: float,
    step_cm1: float,
) -> tuple[np.ndarray, np.ndarray]:
    segments = _select_local_xsc_segments(gas, temperature_c, pressure_hpa, nu_min, nu_max)
    axis = np.arange(nu_min, nu_max + (step_cm1 * 0.5), step_cm1, dtype=float)
    if axis.size < 2:
        raise ValueError("XSC calculation requires at least two spectral points.")

    sigma = np.zeros_like(axis)
    for segment in segments:
        overlap_min = max(segment.nu_min, nu_min)
        overlap_max = min(segment.nu_max, nu_max)
        if overlap_min > overlap_max:
            continue
        mask = (axis >= overlap_min) & (axis <= overlap_max)
        if not np.any(mask):
            continue
        sigma[mask] = np.maximum(
            sigma[mask],
            np.interp(axis[mask], segment.axis_cm1, segment.sigma_cm2_per_molecule),
        )
    return axis, sigma


def _local_cache_covers_range(gas: str, nu_min: float, nu_max: float) -> bool:
    cached = _local_table_range(gas)
    if cached is None:
        return False

    wanted_min = nu_min - FETCH_MARGIN_CM1
    wanted_max = nu_max + FETCH_MARGIN_CM1
    return cached[0] <= wanted_min and cached[1] >= wanted_max


def _xsc_source_details_for_gas(
    gas: str,
    temperature_c: float,
    pressure_hpa: float,
    nu_min: float,
    nu_max: float,
) -> dict[str, Any] | None:
    try:
        segments = _select_local_xsc_segments(gas, temperature_c, pressure_hpa, nu_min, nu_max)
    except Exception:
        return None

    if not segments:
        return None

    temperatures_k = sorted({round(float(segment.temperature_k), 3) for segment in segments if segment.temperature_k is not None})
    pressures_torr = sorted({round(float(segment.pressure_torr), 3) for segment in segments if segment.pressure_torr is not None})
    file_names = sorted({segment.path.name for segment in segments})
    coverage_ranges = _clip_intervals(
        [(segment.nu_min, segment.nu_max) for segment in segments],
        nu_min,
        nu_max,
    )
    return {
        "source": "xsc",
        "temperature_k": temperatures_k[0] if len(temperatures_k) == 1 else None,
        "pressure_torr": pressures_torr[0] if len(pressures_torr) == 1 else None,
        "temperature_k_values": temperatures_k,
        "pressure_torr_values": pressures_torr,
        "files": file_names,
        "coverage_ranges_cm1": [[float(start), float(end)] for start, end in coverage_ranges],
    }


def _line_source_details_for_gas(gas: str, nu_min: float, nu_max: float) -> dict[str, Any] | None:
    cached = _local_table_range(gas)
    if cached is None:
        return None
    clipped = _clip_intervals([cached], nu_min, nu_max)
    if not clipped:
        return None
    return {
        "source": "line_cache",
        "coverage_ranges_cm1": [[float(start), float(end)] for start, end in clipped],
        "table_range_cm1": [float(cached[0]), float(cached[1])],
    }


def _offline_missing_cache_details(missing_gases: list[str], nu_min: float, nu_max: float) -> str:
    missing_local: list[str] = []
    partial_local: list[str] = []
    wanted_min = nu_min - FETCH_MARGIN_CM1
    wanted_max = nu_max + FETCH_MARGIN_CM1

    for gas in missing_gases:
        cached = _local_table_range(gas)
        if cached is None:
            missing_local.append(gas)
            continue
        if cached[0] > wanted_min or cached[1] < wanted_max:
            partial_local.append(f"{gas} ({cached[0]:.2f}-{cached[1]:.2f} cm-1)")

    details: list[str] = []
    if missing_local:
        details.append("Local HITRAN cache is also missing for: " + ", ".join(missing_local) + ".")
    if partial_local:
        details.append("Local HITRAN cache coverage is too small for: " + ", ".join(partial_local) + ".")
    missing_xsc = [gas for gas in missing_gases if not _local_xsc_can_build(gas, PICKLE_REBUILD_TEMPERATURE_C, PICKLE_REBUILD_PRESSURE_HPA, nu_min, nu_max)]
    if missing_xsc:
        details.append("Local HITRAN XSC files are also missing for: " + ", ".join(missing_xsc) + ".")
    if details:
        details.append("The last manual refresh likely returned no line data for these gases/ranges.")
    return " ".join(details)


def _repair_offline_library_from_local_cache(
    missing_gases: list[str],
    nu_min: float,
    nu_max: float,
) -> None:
    metadata: dict[str, Any] = {}
    if OFFLINE_SPECTRA_PATH.exists():
        try:
            metadata = offline_library_metadata()
        except Exception:
            metadata = {}

    repair_temperature_c = float(metadata.get("reference_temperature_c", PICKLE_REBUILD_TEMPERATURE_C))
    repair_pressure_hpa = float(metadata.get("reference_pressure_hpa", PICKLE_REBUILD_PRESSURE_HPA))
    rebuildable_gases = [
        gas
        for gas in missing_gases
        if _local_cache_covers_range(gas, nu_min, nu_max)
        or _local_xsc_can_build(gas, repair_temperature_c, repair_pressure_hpa, nu_min, nu_max)
    ]
    if not rebuildable_gases:
        return

    rebuild_offline_pickle_from_hitran(
        gases=rebuildable_gases,
        range_unit="cm-1",
        range_min=nu_min,
        range_max=nu_max,
        step_cm1=float(metadata.get("native_step_cm1", DEFAULT_MANUAL_STEP_CM1)) or DEFAULT_MANUAL_STEP_CM1,
        temperature_c=repair_temperature_c,
        pressure_hpa=repair_pressure_hpa,
        merge_with_existing=True,
    )


@lru_cache(maxsize=LOCAL_CACHE_MAX_GASES)
def _local_table_range(gas: str) -> tuple[float, float] | None:
    if not _has_local_table_cache(gas):
        return None

    header_path = DATA_DIR / f"{gas}.header"
    data_path = DATA_DIR / f"{gas}.data"
    with header_path.open("r", encoding="utf-8") as header_file:
        header = json.load(header_file)

    positions = header.get("position", {})
    nu_start = int(positions["nu"])
    sorted_positions = sorted((int(value), key) for key, value in positions.items())
    nu_end = None
    for index, (position, key) in enumerate(sorted_positions):
        if key != "nu":
            continue
        if index + 1 < len(sorted_positions):
            nu_end = sorted_positions[index + 1][0]
        break

    def parse_nu(line: str) -> float:
        field = line[nu_start:nu_end].strip() if nu_end is not None else line[nu_start:].strip()
        return float(field)

    first_line = ""
    last_line = ""
    with data_path.open("r", encoding="utf-8", errors="ignore") as data_file:
        for line in data_file:
            stripped = line.rstrip("\n\r")
            if not stripped:
                continue
            if not first_line:
                first_line = stripped
            last_line = stripped

    if not first_line or not last_line:
        return None

    first_nu = parse_nu(first_line)
    last_nu = parse_nu(last_line)
    return min(first_nu, last_nu), max(first_nu, last_nu)


def _ensure_species_available(gas: str, nu_min: float, nu_max: float) -> None:
    wanted_min = nu_min - FETCH_MARGIN_CM1
    wanted_max = nu_max + FETCH_MARGIN_CM1
    cached = _FETCHED_RANGES.get(gas)
    if cached is None:
        cached = _local_table_range(gas)
        if cached is not None:
            _FETCHED_RANGES[gas] = cached

    if cached and cached[0] <= wanted_min and cached[1] >= wanted_max:
        return

    gas_config = GAS_LIBRARY[gas]
    fetch_min = wanted_min
    fetch_max = wanted_max
    if cached:
        fetch_min = min(fetch_min, cached[0])
        fetch_max = max(fetch_max, cached[1])
        reset_hitran_tables((gas,))

    hp.fetch(
        gas,
        gas_config["molecule_id"],
        gas_config["isotope_id"],
        fetch_min,
        fetch_max,
    )
    _local_table_range.cache_clear()
    _FETCHED_RANGES[gas] = (fetch_min, fetch_max)


def _is_hapi_parse_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "parse error" in message or "unknown format of the par value" in message


def _is_no_line_data_error(exc: Exception) -> bool:
    message = str(exc)
    return "Failed to retrieve data for given parameters" in message or "keine HITRAN-Liniendaten" in message


def _sigma_from_hitran_lines(
    gas: str,
    temperature_c: float,
    pressure_hpa: float,
    nu_min: float,
    nu_max: float,
    step_cm1: float,
) -> tuple[np.ndarray, np.ndarray]:
    _ensure_database_started()
    _ensure_species_available(gas, nu_min, nu_max)
    temperature_k = temperature_c + 273.15
    pressure_atm = pressure_hpa / 1013.25
    try:
        current_nu, current_sigma = hp.absorptionCoefficient_Voigt(
            SourceTables=gas,
            HITRAN_units=True,
            Environment={"p": pressure_atm, "T": temperature_k},
            IntensityThreshold=0,
            OmegaRange=(nu_min, nu_max),
            GammaL="gamma_air",
            OmegaStep=step_cm1,
        )
    except Exception as exc:
        if not _is_hapi_parse_error(exc):
            raise

        reset_hitran_tables()
        _FETCHED_RANGES.clear()
        _clear_runtime_caches()
        _ensure_database_started()
        _ensure_species_available(gas, nu_min, nu_max)
        current_nu, current_sigma = hp.absorptionCoefficient_Voigt(
            SourceTables=gas,
            HITRAN_units=True,
            Environment={"p": pressure_atm, "T": temperature_k},
            IntensityThreshold=0,
            OmegaRange=(nu_min, nu_max),
            GammaL="gamma_air",
            OmegaStep=step_cm1,
        )
    return np.asarray(current_nu, dtype=float), np.asarray(current_sigma, dtype=float)


def _sigma_from_local_db(
    gas: str,
    temperature_c: float,
    pressure_hpa: float,
    nu_min: float,
    nu_max: float,
    step_cm1: float,
) -> tuple[np.ndarray, np.ndarray]:
    if _local_xsc_can_build(gas, temperature_c, pressure_hpa, nu_min, nu_max):
        return _sigma_from_local_xsc(
            gas=gas,
            temperature_c=temperature_c,
            pressure_hpa=pressure_hpa,
            nu_min=nu_min,
            nu_max=nu_max,
            step_cm1=step_cm1,
        )
    try:
        return _sigma_from_hitran_lines(
            gas=gas,
            temperature_c=temperature_c,
            pressure_hpa=pressure_hpa,
            nu_min=nu_min,
            nu_max=nu_max,
            step_cm1=step_cm1,
        )
    except Exception as line_exc:
        try:
            return _sigma_from_local_xsc(
                gas=gas,
                temperature_c=temperature_c,
                pressure_hpa=pressure_hpa,
                nu_min=nu_min,
                nu_max=nu_max,
                step_cm1=step_cm1,
            )
        except Exception as xsc_exc:
            if _is_no_line_data_error(line_exc):
                raise ValueError(f"{line_exc}. Local XSC fallback also unavailable: {xsc_exc}") from line_exc
            raise line_exc from xsc_exc


def _write_offline_sigma_library(
    library: dict[str, tuple[np.ndarray, np.ndarray]],
    metadata: dict[str, Any] | None = None,
    coverage_ranges: dict[str, tuple[tuple[float, float], ...]] | None = None,
) -> None:
    serializable_library = {
        gas: (np.asarray(axis, dtype=float), np.asarray(sigma, dtype=float))
        for gas, (axis, sigma) in library.items()
    }
    payload: dict[str, Any] = {
        "library": serializable_library,
        "metadata": {
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "reference_temperature_c": PICKLE_REBUILD_TEMPERATURE_C,
            "reference_pressure_hpa": PICKLE_REBUILD_PRESSURE_HPA,
            "native_step_cm1": float(metadata.get("native_step_cm1")) if metadata and metadata.get("native_step_cm1") is not None else None,
        },
    }
    if metadata:
        payload["metadata"].update(metadata)
    if coverage_ranges is not None:
        payload["coverage_ranges"] = {
            gas: [[float(start), float(end)] for start, end in _normalize_intervals(ranges)]
            for gas, ranges in coverage_ranges.items()
        }
    with OFFLINE_SPECTRA_PATH.open("wb") as file_handle:
        pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    _clear_runtime_caches()


def cleanup_unused_files() -> list[str]:
    removed: list[str] = []
    for path in sorted(BASE_DIR.glob("*_hotw.txt")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    for path in sorted(BASE_DIR.glob("*_ME.txt")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    if LEGACY_DATA_DIR.exists():
        shutil.rmtree(LEGACY_DATA_DIR)
        removed.append(LEGACY_DATA_DIR.name)
    return removed


def reset_hitran_tables(gases: tuple[str, ...] | None = None) -> list[str]:
    selected = tuple(sorted(gases or GAS_LIBRARY.keys()))
    removed: list[str] = []
    for gas in selected:
        for suffix in (".data", ".header"):
            table_path = DATA_DIR / f"{gas}{suffix}"
            if table_path.exists():
                table_path.unlink()
                removed.append(table_path.name)
        _FETCHED_RANGES.pop(gas, None)
    _local_table_range.cache_clear()
    return removed


def _replace_hitran_table_from_temp(temp_table: str, gas: str) -> list[str]:
    replaced: list[str] = []
    for suffix in (".data", ".header"):
        target_path = DATA_DIR / f"{gas}{suffix}"
        temp_path = DATA_DIR / f"{temp_table}{suffix}"
        if target_path.exists():
            target_path.unlink()
            replaced.append(target_path.name)
        if temp_path.exists():
            temp_path.replace(target_path)
    _local_table_range.cache_clear()
    return replaced


def delete_offline_pickle() -> bool:
    if OFFLINE_SPECTRA_PATH.exists():
        OFFLINE_SPECTRA_PATH.unlink()
        _clear_runtime_caches()
        return True
    return False


def rebuild_offline_pickle_from_hitran(
    gases: list[str] | tuple[str, ...] | None = None,
    *,
    range_unit: str = "cm-1",
    range_min: float | None = None,
    range_max: float | None = None,
    step_cm1: float | None = None,
    temperature_c: float = PICKLE_REBUILD_TEMPERATURE_C,
    pressure_hpa: float = PICKLE_REBUILD_PRESSURE_HPA,
    merge_with_existing: bool = True,
) -> str:
    selected_gases = tuple(sorted({gas for gas in (gases or GAS_LIBRARY.keys()) if gas in GAS_LIBRARY}))
    if not selected_gases:
        raise ValueError("At least one supported gas is required for pickle generation.")

    if range_min is None or range_max is None:
        raise ValueError("A spectral minimum and maximum are required for pickle generation.")

    nu_min, nu_max = normalize_wavenumber_window(range_unit, range_min, range_max)

    if merge_with_existing and OFFLINE_SPECTRA_PATH.exists():
        library = {
            gas: (axis.copy(), sigma.copy())
            for gas, (axis, sigma) in _load_offline_sigma_library().items()
        }
        coverage_ranges = dict(_load_offline_coverage_ranges())
        updated_gases: list[str] = []
        for gas in selected_gases:
            if gas in library:
                axis, sigma = library[gas]
                local_step = float(np.median(np.diff(axis))) if axis.size > 1 else float(step_cm1 or DEFAULT_MANUAL_STEP_CM1)
                union_min = min(float(axis[0]), nu_min)
                union_max = max(float(axis[-1]), nu_max)
                union_axis = np.arange(union_min, union_max + (local_step * 0.5), local_step, dtype=float)
                union_sigma = _interpolate_with_zero_outside(union_axis, axis, sigma)
            else:
                local_step = float(step_cm1 or DEFAULT_MANUAL_STEP_CM1)
                union_axis = np.arange(nu_min, nu_max + (local_step * 0.5), local_step, dtype=float)
                union_sigma = np.zeros_like(union_axis)

            current_nu, current_sigma = _sigma_from_local_db(
                gas=gas,
                temperature_c=temperature_c,
                pressure_hpa=pressure_hpa,
                nu_min=nu_min,
                nu_max=nu_max,
                step_cm1=local_step,
            )
            current_ranges = _coverage_ranges_for_gas_request(gas, temperature_c, pressure_hpa, nu_min, nu_max)
            replace_mask = _axis_mask_from_intervals(union_axis, current_ranges)
            if np.any(replace_mask):
                union_sigma[replace_mask] = np.interp(union_axis[replace_mask], current_nu, current_sigma)
            library[gas] = (union_axis, union_sigma)
            coverage_ranges[gas] = _normalize_intervals(list(coverage_ranges.get(gas, tuple())) + list(current_ranges))
            updated_gases.append(gas)

        _write_offline_sigma_library(
            library,
            metadata={
                "native_step_cm1": float(step_cm1 or DEFAULT_MANUAL_STEP_CM1),
                "reference_temperature_c": temperature_c,
                "reference_pressure_hpa": pressure_hpa,
            },
            coverage_ranges=coverage_ranges,
        )
        return (
            f"Offline-PKL fuer {', '.join(updated_gases)} im Bereich {nu_min:.2f}-{nu_max:.2f} cm-1 aktualisiert "
            f"(Referenz: {temperature_c:.1f} °C, {pressure_hpa:.1f} hPa)."
        )

    effective_step = float(step_cm1 or DEFAULT_MANUAL_STEP_CM1)
    common_axis = np.arange(nu_min, nu_max + (effective_step * 0.5), effective_step, dtype=float)
    if common_axis.size < 2:
        raise ValueError("Pickle generation requires at least two spectral points.")

    library: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    coverage_ranges: dict[str, tuple[tuple[float, float], ...]] = {}
    for gas in selected_gases:
        current_nu, current_sigma = _sigma_from_local_db(
            gas=gas,
            temperature_c=temperature_c,
            pressure_hpa=pressure_hpa,
            nu_min=nu_min,
            nu_max=nu_max,
            step_cm1=effective_step,
        )
        gas_sigma = np.zeros_like(common_axis)
        current_ranges = _coverage_ranges_for_gas_request(gas, temperature_c, pressure_hpa, nu_min, nu_max)
        replace_mask = _axis_mask_from_intervals(common_axis, current_ranges)
        if np.any(replace_mask):
            gas_sigma[replace_mask] = np.interp(common_axis[replace_mask], current_nu, current_sigma)
        library[gas] = (common_axis, gas_sigma)
        coverage_ranges[gas] = current_ranges

    _write_offline_sigma_library(
        library,
        metadata={
            "native_step_cm1": effective_step,
            "reference_temperature_c": temperature_c,
            "reference_pressure_hpa": pressure_hpa,
        },
        coverage_ranges=coverage_ranges,
    )
    return (
        f"Offline-PKL neu erzeugt fuer {', '.join(selected_gases)} im Bereich {nu_min:.2f}-{nu_max:.2f} cm-1 "
        f"mit {effective_step:.6f} cm-1 Raster."
    )


@lru_cache(maxsize=1)
def _load_offline_sigma_library() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not OFFLINE_SPECTRA_PATH.exists():
        raise FileNotFoundError(
            f"Offline spectra file not found: {OFFLINE_SPECTRA_PATH.name}."
        )

    with OFFLINE_SPECTRA_PATH.open("rb") as file_handle:
        raw_library = pickle.load(file_handle)

    if not isinstance(raw_library, dict) or not raw_library:
        raise ValueError("Offline spectra file is empty or has an unsupported format.")

    if "library" in raw_library and isinstance(raw_library.get("library"), dict):
        raw_entries = raw_library["library"]
    else:
        raw_entries = raw_library

    library: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gas, payload in raw_entries.items():
        if not isinstance(payload, (tuple, list)) or len(payload) < 2:
            raise ValueError(f"Offline spectra entry for {gas} has an unsupported format.")

        axis = np.asarray(payload[0], dtype=float)
        sigma = np.asarray(payload[1], dtype=float)
        if axis.ndim != 1 or sigma.ndim != 1 or axis.size == 0 or axis.size != sigma.size:
            raise ValueError(f"Offline spectra entry for {gas} is malformed.")

        if axis.size > 1 and axis[0] > axis[-1]:
            axis = axis[::-1]
            sigma = sigma[::-1]

        library[str(gas)] = (axis, sigma)

    return library


@lru_cache(maxsize=1)
def _load_offline_coverage_ranges() -> dict[str, tuple[tuple[float, float], ...]]:
    if not OFFLINE_SPECTRA_PATH.exists():
        raise FileNotFoundError(
            f"Offline spectra file not found: {OFFLINE_SPECTRA_PATH.name}."
        )

    with OFFLINE_SPECTRA_PATH.open("rb") as file_handle:
        raw_library = pickle.load(file_handle)

    coverage_payload = raw_library.get("coverage_ranges", {}) if isinstance(raw_library, dict) else {}
    coverage_ranges: dict[str, tuple[tuple[float, float], ...]] = {}
    if isinstance(coverage_payload, dict):
        for gas, raw_ranges in coverage_payload.items():
            parsed_ranges: list[tuple[float, float]] = []
            if isinstance(raw_ranges, (list, tuple)):
                for raw_range in raw_ranges:
                    if not isinstance(raw_range, (list, tuple)) or len(raw_range) < 2:
                        continue
                    parsed_ranges.append((float(raw_range[0]), float(raw_range[1])))
            coverage_ranges[str(gas)] = _normalize_intervals(parsed_ranges)

    if coverage_ranges:
        return coverage_ranges

    library = _load_offline_sigma_library()
    return {
        gas: ((float(axis[0]), float(axis[-1])),)
        for gas, (axis, _sigma) in library.items()
    }


@lru_cache(maxsize=1)
def offline_library_metadata() -> dict[str, Any]:
    if not OFFLINE_SPECTRA_PATH.exists():
        raise FileNotFoundError(
            f"Offline spectra file not found: {OFFLINE_SPECTRA_PATH.name}."
        )

    with OFFLINE_SPECTRA_PATH.open("rb") as file_handle:
        raw_library = pickle.load(file_handle)

    file_stat = OFFLINE_SPECTRA_PATH.stat()
    fallback_updated_at = datetime.fromtimestamp(file_stat.st_mtime).astimezone().isoformat(timespec="seconds")
    metadata = raw_library.get("metadata", {}) if isinstance(raw_library, dict) else {}
    return {
        "updated_at": metadata.get("updated_at", fallback_updated_at),
        "reference_temperature_c": float(metadata.get("reference_temperature_c", PICKLE_REBUILD_TEMPERATURE_C)),
        "reference_pressure_hpa": float(metadata.get("reference_pressure_hpa", PICKLE_REBUILD_PRESSURE_HPA)),
        "native_step_cm1": float(metadata.get("native_step_cm1", DEFAULT_MANUAL_STEP_CM1)),
    }


def offline_library_summary() -> dict[str, Any]:
    library = _load_offline_sigma_library()
    metadata = offline_library_metadata()
    gases = tuple(sorted(library.keys()))
    coverage_min = max(float(axis[0]) for axis, _ in library.values())
    coverage_max = min(float(axis[-1]) for axis, _ in library.values())
    reference_axis = library[gases[0]][0]
    native_step = float(np.median(np.diff(reference_axis))) if reference_axis.size > 1 else 0.0
    return {
        "gases": gases,
        "coverage_min_cm1": coverage_min,
        "coverage_max_cm1": coverage_max,
        "native_step_cm1": float(metadata.get("native_step_cm1", native_step or 0.0)) or native_step,
        "path": str(OFFLINE_SPECTRA_PATH),
        "updated_at": metadata.get("updated_at"),
        "reference_temperature_c": metadata.get("reference_temperature_c"),
        "reference_pressure_hpa": metadata.get("reference_pressure_hpa"),
    }


@lru_cache(maxsize=64)
def _cached_offline_sigma_bundle(
    gases: tuple[str, ...],
    nu_min: float,
    nu_max: float,
    step_cm1: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    library = _load_offline_sigma_library()
    missing_gases = [gas for gas in gases if gas not in library]
    if missing_gases:
        _repair_offline_library_from_local_cache(missing_gases, nu_min, nu_max)
        library = _load_offline_sigma_library()
    coverage_ranges = _load_offline_coverage_ranges()
    reference_axis = next((library[gas][0] for gas in gases if gas in library), None)
    if reference_axis is None:
        target_axis = np.arange(nu_min, nu_max + (float(step_cm1) * 0.5), float(step_cm1), dtype=float)
        return target_axis, {gas: np.zeros_like(target_axis) for gas in gases}
    native_step = float(np.median(np.diff(reference_axis))) if reference_axis.size > 1 else step_cm1
    target_step = max(float(step_cm1), native_step)
    point_count = max(2, int(np.floor((nu_max - nu_min) / target_step)) + 1)
    target_axis = np.linspace(nu_min, nu_max, point_count, dtype=float)

    sigma_map: dict[str, np.ndarray] = {}
    for gas in gases:
        if gas not in library:
            sigma_map[gas] = np.zeros_like(target_axis)
            continue
        gas_axis, gas_sigma = library[gas]
        gas_ranges = _clip_intervals(coverage_ranges.get(gas, ((float(gas_axis[0]), float(gas_axis[-1])),)), nu_min, nu_max)
        interpolated = _interpolate_with_zero_outside(target_axis, gas_axis, gas_sigma)
        if gas_ranges:
            interpolated[~_axis_mask_from_intervals(target_axis, gas_ranges)] = 0.0
        else:
            interpolated[:] = 0.0
        sigma_map[gas] = interpolated

    return target_axis, sigma_map


def refresh_hitran_database(
    gases: list[str] | tuple[str, ...],
    range_unit: str,
    range_min: float,
    range_max: float,
) -> str:
    selected_gases = tuple(sorted({gas for gas in gases if gas in GAS_LIBRARY}))
    if not selected_gases:
        raise ValueError("Select at least one supported gas before starting a HITRAN refresh.")

    nu_min, nu_max = normalize_wavenumber_window(range_unit, range_min, range_max)
    fetch_min = nu_min - FETCH_MARGIN_CM1
    fetch_max = nu_max + FETCH_MARGIN_CM1

    _ensure_database_started()
    replaced_tables: list[str] = []
    successful_line_gases: list[str] = []
    successful_xsc_gases: list[str] = []
    failed_gases: list[str] = []
    for gas in selected_gases:
        if _local_xsc_can_build(
            gas,
            PICKLE_REBUILD_TEMPERATURE_C,
            PICKLE_REBUILD_PRESSURE_HPA,
            nu_min,
            nu_max,
        ):
            successful_xsc_gases.append(gas)
            continue
        gas_config = GAS_LIBRARY[gas]
        temp_table = f"__tmp__{gas}"
        reset_hitran_tables((temp_table,))
        try:
            hp.fetch(
                temp_table,
                gas_config["molecule_id"],
                gas_config["isotope_id"],
                fetch_min,
                fetch_max,
            )
            replaced_tables.extend(_replace_hitran_table_from_temp(temp_table, gas))
            _FETCHED_RANGES[gas] = (fetch_min, fetch_max)
            successful_line_gases.append(gas)
        except Exception as exc:
            reset_hitran_tables((temp_table,))
            original_exc_text = str(exc)
            exc_text = original_exc_text
            if gas == "SF6" and "Failed to retrieve data for given parameters" in original_exc_text:
                exc_text = (
                    "aktueller HITRAN/HAPI-Refreshpfad liefert fuer SF6 keine Linienliste; "
                    "SF6 ist dort sehr wahrscheinlich nur als Cross-Section/XSC verfuegbar"
                )
            elif "Failed to retrieve data for given parameters" in original_exc_text:
                exc_text = (
                    "keine HITRAN-Liniendaten im angeforderten Bereich "
                    f"{nu_min:.2f}-{nu_max:.2f} cm-1"
                )
            failed_gases.append(f"{gas} ({exc_text})")

    successful_gases = successful_line_gases + successful_xsc_gases
    if not successful_gases:
        failure_preview = "; ".join(failed_gases[:8])
        raise ValueError(
            "Keine HITRAN-Daten fuer die ausgewaehlten Gase im angeforderten Bereich geladen. "
            + failure_preview
        )

    _clear_runtime_caches()
    removed_legacy_files = cleanup_unused_files()
    offline_message = rebuild_offline_pickle_from_hitran(
        gases=successful_gases,
        range_unit=range_unit,
        range_min=range_min,
        range_max=range_max,
        step_cm1=DEFAULT_MANUAL_STEP_CM1,
        temperature_c=PICKLE_REBUILD_TEMPERATURE_C,
        pressure_hpa=PICKLE_REBUILD_PRESSURE_HPA,
        merge_with_existing=True,
    )

    cleanup_bits: list[str] = []
    if replaced_tables:
        cleanup_bits.append("Ersetzte Tabellen: " + ", ".join(replaced_tables[:12]))
    if successful_xsc_gases:
        cleanup_bits.append("Lokale XSC genutzt: " + ", ".join(successful_xsc_gases[:12]))
    if removed_legacy_files:
        cleanup_bits.append("Altdateien: " + ", ".join(removed_legacy_files))
    cleanup_suffix = f" Aufgeraeumt ({'; '.join(cleanup_bits)})." if cleanup_bits else ""

    skipped_suffix = ""
    if failed_gases:
        skipped_preview = "; ".join(failed_gases[:8])
        remaining = len(failed_gases) - 8
        if remaining > 0:
            skipped_preview += f"; +{remaining} weitere"
        skipped_suffix = " Uebersprungen ohne Treffer/bei Fehler: " + skipped_preview + "."

    return (
        f"Lokale Spektralquellen fuer {', '.join(successful_gases)} aktualisiert "
        f"({fetch_min:.2f}-{fetch_max:.2f} cm-1 inkl. Rand). "
        "Die Live-Berechnung nutzt diese lokalen Quellen sofort; offline funktionieren danach genau diese lokal gecachten Bereiche auch ohne Download."
        f" {offline_message}{skipped_suffix}{cleanup_suffix}"
    )


@lru_cache(maxsize=LOCAL_CACHE_MAX_GASES)
def _cached_sigma_bundle(
    gases: tuple[str, ...],
    temperature_c: float,
    pressure_hpa: float,
    nu_min: float,
    nu_max: float,
    step_cm1: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sigma_map: dict[str, np.ndarray] = {}
    axis: np.ndarray | None = None
    for gas in gases:
        try:
            current_nu_array, current_sigma_array = _sigma_from_local_db(
                gas=gas,
                temperature_c=temperature_c,
                pressure_hpa=pressure_hpa,
                nu_min=nu_min,
                nu_max=nu_max,
                step_cm1=step_cm1,
            )
        except Exception:
            current_nu_array, offline_sigma_map = _cached_offline_sigma_bundle(
                (gas,),
                nu_min,
                nu_max,
                step_cm1,
            )
            current_sigma_array = offline_sigma_map[gas]

        if axis is None:
            axis = np.asarray(current_nu_array, dtype=float)
        if current_nu_array.shape != axis.shape or not np.allclose(current_nu_array, axis):
            sigma_map[gas] = np.interp(axis, current_nu_array, current_sigma_array)
        else:
            sigma_map[gas] = np.asarray(current_sigma_array, dtype=float)

    if axis is None:
        axis = np.arange(nu_min, nu_max + (float(step_cm1) * 0.5), float(step_cm1), dtype=float)
    return axis, sigma_map


def build_manual_spectrum(
    concentrations: dict[str, float],
    temperature_c: float,
    pressure_hpa: float,
    range_unit: str,
    range_min: float,
    range_max: float,
    step_cm1: float | None = None,
    data_source: str = LIVE_DB_MODE,
) -> ManualSpectrumResult:
    active_concentrations = {
        gas: value for gas, value in concentrations.items() if gas in GAS_LIBRARY and value > 0
    }
    if not active_concentrations:
        raise ValueError("At least one gas with a positive concentration is required.")

    nu_min, nu_max = normalize_wavenumber_window(range_unit, range_min, range_max)
    span_cm1 = nu_max - nu_min
    effective_step = step_cm1 or recommended_step_cm1(span_cm1, manual_mode=True)
    gas_tuple = tuple(sorted(active_concentrations.keys()))
    if data_source == OFFLINE_DB_MODE:
        metadata = offline_library_metadata()
        effective_step = float(metadata.get("native_step_cm1", DEFAULT_MANUAL_STEP_CM1))
        temperature_c = float(metadata.get("reference_temperature_c", PICKLE_REBUILD_TEMPERATURE_C))
        pressure_hpa = float(metadata.get("reference_pressure_hpa", PICKLE_REBUILD_PRESSURE_HPA))
        wavenumber_cm1, sigma_map = _cached_offline_sigma_bundle(
            gas_tuple,
            round(nu_min, 6),
            round(nu_max, 6),
            round(effective_step, 6),
        )
    else:
        wavenumber_cm1, sigma_map = _cached_sigma_bundle(
            gas_tuple,
            round(float(temperature_c), 4),
            round(float(pressure_hpa), 4),
            round(nu_min, 6),
            round(nu_max, 6),
            round(effective_step, 6),
        )
    if data_source == OFFLINE_DB_MODE:
        offline_coverage_ranges = _load_offline_coverage_ranges()
        coverage_ranges_cm1_by_gas = {
            gas: _clip_intervals(offline_coverage_ranges.get(gas, tuple()), float(wavenumber_cm1[0]), float(wavenumber_cm1[-1]))
            for gas in gas_tuple
        }
        source_details_by_gas = {
            gas: {
                "source": "offline_db",
                "reference_temperature_c": float(temperature_c),
                "reference_pressure_hpa": float(pressure_hpa),
                "coverage_ranges_cm1": [[float(start), float(end)] for start, end in coverage_ranges_cm1_by_gas.get(gas, tuple())],
            }
            for gas in gas_tuple
        }
    else:
        coverage_ranges_cm1_by_gas = {
            gas: _coverage_ranges_for_gas_request(gas, temperature_c, pressure_hpa, float(wavenumber_cm1[0]), float(wavenumber_cm1[-1]))
            for gas in gas_tuple
        }
        source_details_by_gas = {}
        request_nu_min = float(wavenumber_cm1[0])
        request_nu_max = float(wavenumber_cm1[-1])
        for gas in gas_tuple:
            xsc_details = _xsc_source_details_for_gas(gas, temperature_c, pressure_hpa, request_nu_min, request_nu_max)
            if xsc_details is not None:
                source_details_by_gas[gas] = xsc_details
                continue
            line_details = _line_source_details_for_gas(gas, request_nu_min, request_nu_max)
            if line_details is not None:
                source_details_by_gas[gas] = line_details
            else:
                source_details_by_gas[gas] = {
                    "source": "unavailable",
                    "coverage_ranges_cm1": [],
                }
    missing_ranges_cm1_by_gas = {
        gas: _complement_intervals(coverage_ranges_cm1_by_gas.get(gas, tuple()), float(wavenumber_cm1[0]), float(wavenumber_cm1[-1]))
        for gas in gas_tuple
    }
    wavelength_um = np.asarray(wavenumber_cm1_to_wavelength_um(wavenumber_cm1), dtype=float)
    number_density = total_number_density_cm3(temperature_c, pressure_hpa)
    total_sigma = np.zeros_like(wavenumber_cm1, dtype=float)
    total_alpha = np.zeros_like(wavenumber_cm1, dtype=float)
    components: dict[str, ComponentSpectrum] = {}

    for gas in gas_tuple:
        sigma = np.asarray(sigma_map[gas], dtype=float)
        alpha = sigma * number_density * active_concentrations[gas]
        total_sigma += sigma
        total_alpha += alpha
        peak_index = int(np.argmax(alpha))
        components[gas] = ComponentSpectrum(
            gas=gas,
            color=str(GAS_LIBRARY[gas].get("plot_color", GAS_LIBRARY[gas]["color"])),
            concentration=active_concentrations[gas],
            sigma_cm2_per_molecule=sigma,
            alpha_per_cm=alpha,
            peak_wavenumber_cm1=float(wavenumber_cm1[peak_index]),
            peak_wavelength_um=float(wavelength_um[peak_index]),
        )

    range_label = (
        f"{wavelength_um.min():.4f}-{wavelength_um.max():.4f} um"
        if range_unit == "um"
        else f"{nu_min:.2f}-{nu_max:.2f} cm-1"
    )
    return ManualSpectrumResult(
        wavenumber_cm1=np.asarray(wavenumber_cm1, dtype=float),
        wavelength_um=wavelength_um,
        total_sigma_cm2_per_molecule=total_sigma,
        total_alpha_per_cm=total_alpha,
        components=components,
        temperature_c=temperature_c,
        pressure_hpa=pressure_hpa,
        step_cm1=effective_step,
        range_label=range_label,
        coverage_ranges_cm1_by_gas=coverage_ranges_cm1_by_gas,
        missing_ranges_cm1_by_gas=missing_ranges_cm1_by_gas,
        source_details_by_gas=source_details_by_gas,
    )


def serialize_manual_result(result: ManualSpectrumResult) -> dict[str, Any]:
    return {
        "wavenumber_cm1": result.wavenumber_cm1.tolist(),
        "wavelength_um": result.wavelength_um.tolist(),
        "total_sigma_cm2_per_molecule": result.total_sigma_cm2_per_molecule.tolist(),
        "total_alpha_per_cm": result.total_alpha_per_cm.tolist(),
        "components": {
            gas: {
                "gas": component.gas,
                "color": component.color,
                "concentration": component.concentration,
                "sigma_cm2_per_molecule": component.sigma_cm2_per_molecule.tolist(),
                "alpha_per_cm": component.alpha_per_cm.tolist(),
                "peak_wavenumber_cm1": component.peak_wavenumber_cm1,
                "peak_wavelength_um": component.peak_wavelength_um,
            }
            for gas, component in result.components.items()
        },
        "temperature_c": result.temperature_c,
        "pressure_hpa": result.pressure_hpa,
        "step_cm1": result.step_cm1,
        "range_label": result.range_label,
        "coverage_ranges_cm1_by_gas": {
            gas: [[float(start), float(end)] for start, end in ranges]
            for gas, ranges in result.coverage_ranges_cm1_by_gas.items()
        },
        "missing_ranges_cm1_by_gas": {
            gas: [[float(start), float(end)] for start, end in ranges]
            for gas, ranges in result.missing_ranges_cm1_by_gas.items()
        },
        "source_details_by_gas": result.source_details_by_gas,
    }


def deserialize_manual_result(payload: dict[str, Any]) -> ManualSpectrumResult:
    return ManualSpectrumResult(
        wavenumber_cm1=np.asarray(payload["wavenumber_cm1"], dtype=float),
        wavelength_um=np.asarray(payload["wavelength_um"], dtype=float),
        total_sigma_cm2_per_molecule=np.asarray(payload["total_sigma_cm2_per_molecule"], dtype=float),
        total_alpha_per_cm=np.asarray(payload["total_alpha_per_cm"], dtype=float),
        components={
            gas: ComponentSpectrum(
                gas=component["gas"],
                color=component["color"],
                concentration=float(component["concentration"]),
                sigma_cm2_per_molecule=np.asarray(component["sigma_cm2_per_molecule"], dtype=float),
                alpha_per_cm=np.asarray(component["alpha_per_cm"], dtype=float),
                peak_wavenumber_cm1=float(component["peak_wavenumber_cm1"]),
                peak_wavelength_um=float(component["peak_wavelength_um"]),
            )
            for gas, component in payload["components"].items()
        },
        temperature_c=float(payload["temperature_c"]),
        pressure_hpa=float(payload["pressure_hpa"]),
        step_cm1=float(payload["step_cm1"]),
        range_label=payload["range_label"],
        coverage_ranges_cm1_by_gas={
            gas: tuple((float(start), float(end)) for start, end in ranges)
            for gas, ranges in payload.get("coverage_ranges_cm1_by_gas", {}).items()
        },
        missing_ranges_cm1_by_gas={
            gas: tuple((float(start), float(end)) for start, end in ranges)
            for gas, ranges in payload.get("missing_ranges_cm1_by_gas", {}).items()
        },
        source_details_by_gas={
            str(gas): details
            for gas, details in payload.get("source_details_by_gas", {}).items()
            if isinstance(details, dict)
        },
    )


def nearest_sample_index(result: ManualSpectrumResult, wavelength_um: float) -> int:
    return int(np.argmin(np.abs(result.wavelength_um - wavelength_um)))


def hover_payload(result: ManualSpectrumResult, wavelength_um: float) -> dict[str, Any]:
    sample_index = nearest_sample_index(result, wavelength_um)
    payload = {
        "wavelength_um": float(result.wavelength_um[sample_index]),
        "wavenumber_cm1": float(result.wavenumber_cm1[sample_index]),
        "total_sigma_cm2_per_molecule": float(result.total_sigma_cm2_per_molecule[sample_index]),
        "total_alpha_per_cm": float(result.total_alpha_per_cm[sample_index]),
        "components": {},
    }
    for gas, component in result.components.items():
        payload["components"][gas] = {
            "color": component.color,
            "sigma_cm2_per_molecule": float(component.sigma_cm2_per_molecule[sample_index]),
            "alpha_per_cm": float(component.alpha_per_cm[sample_index]),
            "concentration": component.concentration,
        }
    return payload


def _peak_candidates(
    signal: np.ndarray,
    competing_signal: np.ndarray,
    max_candidates: int,
) -> np.ndarray:
    if signal.size < 4 or np.max(signal) <= 0:
        return np.array([], dtype=int)
    prominence = max(np.max(signal) * 0.03, np.percentile(signal, 95) * 0.3)
    indices, _ = find_peaks(signal, prominence=prominence)
    if indices.size == 0:
        return np.array([int(np.argmax(signal))], dtype=int)
    peak_indices = indices.tolist()
    gas_peak = float(np.max(signal))
    strongest = sorted(peak_indices, key=lambda idx: signal[idx], reverse=True)[:max_candidates]
    balanced = sorted(
        peak_indices,
        key=lambda idx: (
            math.sqrt(max(signal[idx] / (gas_peak + 1.0e-30), 0.0))
            * math.log10(1.0 + min(signal[idx] / (competing_signal[idx] + 1.0e-30), 1.0e6)),
            signal[idx] / (gas_peak + 1.0e-30),
            signal[idx],
        ),
        reverse=True,
    )[:max_candidates]
    most_selective = sorted(
        peak_indices,
        key=lambda idx: (
            min(signal[idx] / (competing_signal[idx] + 1.0e-30), 1.0e6)
            * (0.35 + 0.65 * math.sqrt(max(signal[idx] / (gas_peak + 1.0e-30), 0.0))),
            signal[idx] / (gas_peak + 1.0e-30),
            signal[idx],
        ),
        reverse=True,
    )[:max_candidates]

    combined: list[int] = []
    for idx in strongest + balanced + most_selective:
        if idx not in combined:
            combined.append(idx)
    return np.asarray(combined, dtype=int)


def _peak_to_peak(profile: np.ndarray) -> float:
    if profile.size == 0:
        return 0.0
    return float(np.max(profile) - np.min(profile))


def _second_derivative_profile(profile: np.ndarray, axis: np.ndarray) -> np.ndarray:
    if profile.size < 3:
        return np.zeros_like(profile)
    first_derivative = np.gradient(profile, axis)
    return np.asarray(np.gradient(first_derivative, axis), dtype=float)


def _normalized_shape(profile: np.ndarray) -> np.ndarray:
    if profile.size == 0:
        return np.zeros_like(profile)
    centered = np.asarray(profile - float(np.mean(profile)), dtype=float)
    scale = float(np.max(np.abs(centered)))
    if scale <= 1.0e-30:
        return np.zeros_like(centered)
    return centered / scale


def _wms2f_shape_similarity(
    target_profile: np.ndarray,
    total_profile: np.ndarray,
    axis: np.ndarray,
) -> float:
    target_2f = _normalized_shape(_second_derivative_profile(target_profile, axis))
    total_2f = _normalized_shape(_second_derivative_profile(total_profile, axis))
    if target_2f.size < 2:
        return 1.0
    rmse = float(np.sqrt(np.mean((target_2f - total_2f) ** 2)))
    correlation = float(np.corrcoef(target_2f, total_2f)[0, 1])
    if not np.isfinite(correlation):
        correlation = 1.0
    rmse_similarity = max(0.0, 1.0 - (rmse / 2.0))
    corr_similarity = min(1.0, max(0.0, (correlation + 1.0) / 2.0))
    return float((rmse_similarity * 0.65) + (corr_similarity * 0.35))


def _peak_region_bounds(profile: np.ndarray, peak_index: int) -> tuple[int, int]:
    peak_value = float(profile[peak_index])
    if peak_value <= 0:
        return peak_index, peak_index
    threshold = peak_value * 0.55
    left = peak_index
    right = peak_index
    while left > 0 and float(profile[left - 1]) >= threshold:
        left -= 1
    while right + 1 < profile.size and float(profile[right + 1]) >= threshold:
        right += 1
    left = min(left, max(0, peak_index - 2))
    right = max(right, min(profile.size - 1, peak_index + 2))
    return left, right


def _peak_flank_baseline(profile: np.ndarray, left: int, right: int) -> float:
    flank_levels: list[float] = []
    left_slice = profile[max(0, left - 3):left]
    right_slice = profile[right + 1:min(profile.size, right + 4)]
    if left_slice.size:
        flank_levels.append(float(np.mean(left_slice)))
    if right_slice.size:
        flank_levels.append(float(np.mean(right_slice)))
    if flank_levels:
        return float(sum(flank_levels) / len(flank_levels))
    return float(np.min(profile))


def _window_metric_acceptance(metric: WindowGasMetric) -> tuple[bool, tuple[str, ...]]:
    direct_signal_ok = metric.signal_to_interference >= MIN_ACCEPTABLE_SIGNAL_TO_INTERFERENCE
    soft_signal_ok = (
        metric.signal_to_interference >= 0.45
        and metric.peak_region_purity >= 0.68
        and metric.peak_region_delta_alpha_selectivity >= 1.8
        and metric.peak_region_wms2f_selectivity >= 4.0
    )
    derivative_escape_hatch = (
        metric.peak_region_delta_alpha_selectivity >= 1.8
        and metric.peak_region_wms2f_selectivity >= 4.0
        and (
            metric.peak_region_wms2f_shape_similarity >= 0.4
            or metric.peak_region_purity >= 0.68
            or metric.signal_to_interference >= 0.45
        )
    )
    strong_multigas_escape_hatch = (
        metric.signal_to_interference >= 4.0
        and metric.peak_region_delta_alpha_selectivity >= 1.1
        and metric.peak_region_wms2f_selectivity >= 0.7
        and (
            metric.peak_region_purity >= 0.4
            or metric.peak_region_selectivity >= 1.5
            or metric.peak_region_wms2f_shape_similarity >= 0.2
        )
    )
    ultra_clean_overlap_escape_hatch = (
        metric.signal_to_interference >= 0.25
        and metric.peak_region_purity >= 0.95
        and metric.peak_region_delta_alpha_selectivity >= 1.6
        and metric.peak_region_wms2f_selectivity >= 4.0
        and metric.peak_region_wms2f_shape_similarity >= 0.9
    )
    signal_ok = (
        direct_signal_ok
        or soft_signal_ok
        or derivative_escape_hatch
        or strong_multigas_escape_hatch
        or ultra_clean_overlap_escape_hatch
    )
    delta_alpha_ok = metric.peak_region_delta_alpha_selectivity >= 1.1
    wms2f_ok = metric.peak_region_wms2f_selectivity >= 0.7
    shape_ok = (
        metric.peak_region_wms2f_shape_similarity >= MIN_ACCEPTABLE_WMS2F_SHAPE_SIMILARITY
        or derivative_escape_hatch
        or strong_multigas_escape_hatch
        or ultra_clean_overlap_escape_hatch
        or (metric.peak_region_purity >= 0.9 and metric.peak_region_wms2f_selectivity >= 4.0)
    )
    reasons: list[str] = []
    if not signal_ok:
        reasons.append("signal")
    if not delta_alpha_ok:
        reasons.append("delta_alpha")
    if not wms2f_ok:
        reasons.append("wms2f")
    if not shape_ok:
        reasons.append("shape")
    return (signal_ok and delta_alpha_ok and wms2f_ok and shape_ok), tuple(reasons)


def _best_window_metric_for_gas(
    result: ManualSpectrumResult,
    target_gases: tuple[str, ...],
    interference_gases: tuple[str, ...],
    gas: str,
    mask: np.ndarray,
) -> tuple[WindowGasMetric | None, float]:
    component = result.components[gas]
    alpha_window = component.alpha_per_cm[mask]
    sigma_window = component.sigma_cm2_per_molecule[mask]
    if alpha_window.size == 0:
        return None, float("-inf")

    local_wavelengths = result.wavelength_um[mask]
    local_wavenumbers = result.wavenumber_cm1[mask]
    nuisance_profiles = {
        other_gas: result.components[other_gas].alpha_per_cm[mask]
        for other_gas in sorted(set(interference_gases) | set(target_gases))
        if other_gas != gas and other_gas in result.components
    }
    nuisance_total_profile = np.zeros_like(alpha_window)
    for profile in nuisance_profiles.values():
        nuisance_total_profile += profile
    interference_nuisances = tuple(
        other_gas for other_gas in interference_gases if other_gas in nuisance_profiles
    )
    target_nuisances = tuple(
        other_gas for other_gas in target_gases if other_gas != gas and other_gas in nuisance_profiles
    )

    scenario_scales: list[dict[str, float]] = []
    scenario_signatures: set[tuple[tuple[str, float], ...]] = set()

    def add_scenario(overrides: dict[str, float]) -> None:
        scales = {other_gas: 1.0 for other_gas in nuisance_profiles}
        scales.update(overrides)
        signature = tuple((other_gas, round(scales[other_gas], 6)) for other_gas in sorted(scales))
        if signature in scenario_signatures:
            return
        scenario_signatures.add(signature)
        scenario_scales.append(scales)

    add_scenario({})
    for other_gas in interference_nuisances:
        for factor in INTERFERENCE_WORST_CASE_SWEEP_FACTORS:
            add_scenario({other_gas: factor})
    for other_gas in target_nuisances:
        for factor in TARGET_WORST_CASE_SWEEP_FACTORS:
            add_scenario({other_gas: factor})
    if target_nuisances:
        add_scenario({other_gas: TARGET_WORST_CASE_SWEEP_FACTORS[-1] for other_gas in target_nuisances})

    candidate_local_indices = _peak_candidates(
        alpha_window,
        competing_signal=nuisance_total_profile,
        max_candidates=MAX_LOCAL_PEAKS_PER_WINDOW_GAS,
    )
    best_metric: WindowGasMetric | None = None
    best_metric_score = float("-inf")
    best_metric_sort_key: tuple[float, float, float, float, float, float] | None = None
    gas_peak = float(np.max(alpha_window))

    for local_peak_idx in candidate_local_indices:
        peak_alpha = float(alpha_window[local_peak_idx])
        if peak_alpha <= 0:
            continue
        peak_sigma = float(sigma_window[local_peak_idx])
        peak_wl = float(local_wavelengths[local_peak_idx])
        peak_nu = float(local_wavenumbers[local_peak_idx])
        region_left, region_right = _peak_region_bounds(alpha_window, local_peak_idx)
        region_slice = slice(region_left, region_right + 1)
        target_region = alpha_window[region_slice]
        region_wavelengths = local_wavelengths[region_slice]
        target_baseline = _peak_flank_baseline(alpha_window, region_left, region_right)
        target_peak_contrast = max(0.0, peak_alpha - target_baseline)
        target_delta_alpha = _peak_to_peak(target_region)
        target_wms2f = _second_derivative_profile(target_region, region_wavelengths)
        target_wms2f_span = _peak_to_peak(target_wms2f)

        worst_signal_to_interference = float("inf")
        worst_peak_region_selectivity = float("inf")
        worst_peak_region_purity = 1.0
        worst_delta_alpha_selectivity = float("inf")
        worst_wms2f_selectivity = float("inf")
        worst_wms2f_shape_similarity = 1.0
        max_interference_alpha = 0.0
        max_other_peak_contrast = 0.0
        max_other_delta_alpha = 0.0

        for scales in scenario_scales:
            other_profile = np.zeros_like(alpha_window)
            for other_gas, profile in nuisance_profiles.items():
                other_profile += profile * scales.get(other_gas, 1.0)
            total_profile = alpha_window + other_profile
            other_region = other_profile[region_slice]
            total_region = total_profile[region_slice]
            other_alpha = float(other_profile[local_peak_idx])
            signal_to_interference = peak_alpha / (other_alpha + 1.0e-30)
            other_baseline = _peak_flank_baseline(other_profile, region_left, region_right)
            total_baseline = _peak_flank_baseline(total_profile, region_left, region_right)
            other_peak_contrast = max(0.0, float(np.max(other_region)) - other_baseline)
            total_peak_contrast = max(0.0, float(np.max(total_region)) - total_baseline)
            peak_region_selectivity = target_peak_contrast / (other_peak_contrast + 1.0e-30)
            peak_region_purity = min(1.0, max(0.0, target_peak_contrast / (total_peak_contrast + 1.0e-30)))
            other_delta_alpha = _peak_to_peak(other_region)
            delta_alpha_selectivity = target_delta_alpha / (other_delta_alpha + 1.0e-30)
            other_wms2f = _second_derivative_profile(other_region, region_wavelengths)
            other_wms2f_span = _peak_to_peak(other_wms2f)
            wms2f_selectivity = target_wms2f_span / (other_wms2f_span + 1.0e-30)
            wms2f_shape_similarity_score = _wms2f_shape_similarity(target_region, total_region, region_wavelengths)

            worst_signal_to_interference = min(worst_signal_to_interference, signal_to_interference)
            worst_peak_region_selectivity = min(worst_peak_region_selectivity, peak_region_selectivity)
            worst_peak_region_purity = min(worst_peak_region_purity, peak_region_purity)
            worst_delta_alpha_selectivity = min(worst_delta_alpha_selectivity, delta_alpha_selectivity)
            worst_wms2f_selectivity = min(worst_wms2f_selectivity, wms2f_selectivity)
            worst_wms2f_shape_similarity = min(worst_wms2f_shape_similarity, wms2f_shape_similarity_score)
            max_interference_alpha = max(max_interference_alpha, other_alpha)
            max_other_peak_contrast = max(max_other_peak_contrast, other_peak_contrast)
            max_other_delta_alpha = max(max_other_delta_alpha, other_delta_alpha)

        prominence_ratio = peak_alpha / (gas_peak + 1.0e-30)
        if prominence_ratio < MIN_CANDIDATE_PROMINENCE_RATIO:
            continue

        strength_ratio = math.sqrt(max(prominence_ratio, 0.0))
        metric_score = 0.0
        metric_score += (prominence_ratio * 85.0)
        metric_score += strength_ratio * 65.0
        metric_score += min(worst_signal_to_interference, 25.0) * 8.0
        metric_score += min(worst_peak_region_selectivity, 25.0) * 14.0
        metric_score += min(max(0.0, worst_peak_region_purity), 1.0) * 95.0
        metric_score += min(worst_delta_alpha_selectivity, 25.0) * 10.0
        metric_score += min(worst_wms2f_selectivity, 25.0) * 12.0
        metric_score += min(max(0.0, worst_wms2f_shape_similarity), 1.0) * 150.0
        metric_sort_key = (
            metric_score,
            strength_ratio,
            peak_alpha,
            worst_peak_region_purity,
            worst_peak_region_selectivity,
            worst_signal_to_interference,
        )
        if best_metric_sort_key is not None and metric_sort_key <= best_metric_sort_key:
            continue

        best_metric_score = metric_score
        best_metric_sort_key = metric_sort_key
        best_metric = WindowGasMetric(
            gas=gas,
            peak_alpha_per_cm=peak_alpha,
            peak_sigma_cm2_per_molecule=peak_sigma,
            peak_wavelength_um=peak_wl,
            peak_wavenumber_cm1=peak_nu,
            peak_index=int(local_peak_idx),
            interference_alpha_per_cm=float(max_interference_alpha),
            signal_to_interference=float(worst_signal_to_interference),
            prominence_ratio=float(prominence_ratio),
            peak_region_target_contrast_per_cm=float(target_peak_contrast),
            peak_region_interference_contrast_per_cm=float(max_other_peak_contrast),
            peak_region_selectivity=float(worst_peak_region_selectivity),
            peak_region_purity=float(worst_peak_region_purity),
            peak_region_target_delta_alpha_per_cm=float(target_delta_alpha),
            peak_region_interference_delta_alpha_per_cm=float(max_other_delta_alpha),
            peak_region_delta_alpha_selectivity=float(worst_delta_alpha_selectivity),
            peak_region_wms2f_selectivity=float(worst_wms2f_selectivity),
            peak_region_wms2f_shape_similarity=float(worst_wms2f_shape_similarity),
        )

    return best_metric, best_metric_score


def _interval_overlap_fraction(
    left_min: float,
    left_max: float,
    right_min: float,
    right_max: float,
) -> float:
    overlap = max(0.0, min(left_max, right_max) - max(left_min, right_min))
    if overlap <= 0:
        return 0.0
    smaller_width = max(min(left_max - left_min, right_max - right_min), 1.0e-12)
    return overlap / smaller_width


def _evaluate_window_candidate(
    result: ManualSpectrumResult,
    target_gases: tuple[str, ...],
    interference_gases: tuple[str, ...],
    wavelength_min_um: float,
    wavelength_max_um: float,
    tuning_span_nm: float,
    seed_gas: str,
    seed_index: int,
) -> LaserWindowCandidate | None:
    mask = (result.wavelength_um >= wavelength_min_um) & (result.wavelength_um <= wavelength_max_um)
    if int(np.sum(mask)) < 8:
        return None

    coverage: list[str] = []
    gas_metrics: dict[str, WindowGasMetric] = {}
    score = 0.0
    center_wl = float(result.wavelength_um[seed_index])

    for gas in target_gases:
        best_metric, best_metric_score = _best_window_metric_for_gas(
            result=result,
            target_gases=target_gases,
            interference_gases=interference_gases,
            gas=gas,
            mask=mask,
        )
        if best_metric is None:
            continue
        accepted, _ = _window_metric_acceptance(best_metric)
        if not accepted:
            continue

        coverage.append(gas)
        score += best_metric_score
        gas_metrics[gas] = best_metric

    if seed_gas not in gas_metrics:
        return None

    if len(coverage) == 0:
        return None

    worst_signal_to_interference = min(
        metric.signal_to_interference for metric in gas_metrics.values()
    )
    mean_signal_to_interference = sum(
        metric.signal_to_interference for metric in gas_metrics.values()
    ) / len(gas_metrics)
    worst_peak_region_selectivity = min(metric.peak_region_selectivity for metric in gas_metrics.values())
    mean_peak_region_selectivity = sum(metric.peak_region_selectivity for metric in gas_metrics.values()) / len(gas_metrics)
    worst_peak_region_purity = min(metric.peak_region_purity for metric in gas_metrics.values())
    mean_peak_region_purity = sum(metric.peak_region_purity for metric in gas_metrics.values()) / len(gas_metrics)
    worst_delta_alpha_selectivity = min(metric.peak_region_delta_alpha_selectivity for metric in gas_metrics.values())
    mean_delta_alpha_selectivity = sum(metric.peak_region_delta_alpha_selectivity for metric in gas_metrics.values()) / len(gas_metrics)
    worst_wms2f_selectivity = min(metric.peak_region_wms2f_selectivity for metric in gas_metrics.values())
    mean_wms2f_selectivity = sum(metric.peak_region_wms2f_selectivity for metric in gas_metrics.values()) / len(gas_metrics)
    worst_wms2f_shape_similarity = min(metric.peak_region_wms2f_shape_similarity for metric in gas_metrics.values())
    mean_wms2f_shape_similarity = sum(metric.peak_region_wms2f_shape_similarity for metric in gas_metrics.values()) / len(gas_metrics)

    required_min_um = min(metric.peak_wavelength_um for metric in gas_metrics.values())
    required_max_um = max(metric.peak_wavelength_um for metric in gas_metrics.values())
    required_span_nm = max(0.0, (required_max_um - required_min_um) * 1000.0)
    if required_span_nm > tuning_span_nm + 1.0e-9:
        return None

    score += 180.0 * len(coverage) * len(coverage)
    score -= max(0.0, 3.0 - worst_signal_to_interference) * 95.0
    score -= max(0.0, 5.0 - mean_signal_to_interference) * 30.0
    score -= max(0.0, 2.0 - worst_peak_region_selectivity) * 180.0
    score -= max(0.0, 4.0 - mean_peak_region_selectivity) * 52.0
    score -= max(0.0, 0.72 - worst_peak_region_purity) * 260.0
    score -= max(0.0, 0.82 - mean_peak_region_purity) * 110.0
    score -= max(0.0, 2.0 - worst_delta_alpha_selectivity) * 220.0
    score -= max(0.0, 4.0 - mean_delta_alpha_selectivity) * 80.0
    score -= max(0.0, 1.5 - worst_wms2f_selectivity) * 260.0
    score -= max(0.0, 3.0 - mean_wms2f_selectivity) * 90.0
    score -= max(0.0, 0.72 - worst_wms2f_shape_similarity) * 320.0
    score -= max(0.0, 0.84 - mean_wms2f_shape_similarity) * 140.0
    score -= required_span_nm * 12.0
    score -= max(0.0, required_span_nm - 2.0) * 6.0

    if score <= 0.0 and (len(coverage) < 2 or score < MIN_MULTI_TARGET_CANDIDATE_SCORE):
        return None

    return LaserWindowCandidate(
        window_id=f"{seed_gas}-{seed_index}-{required_min_um:.6f}-{required_max_um:.6f}",
        wavelength_min_um=float(required_min_um),
        wavelength_max_um=float(required_max_um),
        center_wavelength_um=center_wl,
        tuning_span_nm=float(required_span_nm),
        coverage=tuple(sorted(set(coverage))),
        score=float(score),
        gas_metrics=gas_metrics,
    )


def diagnose_search_window(
    target_concentrations: dict[str, float],
    interference_concentrations: dict[str, float],
    temperature_c: float,
    pressure_hpa: float,
    wavelength_min_um: float,
    wavelength_max_um: float,
    step_cm1: float | None = None,
    data_source: str = LIVE_DB_MODE,
) -> WindowCandidateDiagnostic:
    target_gases = tuple(sorted(gas for gas, value in target_concentrations.items() if value > 0))
    if not target_gases:
        raise ValueError("At least one target gas with positive minimum concentration is required.")

    merged_concentrations = {**interference_concentrations, **target_concentrations}
    effective_step = step_cm1 or recommended_step_cm1(
        abs(float(wavelength_um_to_wavenumber_cm1(wavelength_min_um)) - float(wavelength_um_to_wavenumber_cm1(wavelength_max_um))),
        manual_mode=False,
    )
    result = build_manual_spectrum(
        concentrations=merged_concentrations,
        temperature_c=temperature_c,
        pressure_hpa=pressure_hpa,
        range_unit="um",
        range_min=wavelength_min_um,
        range_max=wavelength_max_um,
        step_cm1=effective_step,
        data_source=data_source,
    )
    interference_gases = tuple(sorted(gas for gas, value in interference_concentrations.items() if value > 0))
    mask = (result.wavelength_um >= wavelength_min_um) & (result.wavelength_um <= wavelength_max_um)
    if int(np.sum(mask)) < 8:
        raise ValueError("Window is too narrow for diagnostics.")

    gas_diagnostics: dict[str, WindowGasDiagnostic] = {}
    coverage: list[str] = []
    score = 0.0
    for gas in target_gases:
        metric, metric_score = _best_window_metric_for_gas(
            result=result,
            target_gases=target_gases,
            interference_gases=interference_gases,
            gas=gas,
            mask=mask,
        )
        if metric is None:
            gas_diagnostics[gas] = WindowGasDiagnostic(
                gas=gas,
                accepted=False,
                rejection_reasons=("no_peak_candidate",),
                metric=None,
            )
            continue
        accepted, rejection_reasons = _window_metric_acceptance(metric)
        if accepted:
            coverage.append(gas)
            score += metric_score
        gas_diagnostics[gas] = WindowGasDiagnostic(
            gas=gas,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            metric=metric,
        )

    return WindowCandidateDiagnostic(
        wavelength_min_um=float(wavelength_min_um),
        wavelength_max_um=float(wavelength_max_um),
        tuning_span_nm=max(0.0, (float(wavelength_max_um) - float(wavelength_min_um)) * 1000.0),
        coverage=tuple(sorted(coverage)),
        score=float(score),
        gas_diagnostics=gas_diagnostics,
    )


def suggest_laser_plans(
    target_concentrations: dict[str, float],
    interference_concentrations: dict[str, float],
    temperature_c: float,
    pressure_hpa: float,
    range_unit: str,
    range_min: float,
    range_max: float,
    tuning_range_nm: float,
    max_lasers: int,
    step_cm1: float | None = None,
    data_source: str = LIVE_DB_MODE,
    max_peak_candidates_per_gas: int = 40,
    top_window_pool: int = 90,
    top_plan_count: int = 24,
) -> tuple[list[LaserPlan], ManualSpectrumResult]:
    if max_lasers < 1:
        raise ValueError("At least one laser must be allowed.")

    target_gases = tuple(sorted(gas for gas, value in target_concentrations.items() if value > 0))
    if not target_gases:
        raise ValueError("At least one target gas with positive minimum concentration is required.")

    merged_concentrations = {**interference_concentrations, **target_concentrations}
    span_cm1 = normalize_wavenumber_window(range_unit, range_min, range_max)
    effective_step = step_cm1 or recommended_step_cm1(span_cm1[1] - span_cm1[0], manual_mode=False)
    manual_result = build_manual_spectrum(
        concentrations=merged_concentrations,
        temperature_c=temperature_c,
        pressure_hpa=pressure_hpa,
        range_unit=range_unit,
        range_min=range_min,
        range_max=range_max,
        step_cm1=effective_step,
        data_source=data_source,
    )
    interference_gases = tuple(sorted(gas for gas, value in interference_concentrations.items() if value > 0))
    wavelength_min_um = float(np.min(manual_result.wavelength_um))
    wavelength_max_um = float(np.max(manual_result.wavelength_um))

    candidate_windows: dict[tuple[float, float], LaserWindowCandidate] = {}
    half_window_um = tuning_range_nm / 2000.0

    for gas in target_gases:
        signal = manual_result.components[gas].alpha_per_cm
        competing_signal = manual_result.total_alpha_per_cm - signal
        peak_indices = _peak_candidates(
            signal,
            competing_signal=competing_signal,
            max_candidates=max_peak_candidates_per_gas,
        )
        for peak_index in peak_indices:
            peak_wl = float(manual_result.wavelength_um[peak_index])
            window_min = max(wavelength_min_um, peak_wl - half_window_um)
            window_max = min(wavelength_max_um, peak_wl + half_window_um)
            if (window_max - window_min) <= 0:
                continue
            candidate = _evaluate_window_candidate(
                result=manual_result,
                target_gases=target_gases,
                interference_gases=interference_gases,
                wavelength_min_um=window_min,
                wavelength_max_um=window_max,
                tuning_span_nm=tuning_range_nm,
                seed_gas=gas,
                seed_index=int(peak_index),
            )
            if candidate is None:
                continue
            key = (round(candidate.wavelength_min_um, 6), round(candidate.wavelength_max_um, 6))
            current_best = candidate_windows.get(key)
            if current_best is None or candidate.score > current_best.score:
                candidate_windows[key] = candidate

    sorted_windows = sorted(candidate_windows.values(), key=lambda item: item.score, reverse=True)
    ranked_windows: list[LaserWindowCandidate] = []
    for candidate in sorted_windows:
        too_similar = any(
            candidate.coverage == existing.coverage
            and _interval_overlap_fraction(
                candidate.wavelength_min_um,
                candidate.wavelength_max_um,
                existing.wavelength_min_um,
                existing.wavelength_max_um,
            ) >= 0.85
            for existing in ranked_windows
        )
        if too_similar:
            continue
        ranked_windows.append(candidate)
        if len(ranked_windows) >= top_window_pool:
            break

    plans: list[LaserPlan] = []
    for laser_count in range(1, min(max_lasers, len(ranked_windows)) + 1):
        for combo in combinations(ranked_windows, laser_count):
            covered_targets = tuple(sorted({gas for window in combo for gas in window.coverage}))
            missing_targets = tuple(sorted(set(target_gases) - set(covered_targets)))
            metrics = [metric for window in combo for metric in window.gas_metrics.values()]
            coverage_counts = [len(window.coverage) for window in combo]
            overlap_occurrences = sum(coverage_counts) - len(covered_targets)
            coverage_imbalance = (max(coverage_counts) - min(coverage_counts)) if coverage_counts else 0
            total_tuning_span_nm = sum(window.tuning_span_nm for window in combo)
            max_window_span_nm = max((window.tuning_span_nm for window in combo), default=0.0)
            weakest_peak_alpha = min((metric.peak_alpha_per_cm for metric in metrics), default=1.0e-30)
            mean_peak_alpha = sum((metric.peak_alpha_per_cm for metric in metrics), 0.0) / max(len(metrics), 1)
            weakest_margin_score = min(
                (
                    math.log1p(max(metric.signal_to_interference, 0.0))
                    + math.log1p(max(metric.peak_region_delta_alpha_selectivity, 0.0))
                    + math.log1p(max(metric.peak_region_wms2f_selectivity, 0.0))
                    + max(metric.peak_region_wms2f_shape_similarity, 0.0)
                )
                for metric in metrics
            ) if metrics else 0.0
            combo_score = 0.0
            combo_score += 6_000_000.0 * len(covered_targets)
            combo_score -= 8_000_000.0 * len(missing_targets)
            combo_score -= 150_000.0 * (laser_count - 1)
            combo_score -= 1_400_000.0 * overlap_occurrences
            combo_score -= 450_000.0 * coverage_imbalance
            combo_score -= total_tuning_span_nm * 10_000.0
            combo_score -= max_window_span_nm * 8_000.0
            combo_score += math.log10(max(weakest_peak_alpha, 1.0e-30)) * 50_000.0
            combo_score += math.log10(max(mean_peak_alpha, 1.0e-30)) * 15_000.0
            combo_score += weakest_margin_score * 5_000.0
            combo_score += sum(window.score for window in combo) * 6.0
            if not missing_targets:
                combo_score += 3_000_000.0
            plans.append(
                LaserPlan(
                    rank=0,
                    score=float(combo_score),
                    covered_targets=covered_targets,
                    missing_targets=missing_targets,
                    windows=combo,
                )
            )

    plans = sorted(plans, key=lambda plan: plan.score, reverse=True)

    def is_plan_too_similar(
        candidate: LaserPlan,
        existing_plans: list[LaserPlan],
        overlap_threshold: float,
    ) -> bool:
        for existing in existing_plans:
            if candidate.covered_targets != existing.covered_targets:
                continue
            overlap_hits = 0
            for window in candidate.windows:
                if any(
                    _interval_overlap_fraction(
                        window.wavelength_min_um,
                        window.wavelength_max_um,
                        existing_window.wavelength_min_um,
                        existing_window.wavelength_max_um,
                    ) >= overlap_threshold
                    for existing_window in existing.windows
                ):
                    overlap_hits += 1
            if overlap_hits == len(candidate.windows):
                return True
        return False

    ranked_plans: list[LaserPlan] = []
    seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    def append_ranked_plans(
        overlap_threshold: float,
        limit: int,
        *,
        unique_coverage_only: bool = False,
    ) -> None:
        seen_coverage_sets = {plan.covered_targets for plan in ranked_plans}
        for plan in plans:
            if len(ranked_plans) >= limit:
                break
            if unique_coverage_only and plan.covered_targets in seen_coverage_sets:
                continue
            signature = (
                tuple(window.window_id for window in plan.windows),
                plan.covered_targets,
            )
            if signature in seen_signatures:
                continue
            if is_plan_too_similar(plan, ranked_plans, overlap_threshold):
                continue
            seen_signatures.add(signature)
            ranked_plans.append(
                LaserPlan(
                    rank=len(ranked_plans) + 1,
                    score=plan.score,
                    covered_targets=plan.covered_targets,
                    missing_targets=plan.missing_targets,
                    windows=plan.windows,
                )
            )
            seen_coverage_sets.add(plan.covered_targets)

    append_ranked_plans(
        overlap_threshold=0.8,
        limit=top_plan_count,
        unique_coverage_only=True,
    )

    fallback_fill_target = min(top_plan_count, max(10, len(target_gases) * 4))
    if len(ranked_plans) < fallback_fill_target:
        append_ranked_plans(overlap_threshold=0.97, limit=fallback_fill_target)

    ranked_plans = [
        LaserPlan(
            rank=index,
            score=plan.score,
            covered_targets=plan.covered_targets,
            missing_targets=plan.missing_targets,
            windows=plan.windows,
        )
        for index, plan in enumerate(sorted(ranked_plans, key=lambda plan: plan.score, reverse=True), start=1)
    ]

    return ranked_plans, manual_result


def serialize_laser_plan(plan: LaserPlan) -> dict[str, Any]:
    return {
        "rank": plan.rank,
        "score": plan.score,
        "covered_targets": list(plan.covered_targets),
        "missing_targets": list(plan.missing_targets),
        "windows": [
            {
                "window_id": window.window_id,
                "wavelength_min_um": window.wavelength_min_um,
                "wavelength_max_um": window.wavelength_max_um,
                "center_wavelength_um": window.center_wavelength_um,
                "tuning_span_nm": window.tuning_span_nm,
                "coverage": list(window.coverage),
                "score": window.score,
                "gas_metrics": {
                    gas: asdict(metric)
                    for gas, metric in window.gas_metrics.items()
                },
            }
            for window in plan.windows
        ],
    }


def deserialize_laser_plan(payload: dict[str, Any]) -> LaserPlan:
    return LaserPlan(
        rank=int(payload["rank"]),
        score=float(payload["score"]),
        covered_targets=tuple(payload["covered_targets"]),
        missing_targets=tuple(payload["missing_targets"]),
        windows=tuple(
            LaserWindowCandidate(
                window_id=window["window_id"],
                wavelength_min_um=float(window["wavelength_min_um"]),
                wavelength_max_um=float(window["wavelength_max_um"]),
                center_wavelength_um=float(window["center_wavelength_um"]),
                tuning_span_nm=float(window["tuning_span_nm"]),
                coverage=tuple(window["coverage"]),
                score=float(window["score"]),
                gas_metrics={
                    gas: WindowGasMetric(**metric)
                    for gas, metric in window["gas_metrics"].items()
                },
            )
            for window in payload["windows"]
        ),
    )