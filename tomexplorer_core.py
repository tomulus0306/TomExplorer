from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from itertools import combinations
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

BASE_DIR = Path(__file__).resolve().parent
LEGACY_DATA_DIR = BASE_DIR / "data"
DATA_DIR = BASE_DIR / "hitran_cache"
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


def _safe_gas_key(label: str, molecule_id: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    if not cleaned:
        return f"M{molecule_id}"
    if cleaned[0].isdigit():
        return f"M{molecule_id}_{cleaned}"
    return cleaned


def _dropdown_label(formula: str, molecule_id: int) -> str:
    molecule_name = HITRAN_MOLECULE_NAMES.get(molecule_id)
    if not molecule_name:
        return formula
    return f"{formula} - {molecule_name}"


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

    plot_colors = _component_color_map(tuple(sorted(catalog.keys())))
    for gas, plot_color in plot_colors.items():
        catalog[gas]["plot_color"] = plot_color

    return catalog


GAS_LIBRARY: dict[str, dict[str, Any]] = _build_gas_library()

_FETCHED_RANGES: dict[str, tuple[float, float]] = {}


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
    )


def _ensure_database_started() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    hp.db_begin(str(DATA_DIR))


def _clear_runtime_caches() -> None:
    _load_offline_sigma_library.cache_clear()
    offline_library_metadata.cache_clear()
    _cached_offline_sigma_bundle.cache_clear()
    _cached_sigma_bundle.cache_clear()
    _local_table_range.cache_clear()


def _has_local_table_cache(gas: str) -> bool:
    return (DATA_DIR / f"{gas}.data").exists() and (DATA_DIR / f"{gas}.header").exists()


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


def _sigma_from_local_db(
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


def _write_offline_sigma_library(
    library: dict[str, tuple[np.ndarray, np.ndarray]],
    metadata: dict[str, Any] | None = None,
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
        updated_gases: list[str] = []
        for gas in selected_gases:
            if gas in library:
                axis, sigma = library[gas]
                local_step = float(np.median(np.diff(axis))) if axis.size > 1 else float(step_cm1 or DEFAULT_MANUAL_STEP_CM1)
                union_min = min(float(axis[0]), nu_min)
                union_max = max(float(axis[-1]), nu_max)
                union_axis = np.arange(union_min, union_max + (local_step * 0.5), local_step, dtype=float)
                union_sigma = np.interp(union_axis, axis, sigma)
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
            replace_mask = (union_axis >= nu_min) & (union_axis <= nu_max)
            union_sigma[replace_mask] = np.interp(union_axis[replace_mask], current_nu, current_sigma)
            library[gas] = (union_axis, union_sigma)
            updated_gases.append(gas)

        _write_offline_sigma_library(
            library,
            metadata={
                "native_step_cm1": float(step_cm1 or DEFAULT_MANUAL_STEP_CM1),
                "reference_temperature_c": temperature_c,
                "reference_pressure_hpa": pressure_hpa,
            },
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
    for gas in selected_gases:
        current_nu, current_sigma = _sigma_from_local_db(
            gas=gas,
            temperature_c=temperature_c,
            pressure_hpa=pressure_hpa,
            nu_min=nu_min,
            nu_max=nu_max,
            step_cm1=effective_step,
        )
        library[gas] = (common_axis, np.interp(common_axis, current_nu, current_sigma))

    _write_offline_sigma_library(
        library,
        metadata={
            "native_step_cm1": effective_step,
            "reference_temperature_c": temperature_c,
            "reference_pressure_hpa": pressure_hpa,
        },
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
        raise ValueError(
            "Offline spectra are missing for: "
            + ", ".join(missing_gases)
            + ". Refresh HITRAN manually and rebuild the pickle before using these gases here."
        )

    coverage_min = max(float(library[gas][0][0]) for gas in gases)
    coverage_max = min(float(library[gas][0][-1]) for gas in gases)
    coverage_tolerance_cm1 = max(1e-6, abs(coverage_min) * 1e-12, abs(coverage_max) * 1e-12)
    if nu_min < (coverage_min - coverage_tolerance_cm1) or nu_max > (coverage_max + coverage_tolerance_cm1):
        raise ValueError(
            "Requested range is outside offline pickle coverage "
            f"({coverage_min:.2f}-{coverage_max:.2f} cm-1). "
            "Use the manual HITRAN refresh only to update the local DB, then rebuild the pickle for plotting."
        )
    nu_min = max(nu_min, coverage_min)
    nu_max = min(nu_max, coverage_max)

    reference_axis = library[gases[0]][0]
    native_step = float(np.median(np.diff(reference_axis))) if reference_axis.size > 1 else step_cm1
    target_step = max(float(step_cm1), native_step)
    point_count = max(2, int(np.floor((nu_max - nu_min) / target_step)) + 1)
    target_axis = np.linspace(nu_min, nu_max, point_count, dtype=float)

    sigma_map: dict[str, np.ndarray] = {}
    for gas in gases:
        gas_axis, gas_sigma = library[gas]
        sigma_map[gas] = np.interp(target_axis, gas_axis, gas_sigma)

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
    successful_gases: list[str] = []
    failed_gases: list[str] = []
    for gas in selected_gases:
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
            successful_gases.append(gas)
        except Exception as exc:
            reset_hitran_tables((temp_table,))
            failed_gases.append(f"{gas} ({exc})")

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
        f"Lokaler HITRAN/HAPI-Cache fuer {', '.join(successful_gases)} aktualisiert "
        f"({fetch_min:.2f}-{fetch_max:.2f} cm-1 inkl. Rand). "
        "Die Live-Berechnung nutzt diese Tabellen sofort; offline funktionieren danach genau diese lokal gecachten Bereiche auch ohne Download."
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
    try:
        for gas in gases:
            current_nu_array, current_sigma_array = _sigma_from_local_db(
                gas=gas,
                temperature_c=temperature_c,
                pressure_hpa=pressure_hpa,
                nu_min=nu_min,
                nu_max=nu_max,
                step_cm1=step_cm1,
            )
            if axis is None:
                axis = current_nu_array
            sigma_map[gas] = current_sigma_array
    except Exception:
        offline_axis, offline_sigma_map = _cached_offline_sigma_bundle(
            gases,
            nu_min,
            nu_max,
            step_cm1,
        )
        return offline_axis, {
            gas: np.asarray(offline_sigma_map[gas], dtype=float)
            for gas in gases
        }

    if axis is None:
        raise ValueError("No gases were provided for spectrum calculation.")
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
    strongest = sorted(peak_indices, key=lambda idx: signal[idx], reverse=True)[:max_candidates]
    most_selective = sorted(
        peak_indices,
        key=lambda idx: (
            signal[idx] / (competing_signal[idx] + 1.0e-30),
            signal[idx],
        ),
        reverse=True,
    )[:max_candidates]

    combined: list[int] = []
    for idx in strongest + most_selective:
        if idx not in combined:
            combined.append(idx)
    return np.asarray(combined, dtype=int)


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
    def peak_to_peak(profile: np.ndarray) -> float:
        if profile.size == 0:
            return 0.0
        return float(np.max(profile) - np.min(profile))

    def second_derivative_profile(profile: np.ndarray, axis: np.ndarray) -> np.ndarray:
        if profile.size < 3:
            return np.zeros_like(profile)
        first_derivative = np.gradient(profile, axis)
        return np.asarray(np.gradient(first_derivative, axis), dtype=float)

    def normalized_shape(profile: np.ndarray) -> np.ndarray:
        if profile.size == 0:
            return np.zeros_like(profile)
        centered = np.asarray(profile - float(np.mean(profile)), dtype=float)
        scale = float(np.max(np.abs(centered)))
        if scale <= 1.0e-30:
            return np.zeros_like(centered)
        return centered / scale

    def wms2f_shape_similarity(
        target_profile: np.ndarray,
        total_profile: np.ndarray,
        axis: np.ndarray,
    ) -> float:
        target_2f = normalized_shape(second_derivative_profile(target_profile, axis))
        total_2f = normalized_shape(second_derivative_profile(total_profile, axis))
        if target_2f.size < 2:
            return 1.0
        rmse = float(np.sqrt(np.mean((target_2f - total_2f) ** 2)))
        correlation = float(np.corrcoef(target_2f, total_2f)[0, 1])
        if not np.isfinite(correlation):
            correlation = 1.0
        rmse_similarity = max(0.0, 1.0 - (rmse / 2.0))
        corr_similarity = min(1.0, max(0.0, (correlation + 1.0) / 2.0))
        return float((rmse_similarity * 0.65) + (corr_similarity * 0.35))

    def peak_region_bounds(profile: np.ndarray, peak_index: int) -> tuple[int, int]:
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

    def peak_flank_baseline(profile: np.ndarray, left: int, right: int) -> float:
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

    mask = (result.wavelength_um >= wavelength_min_um) & (result.wavelength_um <= wavelength_max_um)
    if int(np.sum(mask)) < 8:
        return None

    coverage: list[str] = []
    gas_metrics: dict[str, WindowGasMetric] = {}
    score = 0.0
    center_wl = float(result.wavelength_um[seed_index])
    peak_region_selectivity_values: list[float] = []
    peak_region_purity_values: list[float] = []
    delta_alpha_selectivity_values: list[float] = []
    wms2f_selectivity_values: list[float] = []
    wms2f_shape_similarity_values: list[float] = []

    for gas in target_gases:
        component = result.components[gas]
        alpha_window = component.alpha_per_cm[mask]
        sigma_window = component.sigma_cm2_per_molecule[mask]
        if alpha_window.size == 0:
            continue
        local_peak_idx = int(np.argmax(alpha_window))
        peak_alpha = float(alpha_window[local_peak_idx])
        if peak_alpha <= 0:
            continue
        peak_sigma = float(sigma_window[local_peak_idx])
        local_wavelengths = result.wavelength_um[mask]
        local_wavenumbers = result.wavenumber_cm1[mask]
        peak_wl = float(local_wavelengths[local_peak_idx])
        peak_nu = float(local_wavenumbers[local_peak_idx])
        region_left, region_right = peak_region_bounds(alpha_window, local_peak_idx)
        region_slice = slice(region_left, region_right + 1)
        target_region = alpha_window[region_slice]
        region_wavelengths = local_wavelengths[region_slice]
        target_baseline = peak_flank_baseline(alpha_window, region_left, region_right)
        target_peak_contrast = max(0.0, peak_alpha - target_baseline)
        target_delta_alpha = peak_to_peak(target_region)
        target_wms2f = second_derivative_profile(target_region, region_wavelengths)
        target_wms2f_span = peak_to_peak(target_wms2f)

        nuisance_profiles = {
            other_gas: result.components[other_gas].alpha_per_cm[mask]
            for other_gas in sorted(set(interference_gases) | set(target_gases))
            if other_gas != gas and other_gas in result.components
        }
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
            signature = tuple(
                (other_gas, round(scales[other_gas], 6))
                for other_gas in sorted(scales)
            )
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
            other_baseline = peak_flank_baseline(other_profile, region_left, region_right)
            total_baseline = peak_flank_baseline(total_profile, region_left, region_right)
            other_peak_contrast = max(
                0.0,
                float(np.max(other_region)) - other_baseline,
            )
            total_peak_contrast = max(
                0.0,
                float(np.max(total_region)) - total_baseline,
            )
            peak_region_selectivity = target_peak_contrast / (other_peak_contrast + 1.0e-30)
            peak_region_purity = min(
                1.0,
                max(0.0, target_peak_contrast / (total_peak_contrast + 1.0e-30)),
            )
            other_delta_alpha = peak_to_peak(other_region)
            delta_alpha_selectivity = target_delta_alpha / (other_delta_alpha + 1.0e-30)
            other_wms2f = second_derivative_profile(other_region, region_wavelengths)
            other_wms2f_span = peak_to_peak(other_wms2f)
            wms2f_selectivity = target_wms2f_span / (other_wms2f_span + 1.0e-30)
            wms2f_shape_similarity_score = wms2f_shape_similarity(
                target_region,
                total_region,
                region_wavelengths,
            )

            worst_signal_to_interference = min(worst_signal_to_interference, signal_to_interference)
            worst_peak_region_selectivity = min(worst_peak_region_selectivity, peak_region_selectivity)
            worst_peak_region_purity = min(worst_peak_region_purity, peak_region_purity)
            worst_delta_alpha_selectivity = min(worst_delta_alpha_selectivity, delta_alpha_selectivity)
            worst_wms2f_selectivity = min(worst_wms2f_selectivity, wms2f_selectivity)
            worst_wms2f_shape_similarity = min(
                worst_wms2f_shape_similarity,
                wms2f_shape_similarity_score,
            )
            max_interference_alpha = max(max_interference_alpha, other_alpha)
            max_other_peak_contrast = max(max_other_peak_contrast, other_peak_contrast)
            max_other_delta_alpha = max(max_other_delta_alpha, other_delta_alpha)

        other_alpha = max_interference_alpha
        signal_to_interference = worst_signal_to_interference
        other_peak_contrast = max_other_peak_contrast
        peak_region_selectivity = worst_peak_region_selectivity
        peak_region_purity = worst_peak_region_purity
        other_delta_alpha = max_other_delta_alpha
        delta_alpha_selectivity = worst_delta_alpha_selectivity
        wms2f_selectivity = worst_wms2f_selectivity
        wms2f_shape_similarity_score = worst_wms2f_shape_similarity
        gas_peak = float(np.max(component.alpha_per_cm))
        prominence_ratio = peak_alpha / (gas_peak + 1.0e-30)
        if prominence_ratio < 0.12:
            continue

        coverage.append(gas)
        peak_region_selectivity_values.append(peak_region_selectivity)
        peak_region_purity_values.append(peak_region_purity)
        delta_alpha_selectivity_values.append(delta_alpha_selectivity)
        wms2f_selectivity_values.append(wms2f_selectivity)
        wms2f_shape_similarity_values.append(wms2f_shape_similarity_score)
        score += (prominence_ratio * 42.0)
        score += min(signal_to_interference, 25.0) * 6.0
        score += min(peak_region_selectivity, 25.0) * 14.0
        score += min(max(0.0, peak_region_purity), 1.0) * 95.0
        score += min(delta_alpha_selectivity, 25.0) * 10.0
        score += min(wms2f_selectivity, 25.0) * 12.0
        score += min(max(0.0, wms2f_shape_similarity_score), 1.0) * 150.0
        gas_metrics[gas] = WindowGasMetric(
            gas=gas,
            peak_alpha_per_cm=peak_alpha,
            peak_sigma_cm2_per_molecule=peak_sigma,
            peak_wavelength_um=peak_wl,
            peak_wavenumber_cm1=peak_nu,
            peak_index=local_peak_idx,
            interference_alpha_per_cm=float(other_alpha),
            signal_to_interference=float(signal_to_interference),
            prominence_ratio=float(prominence_ratio),
            peak_region_target_contrast_per_cm=float(target_peak_contrast),
            peak_region_interference_contrast_per_cm=float(other_peak_contrast),
            peak_region_selectivity=float(peak_region_selectivity),
            peak_region_purity=float(peak_region_purity),
            peak_region_target_delta_alpha_per_cm=float(target_delta_alpha),
            peak_region_interference_delta_alpha_per_cm=float(other_delta_alpha),
            peak_region_delta_alpha_selectivity=float(delta_alpha_selectivity),
            peak_region_wms2f_selectivity=float(wms2f_selectivity),
            peak_region_wms2f_shape_similarity=float(wms2f_shape_similarity_score),
        )

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
    worst_peak_region_selectivity = min(peak_region_selectivity_values)
    mean_peak_region_selectivity = sum(peak_region_selectivity_values) / len(peak_region_selectivity_values)
    worst_peak_region_purity = min(peak_region_purity_values)
    mean_peak_region_purity = sum(peak_region_purity_values) / len(peak_region_purity_values)
    worst_delta_alpha_selectivity = min(delta_alpha_selectivity_values)
    mean_delta_alpha_selectivity = sum(delta_alpha_selectivity_values) / len(delta_alpha_selectivity_values)
    worst_wms2f_selectivity = min(wms2f_selectivity_values)
    mean_wms2f_selectivity = sum(wms2f_selectivity_values) / len(wms2f_selectivity_values)
    worst_wms2f_shape_similarity = min(wms2f_shape_similarity_values)
    mean_wms2f_shape_similarity = sum(wms2f_shape_similarity_values) / len(wms2f_shape_similarity_values)

    required_min_um = min(metric.peak_wavelength_um for metric in gas_metrics.values())
    required_max_um = max(metric.peak_wavelength_um for metric in gas_metrics.values())
    required_span_nm = max(0.0, (required_max_um - required_min_um) * 1000.0)
    if required_span_nm > tuning_span_nm + 1.0e-9:
        return None

    score += 12.0 * len(coverage)
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
    max_peak_candidates_per_gas: int = 10,
    top_window_pool: int = 16,
    top_plan_count: int = 20,
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
            combo_score = sum(window.score for window in combo)
            combo_score += 1000.0 * (len(covered_targets) / len(target_gases))
            combo_score -= 25.0 * (laser_count - 1)
            combo_score -= sum(window.tuning_span_nm for window in combo) * 8.0
            plans.append(
                LaserPlan(
                    rank=0,
                    score=float(combo_score),
                    covered_targets=covered_targets,
                    missing_targets=missing_targets,
                    windows=combo,
                )
            )

    plans = sorted(
        plans,
        key=lambda plan: (
            -len(plan.covered_targets),
            len(plan.missing_targets),
            -plan.score,
            len(plan.windows),
        ),
    )

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

    def append_ranked_plans(overlap_threshold: float, limit: int) -> None:
        for plan in plans:
            if len(ranked_plans) >= limit:
                break
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

    append_ranked_plans(overlap_threshold=0.8, limit=top_plan_count)

    fallback_fill_target = min(top_plan_count, max(10, len(target_gases) * 4))
    if len(ranked_plans) < fallback_fill_target:
        append_ranked_plans(overlap_threshold=0.97, limit=fallback_fill_target)

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