from __future__ import annotations

import csv
from datetime import datetime
import io
import os
from pathlib import Path
import threading
import time
from typing import Any
import urllib.request
import webbrowser

import dash
import hapi as hp
import numpy as np
from dash import ClientsideFunction, Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from tomexplorer_core import (
    DEFAULT_MANUAL_STEP_CM1,
    GAS_LIBRARY,
    LaserPlan,
    LIVE_DB_MODE,
    ManualSpectrumResult,
    OFFLINE_DB_MODE,
    PICKLE_REBUILD_PRESSURE_HPA,
    PICKLE_REBUILD_TEMPERATURE_C,
    build_manual_spectrum,
    concentration_to_molar_fraction,
    deserialize_laser_plan,
    deserialize_manual_result,
    downsample_manual_result,
    format_concentration,
    gas_options,
    hover_payload,
    list_manual_result_snapshots,
    load_manual_result_snapshot,
    load_matching_manual_result_snapshot,
    normalize_wavenumber_window,
    offline_library_summary,
    recommended_step_cm1,
    refresh_hitran_database,
    save_manual_result_snapshot,
    serialize_laser_plan,
    serialize_manual_result,
    suggest_laser_plans,
    wavelength_um_to_wavenumber_cm1,
    wavenumber_cm1_to_wavelength_um,
)


APP_TITLE = "TomExplorer"
BASE_DIR = Path(__file__).resolve().parent
LOGO_ASSET_PATH = "/assets/tomexplorer-logo.svg"
BODY_FONT = "Aptos, 'Segoe UI Variable', 'Segoe UI', sans-serif"
DISPLAY_FONT = "Constantia, 'Palatino Linotype', Georgia, serif"
HITRAN_RUNTIME_LABEL = f"HAPI {getattr(hp, 'HAPI_VERSION', 'unbekannt')}"
SEARCH_PLOT_FINE_STEP_CM1 = 0.001
SEARCH_PLOT_FINE_MAX_POINTS = 15000
SEARCH_PLOT_PADDING_UM = 0.00075
PAS_REFERENCE_SIGMA_NV = 18.0
PAS_REFERENCE_POPT_MW = 11.0
PAS_REFERENCE_EPS = 1.0
PAS_REFERENCE_DELTA_ALPHA_MIN = 1.8e-7
PAS_REFERENCE_CONCENTRATION_PPBV = 7.0
UNIT_OPTIONS = [
    {"label": "ppb", "value": "ppb"},
    {"label": "ppm", "value": "ppm"},
    {"label": "%", "value": "%"},
    {"label": "Molenbruch", "value": "fraction"},
]

DEFAULT_MANUAL_GASES: list[str] = []
DEFAULT_MANUAL_CONCENTRATIONS = {
    "CH4": (2.0, "ppm"),
    "C2H6": (100.0, "ppb"),
    "H2O": (1.0, "%"),
    "CO2": (500.0, "ppm"),
}
DEFAULT_TARGET_GASES: list[str] = []
DEFAULT_TARGET_CONCENTRATIONS = {"CH4": (2.0, "ppm"), "C2H6": (100.0, "ppb")}
DEFAULT_INTERFERENCE_GASES: list[str] = []
DEFAULT_INTERFERENCE_CONCENTRATIONS = {
    "H2O": (2.0, "%"),
    "CO2": (500.0, "ppm"),
    "CO": (200.0, "ppb"),
    "N2O": (500.0, "ppb"),
}
ALL_GASES = sorted(GAS_LIBRARY.keys())
CACHE_STALENESS_DAYS = 30
BUTTON_LOCK_HIDDEN = {"display": "none"}
BUTTON_LOCK_VISIBLE = {"display": "flex"}
MANUAL_CANCEL_EVENT = threading.Event()

MANUAL_CONCENTRATION_STATES = [
    State(f"manual-concentration-value-{gas}", "value") for gas in ALL_GASES
] + [
    State(f"manual-concentration-unit-{gas}", "value") for gas in ALL_GASES
]

MANUAL_CONCENTRATION_INPUTS = [
    Input(f"manual-concentration-value-{gas}", "value") for gas in ALL_GASES
] + [
    Input(f"manual-concentration-unit-{gas}", "value") for gas in ALL_GASES
]

TARGET_CONCENTRATION_STATES = [
    State(f"search-target-concentration-value-{gas}", "value") for gas in ALL_GASES
] + [
    State(f"search-target-concentration-unit-{gas}", "value") for gas in ALL_GASES
]

INTERFERENCE_CONCENTRATION_STATES = [
    State(f"search-interference-concentration-value-{gas}", "value") for gas in ALL_GASES
] + [
    State(f"search-interference-concentration-unit-{gas}", "value") for gas in ALL_GASES
]


app = Dash(__name__, title=APP_TITLE)
server = app.server
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel=\"icon\" type=\"image/svg+xml\" href=\"/assets/tomexplorer-logo.svg\">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def default_concentration_for(gas: str, lookup: dict[str, tuple[float, str]]) -> tuple[float, str]:
    return lookup.get(gas, (100.0, "ppb"))


def display_formula(gas: str) -> str:
    return gas.translate(SUBSCRIPT_DIGITS)


def display_formula_plot(gas: str) -> str:
    # Plotly annotation text is easier to read with plain ASCII formulas.
    return gas


def component_visibility_options(serialized_result: dict[str, Any] | None) -> list[dict[str, str]]:
    if not serialized_result:
        return []
    components = serialized_result.get("components", {})
    gases = sorted(components.keys())
    return [{"label": display_formula(gas), "value": gas} for gas in gases]


def normalized_visible_gases(options: list[dict[str, str]], selected: list[str] | None) -> list[str]:
    option_values = [entry["value"] for entry in options]
    if not option_values:
        return []
    if not selected:
        return option_values
    filtered = [gas for gas in selected if gas in option_values]
    return filtered or option_values


def normalize_wavelength_window(range_unit: str, range_min: float, range_max: float) -> tuple[float, float]:
    if range_unit == "um":
        return float(min(range_min, range_max)), float(max(range_min, range_max))

    nu_min, nu_max = normalize_wavenumber_window(range_unit, range_min, range_max)
    wavelength_min_um = float(wavenumber_cm1_to_wavelength_um(nu_max))
    wavelength_max_um = float(wavenumber_cm1_to_wavelength_um(nu_min))
    return min(wavelength_min_um, wavelength_max_um), max(wavelength_min_um, wavelength_max_um)


def format_coverage_interval(interval_cm1: tuple[float, float], range_unit: str) -> str:
    start_cm1, end_cm1 = float(interval_cm1[0]), float(interval_cm1[1])
    if range_unit == "um":
        start_um, end_um = normalize_wavelength_window("cm-1", start_cm1, end_cm1)
        return f"{start_um:.3f}-{end_um:.3f} µm"
    return f"{start_cm1:.2f}-{end_cm1:.2f} cm⁻¹"


def coverage_gap_notice(
    result: ManualSpectrumResult,
    gases: list[str] | tuple[str, ...] | None,
    range_unit: str,
    intro: str,
) -> str:
    gap_map = result.missing_ranges_cm1_by_gas or {}
    notes: list[str] = []
    for gas in gases or []:
        intervals = tuple(gap_map.get(gas, tuple()))
        if not intervals:
            continue
        labels = [format_coverage_interval(interval, range_unit) for interval in intervals[:4]]
        if len(intervals) > 4:
            labels.append(f"+{len(intervals) - 4} weitere")
        notes.append(f"{display_formula(gas)}: {', '.join(labels)}")
    if not notes:
        return ""
    return f" {intro} " + " | ".join(notes) + "."


def _source_detail_text(gas: str, details: dict[str, Any], range_unit: str) -> str:
    source_kind = str(details.get("source", ""))
    ranges_cm1 = details.get("coverage_ranges_cm1", [])
    ranges = [tuple(interval[:2]) for interval in ranges_cm1 if isinstance(interval, list) and len(interval) >= 2]
    coverage_text = ", ".join(format_coverage_interval(interval, range_unit) for interval in ranges[:3]) if ranges else "-"

    if source_kind == "xsc":
        temp_k = details.get("temperature_k")
        pressure_torr = details.get("pressure_torr")
        files = details.get("files", [])
        extra_bits: list[str] = []
        if isinstance(temp_k, (int, float)):
            extra_bits.append(f"T={float(temp_k):.1f} K")
        if isinstance(pressure_torr, (int, float)):
            extra_bits.append(f"p={float(pressure_torr):.1f} Torr")
        if isinstance(files, list) and files:
            extra_bits.append(f"Dateien={len(files)}")
        extra_text = " | " + " | ".join(extra_bits) if extra_bits else ""
        return f"{display_formula_plot(gas)}: XSC{extra_text} | Coverage {coverage_text}"

    if source_kind == "line_cache":
        return f"{display_formula_plot(gas)}: Liniencache | Coverage {coverage_text}"

    if source_kind == "offline_db":
        temp_c = details.get("reference_temperature_c")
        pressure_hpa = details.get("reference_pressure_hpa")
        extra_bits = []
        if isinstance(temp_c, (int, float)):
            extra_bits.append(f"T={float(temp_c):.1f} °C")
        if isinstance(pressure_hpa, (int, float)):
            extra_bits.append(f"p={float(pressure_hpa):.2f} hPa")
        extra_text = " | " + " | ".join(extra_bits) if extra_bits else ""
        return f"{display_formula_plot(gas)}: Offline-DB{extra_text} | Coverage {coverage_text}"

    return f"{display_formula_plot(gas)}: keine Quelle im aktuellen Bereich"


def source_details_panel(
    serialized_result: dict[str, Any] | None,
    visible_gases: list[str] | None,
    range_unit: str,
) -> html.Div:
    if not serialized_result:
        return html.Div(className="source-info-panel", children=[])

    source_details = serialized_result.get("source_details_by_gas", {})
    components = serialized_result.get("components", {})
    visible_set = set(visible_gases or components.keys())
    gases = [gas for gas in sorted(components.keys()) if gas in visible_set]
    rows: list[Any] = []
    for gas in gases:
        details = source_details.get(gas, {"source": "unavailable"})
        rows.append(html.Div(_source_detail_text(gas, details, range_unit), className="source-info-row"))

    return html.Div(
        className="source-info-panel",
        children=[
            html.H4("Datenquelle je Spezies", className="hover-title"),
            html.Div(rows, className="source-info-list"),
        ],
    )


def latest_hitran_cache_age_days() -> float | None:
    headers = list((BASE_DIR / "hitran_cache").glob("*.header"))
    if not headers:
        return None
    newest_mtime = max(path.stat().st_mtime for path in headers)
    return (time.time() - newest_mtime) / 86400.0


def relayout_has_explicit_x_range(relayout_data: dict[str, Any] | None) -> bool:
    if not relayout_data:
        return False
    keys = set(relayout_data.keys())
    return (
        "xaxis.range" in keys
        or ("xaxis.range[0]" in keys and "xaxis.range[1]" in keys)
    )


def cached_hitran_gases() -> list[str]:
    return sorted(
        path.stem
        for path in (BASE_DIR / "hitran_cache").glob("*.header")
        if path.stem in GAS_LIBRARY
    )


def startup_hitran_message() -> str | None:
    cache_age_days = latest_hitran_cache_age_days()
    if cache_age_days is None:
        return "Lokaler HITRAN-Cache wurde noch nicht aufgebaut. Soll jetzt eine HITRAN-Aktualisierung angeboten werden?"
    if cache_age_days > CACHE_STALENESS_DAYS:
        return (
            f"Der lokale HITRAN-Cache ist etwa {cache_age_days:.0f} Tage alt. "
            "Es koennte neuere HITRAN-Daten geben. Soll der lokale Cache jetzt aktualisiert werden?"
        )
    return None


def offline_mode_enabled(selection: list[str] | None) -> bool:
    return OFFLINE_DB_MODE in (selection or [])


def format_file_size(size_bytes: int | float) -> str:
    size = float(size_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    precision = 0 if unit_index == 0 else 1 if size >= 10 else 2
    return f"{size:.{precision}f} {units[unit_index]}"


def molar_fraction_to_value_unit(molar_fraction: float) -> tuple[float, str]:
    value = max(float(molar_fraction), 0.0)
    if value >= 1.0e-2:
        return value * 100.0, "%"
    if value >= 1.0e-6:
        return value * 1.0e6, "ppm"
    return value * 1.0e9, "ppb"


def manual_cache_option_label(entry: dict[str, Any]) -> str:
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    timestamp = str(entry.get("updated_at", ""))
    step = _safe_float(entry.get("step_cm1", 0.0), 0.0)
    temperature_c = _safe_float(entry.get("temperature_c", 0.0), 0.0)
    pressure_hpa = _safe_float(entry.get("pressure_hpa", 0.0), 0.0)
    points = _safe_int(entry.get("point_count", 0), 0)
    return (
        f"{timestamp} | T={temperature_c:.1f} °C | p={pressure_hpa:.2f} hPa | "
        f"Schritt={step:.4f} cm⁻¹ | {points} Punkte"
    )


def offline_db_state() -> tuple[bool, str, str]:
    try:
        summary = offline_library_summary()
    except Exception:
        return False, "(Offline-DB nicht vorhanden)", LIVE_DB_MODE

    updated_at_raw = summary.get("updated_at")
    try:
        updated_at = datetime.fromisoformat(str(updated_at_raw)).strftime("%d.%m.%Y") if updated_at_raw else "unbekannt"
    except ValueError:
        updated_at = str(updated_at_raw)

    step_cm1 = float(summary.get("native_step_cm1", DEFAULT_MANUAL_STEP_CM1))
    temperature_c = float(summary.get("reference_temperature_c", PICKLE_REBUILD_TEMPERATURE_C))
    pressure_hpa = float(summary.get("reference_pressure_hpa", PICKLE_REBUILD_PRESSURE_HPA))
    pkl_size = format_file_size(Path(str(summary.get("path", ""))).stat().st_size)
    meta = (
        f"(Stand {updated_at} | {temperature_c:.1f} °C | {pressure_hpa:.2f} hPa | {step_cm1:.3f} cm⁻¹ | {pkl_size})"
    )
    return True, meta, OFFLINE_DB_MODE


def offline_coverage_label(range_unit: str) -> str | None:
    try:
        summary = offline_library_summary()
    except Exception:
        return None

    coverage_min_cm1 = float(summary.get("coverage_min_cm1", 0.0))
    coverage_max_cm1 = float(summary.get("coverage_max_cm1", 0.0))
    if range_unit == "um":
        coverage_min, coverage_max = normalize_wavelength_window("cm-1", coverage_min_cm1, coverage_max_cm1)
        return f"{coverage_min:.3f}-{coverage_max:.3f} µm"
    return f"{coverage_min_cm1:.2f}-{coverage_max_cm1:.2f} cm⁻¹"


def format_data_source_error(
    exc: Exception,
    data_source: str,
    range_unit: str,
    range_min: float | None,
    range_max: float | None,
) -> str:
    message = str(exc)
    if data_source != OFFLINE_DB_MODE:
        return message

    if "Offline spectra file not found" in message:
        return (
            "Die schnelle Offline-DB ist noch nicht aufgebaut. Im Offline-Modus wird nicht automatisch live nachgeladen. "
            "Bitte zuerst den lokalen HITRAN-Cache aktualisieren oder den Offline-Modus ausschalten."
        )

    if "Offline spectra are missing for:" in message:
        missing = message.split("Offline spectra are missing for:", 1)[1].split(".", 1)[0].strip()
        if "Local HITRAN cache is also missing for:" in message:
            local_missing = message.split("Local HITRAN cache is also missing for:", 1)[1].split(".", 1)[0].strip()
            return (
                f"Die schnelle Offline-DB enthaelt noch keine vorgerechneten Spektren fuer: {missing}. "
                f"Im lokalen HITRAN-Cache fehlen fuer diese Gase aktuell auch die Quelldaten: {local_missing}. "
                "Der letzte manuelle Refresh hat fuer diesen Bereich sehr wahrscheinlich keine HITRAN-Liniendaten geliefert. "
                "Bitte Bereich oder Gaswahl anpassen oder den Offline-Modus ausschalten."
            )

        if "Local HITRAN cache coverage is too small for:" in message:
            local_ranges = message.split("Local HITRAN cache coverage is too small for:", 1)[1].split(".", 1)[0].strip()
            return (
                f"Die schnelle Offline-DB enthaelt noch keine vorgerechneten Spektren fuer: {missing}. "
                f"Die lokalen HITRAN-Tabellen decken den angeforderten Bereich fuer diese Gase noch nicht vollstaendig ab: {local_ranges}. "
                "Bitte den lokalen HITRAN-Cache fuer denselben Bereich erneut aktualisieren oder den Offline-Modus ausschalten."
            )

        return (
            f"Die schnelle Offline-DB enthaelt noch keine vorgerechneten Spektren fuer: {missing}. "
            "Im Offline-Modus wird nicht automatisch live nachgeladen. Bitte den lokalen HITRAN-Cache fuer diese Komponenten aktualisieren oder den Offline-Modus ausschalten."
        )

    if "Requested range is outside offline pickle coverage" in message:
        requested_text = None
        if range_min is not None and range_max is not None:
            requested_min = float(min(range_min, range_max))
            requested_max = float(max(range_min, range_max))
            if range_unit == "um":
                requested_text = f"{requested_min:.3f}-{requested_max:.3f} µm"
            else:
                requested_text = f"{requested_min:.2f}-{requested_max:.2f} cm⁻¹"
        coverage_text = offline_coverage_label(range_unit)
        requested_clause = f"Der gewaehlte Bereich {requested_text} " if requested_text else "Der gewaehlte Bereich "
        coverage_clause = f"liegt ausserhalb der schnellen Offline-DB ({coverage_text}). " if coverage_text else "liegt ausserhalb der schnellen Offline-DB. "
        return (
            requested_clause
            + coverage_clause
            + "Im Offline-Modus wird nicht automatisch live nachgeladen. Bitte Bereich anpassen, den lokalen HITRAN-Cache fuer diesen Bereich aktualisieren oder den Offline-Modus ausschalten."
        )

    return message


def parameter_field(label: Any, component: Any) -> html.Div:
    return html.Div(
        className="field-block",
        children=[
            html.Span(label, className="field-label"),
            component,
        ],
    )


def parse_required_number(value: float | int | None, label: str) -> float:
    if value in (None, ""):
        raise ValueError(f"Bitte einen Wert für {label} eingeben.")
    return float(value)


def sanitize_pas_sigma(value: float | int | None) -> float:
    if value in (None, ""):
        return PAS_REFERENCE_SIGMA_NV
    return max(float(value), 0.0)


def sanitize_pas_popt(value: float | int | None) -> float:
    if value in (None, ""):
        return PAS_REFERENCE_POPT_MW
    return max(float(value), 1.0e-12)


def sanitize_pas_eps(value: float | int | None) -> float:
    if value in (None, ""):
        return PAS_REFERENCE_EPS
    return min(1.0, max(0.0, float(value)))


def compute_pas_delta_alpha_min(
    sigma_3nv: float | int | None,
    p_opt_mw: float | int | None,
    eps_value: float | int | None,
) -> float:
    sigma = sanitize_pas_sigma(sigma_3nv)
    p_opt = sanitize_pas_popt(p_opt_mw)
    eps = sanitize_pas_eps(eps_value)
    return (
        PAS_REFERENCE_DELTA_ALPHA_MIN
        * (sigma / PAS_REFERENCE_SIGMA_NV)
        * (PAS_REFERENCE_POPT_MW / p_opt)
        * (eps / PAS_REFERENCE_EPS)
    )


def pas_signature(serialized_result: dict[str, Any] | None, visible_gases: list[str] | None) -> str:
    if not serialized_result:
        return "empty"
    gases = sorted(visible_gases or list((serialized_result.get("components") or {}).keys()))
    render_revision = serialized_result.get("render_revision")
    point_count = len(serialized_result.get("wavelength_um") or [])
    return f"{render_revision}|{point_count}|{'/'.join(gases)}"


def selected_total_alpha(
    serialized_result: dict[str, Any],
    visible_gases: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = deserialize_manual_result(serialized_result)
    visible_set = {
        gas
        for gas in (visible_gases or list(result.components.keys()))
        if gas in result.components
    }
    if not visible_set:
        visible_set = set(result.components.keys())
    total_alpha = np.zeros_like(result.total_alpha_per_cm, dtype=float)
    for gas in visible_set:
        total_alpha += result.components[gas].alpha_per_cm
    return result.wavelength_um, result.wavenumber_cm1, total_alpha


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - float(target))))


def detect_peak_region(
    x_values: np.ndarray,
    y_values: np.ndarray,
    probe_index: int,
) -> dict[str, Any] | None:
    if x_values.size < 7 or y_values.size != x_values.size:
        return None
    index = int(min(max(probe_index, 0), x_values.size - 1))
    smooth_window = 7
    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    y_smooth = np.convolve(y_values, kernel, mode="same")
    local_span = min(max(12, x_values.size // 250), 80)
    local_start = max(0, index - local_span)
    local_end = min(x_values.size - 1, index + local_span)
    apex_index = int(local_start + np.argmax(y_smooth[local_start : local_end + 1]))

    max_span = max(30, int(x_values.size * 0.12))
    left_index = apex_index
    while left_index > 1 and (apex_index - left_index) < max_span:
        if y_smooth[left_index - 1] <= y_smooth[left_index]:
            left_index -= 1
            continue
        break

    right_index = apex_index
    while right_index < x_values.size - 2 and (right_index - apex_index) < max_span:
        if y_smooth[right_index + 1] <= y_smooth[right_index]:
            right_index += 1
            continue
        break

    if left_index >= apex_index or right_index <= apex_index:
        return None

    baseline_at_apex = float(np.interp(apex_index, [left_index, right_index], [y_values[left_index], y_values[right_index]]))
    delta_alpha = max(float(y_values[apex_index]) - baseline_at_apex, 0.0)
    if not np.isfinite(delta_alpha) or delta_alpha <= 0.0:
        return None

    return {
        "left": int(left_index),
        "right": int(right_index),
        "apex": int(apex_index),
        "delta_alpha": float(delta_alpha),
        "apex_alpha": float(y_values[apex_index]),
    }


def local_maxima_indices(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.asarray([], dtype=int)
    left = values[1:-1] > values[:-2]
    right = values[1:-1] >= values[2:]
    return np.flatnonzero(left & right) + 1


def peak_from_interaction(
    interaction_data: dict[str, Any] | None,
    serialized_result: dict[str, Any] | None,
    visible_gases: list[str] | None,
    x_unit: str,
) -> dict[str, Any] | None:
    if not interaction_data or not serialized_result:
        return None
    points = interaction_data.get("points") or []
    if not points:
        return None
    point = points[0] or {}
    x_value = interaction_x_value(point, x_unit)
    if x_value is None:
        return None

    result = deserialize_manual_result(serialized_result)
    visible_set = {
        gas
        for gas in (visible_gases or list(result.components.keys()))
        if gas in result.components
    }
    if not visible_set:
        visible_set = set(result.components.keys())

    wavelength_um, wavenumber_cm1, total_alpha = selected_total_alpha(serialized_result, list(visible_set))
    x_values = wavenumber_cm1 if x_unit == "cm-1" else wavelength_um
    try:
        probe_index = nearest_index(x_values, float(x_value))
    except Exception:
        return None
    peak = detect_peak_region(x_values, total_alpha, probe_index)
    if not peak:
        return None

    preferred_gas = preferred_gas_from_interaction_point(point, result, visible_set)
    return annotate_peak_metadata(result, visible_set, peak, preferred_gas)


def interaction_x_value(point: dict[str, Any], x_unit: str) -> float | None:
    x_value = point.get("x")
    if x_value not in (None, ""):
        try:
            return float(x_value)
        except Exception:
            return None

    customdata = point.get("customdata")
    if isinstance(customdata, (list, tuple)):
        try:
            if x_unit == "cm-1" and len(customdata) > 1:
                return float(customdata[1])
            if x_unit != "cm-1" and len(customdata) > 0:
                return float(customdata[0]) / 1000.0
        except Exception:
            return None
    return None


def preferred_gas_from_interaction_point(
    point: dict[str, Any],
    result: ManualSpectrumResult,
    visible_set: set[str],
) -> str | None:
    curve_number = point.get("curveNumber")
    if not isinstance(curve_number, (int, float)):
        return None
    gas_traces = [gas for gas in result.components.keys() if gas in visible_set]
    curve_index = int(curve_number)
    if curve_index < 0 or curve_index >= len(gas_traces):
        return None
    return gas_traces[curve_index]


def annotate_peak_metadata(
    result: ManualSpectrumResult,
    visible_set: set[str],
    peak: dict[str, Any],
    preferred_gas: str | None = None,
) -> dict[str, Any]:
    left_index = int(peak["left"])
    right_index = int(peak["right"])
    apex_index = int(peak["apex"])
    dominant_gas = ""
    dominant_delta_alpha = -np.inf
    dominant_alpha = -np.inf
    dominant_concentration_ppbv = float("nan")
    concentration_by_gas_ppbv: dict[str, float] = {}
    for gas in visible_set:
        component = result.components[gas]
        current_alpha = float(component.alpha_per_cm[apex_index])
        baseline_alpha = float(
            np.interp(
                apex_index,
                [left_index, right_index],
                [component.alpha_per_cm[left_index], component.alpha_per_cm[right_index]],
            )
        )
        current_delta_alpha = max(current_alpha - baseline_alpha, 0.0)
        concentration_by_gas_ppbv[gas] = float(component.concentration) * 1.0e9
        if current_delta_alpha > dominant_delta_alpha:
            dominant_delta_alpha = current_delta_alpha
            dominant_alpha = current_alpha
            dominant_gas = gas

    if preferred_gas and preferred_gas in concentration_by_gas_ppbv:
        dominant_gas = preferred_gas
        dominant_concentration_ppbv = concentration_by_gas_ppbv[preferred_gas]
    else:
        if not dominant_gas:
            for gas in visible_set:
                component = result.components[gas]
                current_alpha = float(component.alpha_per_cm[apex_index])
                if current_alpha > dominant_alpha:
                    dominant_alpha = current_alpha
                    dominant_gas = gas
        dominant_concentration_ppbv = concentration_by_gas_ppbv.get(dominant_gas, float("nan"))

    peak["dominant_gas"] = dominant_gas
    peak["dominant_concentration_ppbv"] = (
        float(dominant_concentration_ppbv) if np.isfinite(dominant_concentration_ppbv) and dominant_concentration_ppbv > 0 else float("nan")
    )
    return peak


def peak_from_x_window(
    relayout_data: dict[str, Any] | None,
    serialized_result: dict[str, Any] | None,
    visible_gases: list[str] | None,
    x_unit: str,
) -> dict[str, Any] | None:
    if not relayout_data or not serialized_result:
        return None
    x_range = extract_axis_range(relayout_data, "xaxis")
    if not x_range or len(x_range) != 2:
        return None

    return peak_from_x_bounds(float(x_range[0]), float(x_range[1]), serialized_result, visible_gases, x_unit)


def peak_from_x_bounds(
    x_start: float,
    x_end: float,
    serialized_result: dict[str, Any] | None,
    visible_gases: list[str] | None,
    x_unit: str,
) -> dict[str, Any] | None:
    if not serialized_result:
        return None

    result = deserialize_manual_result(serialized_result)
    visible_set = {
        gas
        for gas in (visible_gases or list(result.components.keys()))
        if gas in result.components
    }
    if not visible_set:
        visible_set = set(result.components.keys())

    wavelength_um, wavenumber_cm1, total_alpha = selected_total_alpha(serialized_result, list(visible_set))
    x_values = wavenumber_cm1 if x_unit == "cm-1" else wavelength_um
    lower = min(float(x_start), float(x_end))
    upper = max(float(x_start), float(x_end))
    mask = (x_values >= lower) & (x_values <= upper)
    if int(np.count_nonzero(mask)) < 4:
        return None
    in_window_indices = np.flatnonzero(mask)
    local_maxima = local_maxima_indices(total_alpha)
    local_maxima_in_window = local_maxima[mask[local_maxima]] if local_maxima.size else np.asarray([], dtype=int)

    # Prefer true local maxima inside the selected bounds to avoid monotonic background trends.
    if local_maxima_in_window.size:
        probe_index = int(local_maxima_in_window[np.argmax(total_alpha[local_maxima_in_window])])
    else:
        return None
    peak = detect_peak_region(x_values, total_alpha, probe_index)
    if not peak:
        return None

    peak_left_x = float(x_values[int(peak["left"])])
    peak_right_x = float(x_values[int(peak["right"])])
    if max(peak_left_x, peak_right_x) < lower or min(peak_left_x, peak_right_x) > upper:
        return None
    return annotate_peak_metadata(result, visible_set, peak)


def format_pas_delta_alpha(value: float) -> str:
    return f"{float(value):.1E}"


def format_pas_lod(delta_alpha_min: float, selected_peak: dict[str, Any] | None) -> str:
    if not selected_peak:
        return ""
    delta_alpha_peak = float(selected_peak.get("delta_alpha", 0.0))
    if not np.isfinite(delta_alpha_peak) or delta_alpha_peak <= 0.0:
        return ""
    simulated_concentration_ppbv = float(selected_peak.get("dominant_concentration_ppbv", float("nan")))
    if np.isfinite(simulated_concentration_ppbv) and simulated_concentration_ppbv > 0.0:
        lod_ppbv = simulated_concentration_ppbv * float(delta_alpha_min) / delta_alpha_peak
    else:
        lod_ppbv = PAS_REFERENCE_CONCENTRATION_PPBV * float(delta_alpha_min) / delta_alpha_peak
    return f"{lod_ppbv:.3g}"


def add_pas_overlay_traces(
    figure: go.Figure,
    x_values: np.ndarray,
    total_y: np.ndarray,
    peak: dict[str, Any] | None,
    log_y: bool,
    log_floor_value: float,
    *,
    color: str,
    fill_color: str | None,
    line_width: float,
) -> None:
    if not peak:
        return
    left = int(peak.get("left", -1))
    right = int(peak.get("right", -1))
    apex = int(peak.get("apex", -1))
    if left < 0 or right >= len(total_y) or left >= right or apex < left or apex > right:
        return

    indices = np.arange(left, right + 1, dtype=int)
    baseline = np.interp(indices, [left, right], [total_y[left], total_y[right]])
    peak_segment = np.asarray(total_y[indices], dtype=float)
    if log_y:
        baseline_plot = baseline.clip(min=log_floor_value)
        peak_plot = np.maximum(peak_segment, baseline).clip(min=log_floor_value)
    else:
        baseline_plot = baseline
        peak_plot = np.maximum(peak_segment, baseline)

    figure.add_trace(
        go.Scatter(
            x=x_values[indices],
            y=baseline_plot,
            mode="lines",
            line={"color": color, "width": 1.3, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values[indices],
            y=peak_plot,
            mode="lines",
            line={"color": color, "width": line_width},
            fill="tonexty" if fill_color else None,
            fillcolor=fill_color,
            hoverinfo="skip",
            showlegend=False,
        )
    )


def concentration_row(prefix: str, gas: str, value: float, unit: str, visible: bool = True) -> html.Div:
    color = GAS_LIBRARY[gas].get("plot_color", GAS_LIBRARY[gas]["color"])
    return html.Div(
        id=f"{prefix}-concentration-row-{gas}",
        className="control-card" if visible else "control-card is-hidden",
        style={"borderLeft": f"4px solid {color}"},
        children=[
            html.Div(
                className="control-row-title",
                children=[
                    html.Span(gas, className="gas-pill", style={"backgroundColor": color}),
                    html.Span(GAS_LIBRARY[gas]["label"], className="gas-label"),
                ],
            ),
            html.Div(
                className="control-grid compact",
                children=[
                    dcc.Input(
                        id=f"{prefix}-concentration-value-{gas}",
                        type="number",
                        value=value,
                        debounce=True,
                        persistence=True,
                        persistence_type="local",
                        className="number-input",
                    ),
                    dcc.Dropdown(
                        id=f"{prefix}-concentration-unit-{gas}",
                        options=UNIT_OPTIONS,
                        value=unit,
                        clearable=False,
                        persistence=True,
                        persistence_type="local",
                        className="mini-dropdown",
                    ),
                ],
            ),
        ],
    )


def controls_section(title: str, description: str, children: list[Any]) -> html.Div:
    content: list[Any] = [html.H3(title, className="section-title")]
    if description:
        content.append(html.P(description, className="section-copy"))
    content.extend(children)
    return html.Div(
        className="panel-section",
        children=content,
    )


def pas_lod_controls(prefix: str) -> html.Div:
    return html.Div(
        className="pas-lod-panel",
        children=[
            html.Div(
                className="pas-lod-top-row",
                children=[
                    html.Button("PAS LoD Berechnung WM", id=f"{prefix}-pas-arm", className="secondary-button pas-lod-button"),
                    html.Span(id=f"{prefix}-pas-prompt", className="pas-lod-prompt"),
                ],
            ),
            html.Div(
                className="pas-lod-grid",
                children=[
                    parameter_field(
                        "3 σ [nV]",
                        dcc.Input(
                            id=f"{prefix}-pas-3sigma",
                            type="number",
                            value=PAS_REFERENCE_SIGMA_NV,
                            className="number-input",
                            debounce=False,
                        ),
                    ),
                    parameter_field(
                        html.Span(["P", html.Sub("opt"), " [mW]"]),
                        dcc.Input(
                            id=f"{prefix}-pas-popt",
                            type="number",
                            value=PAS_REFERENCE_POPT_MW,
                            className="number-input",
                            debounce=False,
                        ),
                    ),
                    parameter_field(
                        "ε",
                        dcc.Input(
                            id=f"{prefix}-pas-eps",
                            type="number",
                            value=PAS_REFERENCE_EPS,
                            min=0,
                            max=1,
                            step="any",
                            className="number-input",
                            debounce=False,
                        ),
                    ),
                    parameter_field(
                        html.Span(["Δα", html.Sub("min"), " [cm⁻¹]"]),
                        dcc.Input(
                            id=f"{prefix}-pas-delta-alpha-min",
                            type="text",
                            value="1.8E-7",
                            readOnly=True,
                            className="number-input",
                        ),
                    ),
                    parameter_field(
                        "LoD [ppbV]",
                        dcc.Input(
                            id=f"{prefix}-pas-lod",
                            type="text",
                            value="",
                            readOnly=True,
                            className="number-input",
                        ),
                    ),
                ],
            ),
            dcc.Store(id=f"{prefix}-pas-state", data={"armed": False, "signature": "empty", "selected": None, "hover": None}),
        ],
    )


def visible_row_classes(selected_gases: list[str] | None) -> list[str]:
    selected_set = set(selected_gases or [])
    return [
        "control-card" if gas in selected_set else "control-card is-hidden"
        for gas in ALL_GASES
    ]


def hero_logo() -> html.Div:
    return html.Div(
        className="hero-logo",
        children=[
            html.Img(
                src=LOGO_ASSET_PATH,
                className="hero-logo-image",
                alt="TomExplorer brand mark with molecule, spectrum, and word mark",
            ),
        ],
    )


def make_axis_config(x_values: list[float]) -> tuple[list[float], list[str]]:
    if not x_values:
        return [], []
    count = min(7, max(3, len(x_values) // 800))
    tick_values = [x_values[0] + idx * (x_values[-1] - x_values[0]) / (count - 1) for idx in range(count)]
    tick_text = [f"{1.0e4 / tick:.1f}" for tick in tick_values]
    return tick_values, tick_text


def axis_labels_for_unit(range_unit: str) -> tuple[str, str]:
    if range_unit == "cm-1":
        return "ν Minimum [cm⁻¹]", "ν Maximum [cm⁻¹]"
    return "λ Minimum [µm]", "λ Maximum [µm]"


def round_range_value(range_unit: str, value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6 if range_unit == "um" else 3)


def convert_range_inputs(
    previous_unit: str,
    next_unit: str,
    range_min: float | None,
    range_max: float | None,
) -> tuple[float | None, float | None]:
    if previous_unit == next_unit:
        return range_min, range_max

    converted_min: float | None = None
    converted_max: float | None = None
    if previous_unit == "um" and next_unit == "cm-1":
        if range_max not in (None, ""):
            converted_min = float(wavelength_um_to_wavenumber_cm1(float(range_max)))
        if range_min not in (None, ""):
            converted_max = float(wavelength_um_to_wavenumber_cm1(float(range_min)))
    elif previous_unit == "cm-1" and next_unit == "um":
        if range_max not in (None, ""):
            converted_min = float(wavenumber_cm1_to_wavelength_um(float(range_max)))
        if range_min not in (None, ""):
            converted_max = float(wavenumber_cm1_to_wavelength_um(float(range_min)))
    else:
        converted_min = range_min
        converted_max = range_max

    return round_range_value(next_unit, converted_min), round_range_value(next_unit, converted_max)


def spectrum_x_values(result: Any, x_unit: str) -> np.ndarray:
    return result.wavenumber_cm1 if x_unit == "cm-1" else result.wavelength_um


def default_x_range(result: Any, x_unit: str) -> list[float]:
    x_values = spectrum_x_values(result, x_unit)
    if x_unit == "cm-1":
        return [float(np.max(x_values)), float(np.min(x_values))]
    return [float(np.min(x_values)), float(np.max(x_values))]


def secondary_axis_config(x_unit: str, x_values: np.ndarray) -> tuple[str, list[float], list[str]]:
    if x_values.size == 0:
        return ("Wellenlänge λ [µm]" if x_unit == "cm-1" else "Wellenzahl ν [cm⁻¹]", [], [])

    count = min(7, max(3, x_values.size // 800))
    tick_values = np.linspace(float(np.min(x_values)), float(np.max(x_values)), count).tolist()
    if x_unit == "cm-1":
        tick_labels = [f"{float(wavenumber_cm1_to_wavelength_um(tick)):.4f}" for tick in tick_values]
        return "Wellenlänge λ [µm]", tick_values, tick_labels

    tick_labels = [f"{float(wavelength_um_to_wavenumber_cm1(tick)):.1f}" for tick in tick_values]
    return "Wellenzahl ν [cm⁻¹]", tick_values, tick_labels


def extract_axis_range(
    relayout_data: dict[str, Any] | None,
    axis_name: str,
) -> list[float] | None:
    if relayout_data:
        autorange_key = f"{axis_name}.autorange"
        if relayout_data.get(autorange_key):
            return None

        start_key = f"{axis_name}.range[0]"
        end_key = f"{axis_name}.range[1]"
        if start_key in relayout_data and end_key in relayout_data:
            return [float(relayout_data[start_key]), float(relayout_data[end_key])]

        raw_range = relayout_data.get(f"{axis_name}.range")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            return [float(raw_range[0]), float(raw_range[1])]
    return None


def preserve_manual_ranges(
    figure_state: dict[str, Any] | None,
    relayout_data: dict[str, Any] | None,
    target_log_y: bool,
    target_y_mode: str,
    target_x_unit: str,
    target_render_revision: int | None,
) -> tuple[list[float] | None, list[float] | None]:
    layout = ((figure_state or {}).get("layout", {}) or {})
    meta = (layout.get("meta", {}) or {}) if isinstance(layout, dict) else {}
    current_render_revision = meta.get("render_revision")
    current_y_mode = meta.get("y_mode")
    current_x_unit = meta.get("x_unit", "um")
    if current_render_revision != target_render_revision:
        return None, None

    x_range = None
    if current_x_unit == target_x_unit:
        x_range = extract_axis_range(relayout_data, "xaxis")
    y_range = extract_axis_range(relayout_data, "yaxis")
    if x_range is None and current_x_unit == target_x_unit:
        current_x_range = ((layout.get("xaxis", {}) or {}).get("range"))
        if isinstance(current_x_range, (list, tuple)) and len(current_x_range) == 2:
            x_range = [float(current_x_range[0]), float(current_x_range[1])]

    if y_range:
        current_y_type = (layout.get("yaxis", {}) or {}).get("type", "linear")
        current_log_y = current_y_type == "log"
        if current_log_y == target_log_y and current_y_mode == target_y_mode:
            return x_range, y_range
    return x_range, None


def current_manual_x_range(
    figure_state: dict[str, Any] | None,
    relayout_data: dict[str, Any] | None,
    target_x_unit: str,
    target_render_revision: int | None,
) -> list[float] | None:
    layout = ((figure_state or {}).get("layout", {}) or {})
    meta = (layout.get("meta", {}) or {}) if isinstance(layout, dict) else {}
    current_render_revision = meta.get("render_revision")
    current_x_unit = meta.get("x_unit", "um")
    if current_render_revision != target_render_revision or current_x_unit != target_x_unit:
        return None

    x_range = extract_axis_range(relayout_data, "xaxis")
    if x_range is not None:
        return x_range

    current_x_range = ((layout.get("xaxis", {}) or {}).get("range"))
    if isinstance(current_x_range, (list, tuple)) and len(current_x_range) == 2:
        return [float(current_x_range[0]), float(current_x_range[1])]
    return None


def build_manual_export_csv(
    serialized_result: dict[str, Any],
    x_unit: str,
    x_range: list[float] | None,
) -> str:
    result = deserialize_manual_result(serialized_result)
    x_values = spectrum_x_values(result, x_unit)
    visible_mask = visible_slice_mask(x_values, x_range)

    visible_wavelength_um = result.wavelength_um[visible_mask]
    visible_wavenumber_cm1 = result.wavenumber_cm1[visible_mask]
    visible_total_sigma = result.total_sigma_cm2_per_molecule[visible_mask]
    visible_total_alpha = result.total_alpha_per_cm[visible_mask]
    visible_components = {
        gas: {
            "sigma": component.sigma_cm2_per_molecule[visible_mask],
            "alpha": component.alpha_per_cm[visible_mask],
            "concentration": component.concentration,
        }
        for gas, component in sorted(result.components.items())
    }

    x_lower = float(np.min(x_values[visible_mask]))
    x_upper = float(np.max(x_values[visible_mask]))
    analyte_summary = "; ".join(
        f"{gas}={format_concentration(component['concentration'])}"
        for gas, component in visible_components.items()
    )

    buffer = io.StringIO()
    buffer.write("# TomExplorer CSV Export\n")
    buffer.write(f"# Exportzeit: {datetime.now().isoformat(timespec='seconds')}\n")
    buffer.write(f"# Analyte und Konzentrationen: {analyte_summary or '-'}\n")
    buffer.write(f"# T [degC]: {result.temperature_c:.6g}\n")
    buffer.write(f"# p [hPa]: {result.pressure_hpa:.6g}\n")
    buffer.write(f"# Schrittweite [cm-1]: {result.step_cm1:.6g}\n")
    buffer.write(f"# Sichtbarer Bereich Wellenlaenge [um]: {float(np.min(visible_wavelength_um)):.10g} bis {float(np.max(visible_wavelength_um)):.10g}\n")
    buffer.write(f"# Sichtbarer Bereich Wellenzahl [cm-1]: {float(np.min(visible_wavenumber_cm1)):.10g} bis {float(np.max(visible_wavenumber_cm1)):.10g}\n")
    buffer.write(
        f"# Sichtbarer Bereich aktuelle x-Achse [{('um' if x_unit == 'um' else 'cm-1')}]: {x_lower:.10g} bis {x_upper:.10g}\n"
    )

    writer = csv.writer(buffer, lineterminator="\n")
    columns = [
        "wavelength_um",
        "wavenumber_cm-1",
        "total_sigma_cm2_per_molecule",
        "total_alpha_per_cm",
    ]
    for gas in visible_components:
        columns.append(f"{gas}_sigma_cm2_per_molecule")
        columns.append(f"{gas}_alpha_per_cm")
    writer.writerow(columns)

    for index in range(len(visible_wavelength_um)):
        row: list[float] = [
            float(visible_wavelength_um[index]),
            float(visible_wavenumber_cm1[index]),
            float(visible_total_sigma[index]),
            float(visible_total_alpha[index]),
        ]
        for gas in visible_components:
            row.append(float(visible_components[gas]["sigma"][index]))
            row.append(float(visible_components[gas]["alpha"][index]))
        writer.writerow(row)
    return buffer.getvalue()


def manual_export_filename(serialized_result: dict[str, Any]) -> str:
    result = deserialize_manual_result(serialized_result)
    gas_label = "-".join(sorted(result.components.keys())[:4]) or "spectrum"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tomexplorer_manual_{gas_label}_{timestamp}.csv"


def auto_linear_y_range(values: np.ndarray) -> list[float]:
    finite_values = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite_values.size == 0:
        return [0.0, 1.0]
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if y_max <= 0:
        return [0.0, 1.0]
    magnitude = max(abs(y_min), abs(y_max), np.finfo(float).tiny)
    padding = max((y_max - y_min) * 0.08, y_max * 0.08, magnitude * 1.0e-6)
    lower = 0.0 if y_min >= 0 else y_min - padding
    upper = y_max + padding
    return [lower, upper]


def visible_slice_mask(x_values: np.ndarray, x_range: list[float] | None) -> np.ndarray:
    if not x_range or len(x_range) != 2:
        return np.ones_like(x_values, dtype=bool)
    lower = min(float(x_range[0]), float(x_range[1]))
    upper = max(float(x_range[0]), float(x_range[1]))
    mask = (x_values >= lower) & (x_values <= upper)
    if np.any(mask):
        return mask
    return np.ones_like(x_values, dtype=bool)


def auto_log_y_range(values: np.ndarray, log_level: int) -> list[float]:
    positive_values = np.asarray(values[values > 0], dtype=float)
    if positive_values.size == 0:
        return [-12.0, 0.0]
    visible_max = float(np.max(positive_values))
    lower = max(visible_max / (10 ** int(log_level)), 1.0e-35)
    upper = max(visible_max * 1.08, lower * 1.0001)
    return [float(np.log10(lower)), float(np.log10(upper))]


def hover_capture_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    customdata: np.ndarray,
    log_y: bool,
    log_floor_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level_count = 14
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))

    if log_y:
        lower = max(log_floor_value, 1.0e-35)
        upper = max(y_max, lower * 1.0001)
        levels = np.geomspace(lower, upper, num=level_count)
    else:
        if np.isclose(y_min, y_max):
            y_span = max(abs(y_max) * 0.05, 1.0e-12)
            levels = np.linspace(y_min, y_max + y_span, num=level_count)
        else:
            levels = np.linspace(y_min, y_max, num=level_count)

    x_grid = np.tile(x_values, level_count)
    y_grid = np.repeat(levels, len(x_values))
    custom_grid = np.tile(customdata, (level_count, 1))
    return x_grid, y_grid, custom_grid


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffaf2",
        plot_bgcolor="#fffdf8",
        font={"family": BODY_FONT, "size": 14, "color": "#1f2937"},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 18, "family": DISPLAY_FONT},
            }
        ],
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return figure


def make_spectrum_figure(
    serialized_result: dict[str, Any],
    y_mode: str,
    log_y: bool,
    title: str,
    x_unit: str = "um",
    highlighted_windows: list[dict[str, float]] | None = None,
    highlighted_lines: list[dict[str, Any]] | None = None,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    log_level: int = 5,
    revision_key: str | None = None,
    preserve_ui_state: bool = True,
    visible_gases: list[str] | None = None,
    pas_state: dict[str, Any] | None = None,
) -> go.Figure:
    result = deserialize_manual_result(serialized_result)
    visible_set = {
        gas
        for gas in (visible_gases or list(result.components.keys()))
        if gas in result.components
    }
    if not visible_set:
        visible_set = set(result.components.keys())
    selected_components = [
        (gas, component)
        for gas, component in result.components.items()
        if gas in visible_set
    ]
    render_revision = serialized_result.get("render_revision")
    x_values = spectrum_x_values(result, x_unit)
    secondary_title, secondary_ticks, secondary_labels = secondary_axis_config(x_unit, x_values)
    customdata = np.column_stack((result.wavelength_um * 1000.0, result.wavenumber_cm1))
    total_y = np.zeros_like(result.wavenumber_cm1, dtype=float)
    for _gas, component in selected_components:
        total_y += component.alpha_per_cm if y_mode == "alpha" else component.sigma_cm2_per_molecule
    y_title = "Alpha [1/cm]" if y_mode == "alpha" else "Sigma [cm²/Molekül]"
    positive_total = total_y[total_y > 0]
    log_floor_value = 1.0e-35
    if log_y and positive_total.size:
        log_floor_value = max(float(np.max(positive_total)) / (10 ** int(log_level)), 1.0e-35)
    visible_mask = visible_slice_mask(x_values, x_range)
    visible_total = total_y[visible_mask]
    revision_suffix = f":{revision_key}" if revision_key else ""
    layout_uirevision = f"manual:{render_revision}:{x_unit}:{y_mode}:{'log' if log_y else 'linear'}:{int(log_level)}{revision_suffix}" if preserve_ui_state else None
    xaxis_uirevision = f"manual-x:{render_revision}:{x_unit}{revision_suffix}" if preserve_ui_state else None
    yaxis_uirevision = f"manual-y:{render_revision}:{y_mode}:{'log' if log_y else 'linear'}{revision_suffix}" if preserve_ui_state else None
    layout_meta = {"render_revision": render_revision, "y_mode": y_mode, "x_unit": x_unit, "revision_key": revision_key} if preserve_ui_state else {"x_unit": x_unit, "revision_key": revision_key}
    figure = go.Figure()

    for gas, component in selected_components:
        y_values = component.alpha_per_cm if y_mode == "alpha" else component.sigma_cm2_per_molecule
        if log_y:
            y_values = y_values.clip(min=log_floor_value)
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                customdata=customdata,
                mode="lines",
                name=f"{display_formula(gas)} [{format_concentration(component.concentration)}]",
                line={"color": component.color, "width": 1.8},
                hoverinfo="skip",
                hovertemplate=None,
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=total_y.clip(min=log_floor_value) if log_y else total_y,
            customdata=customdata,
            mode="lines",
            name="Einhüllende",
            line={"color": "#000000", "width": 1.2, "dash": "1px,2px"},
            hoverinfo="skip",
            hovertemplate=None,
            showlegend=False,
        )
    )

    if pas_state:
        add_pas_overlay_traces(
            figure,
            x_values,
            total_y,
            pas_state.get("hover"),
            log_y,
            log_floor_value,
            color="rgba(220, 38, 38, 0.92)",
            fill_color=None,
            line_width=2.0,
        )
        add_pas_overlay_traces(
            figure,
            x_values,
            total_y,
            pas_state.get("selected"),
            log_y,
            log_floor_value,
            color="rgba(220, 38, 38, 0.98)",
            fill_color="rgba(220, 38, 38, 0.24)",
            line_width=2.2,
        )

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=total_y.clip(min=log_floor_value) if log_y else total_y,
            customdata=customdata,
            mode="lines",
            line={"color": "rgba(0,0,0,0)", "width": 24},
            hovertemplate="<extra></extra>",
            showlegend=False,
            name="hover-capture",
        )
    )

    capture_x, capture_y, capture_customdata = hover_capture_grid(
        x_values,
        total_y.clip(min=log_floor_value) if log_y else total_y,
        customdata,
        log_y,
        log_floor_value,
    )
    figure.add_trace(
        go.Scatter(
            x=capture_x,
            y=capture_y,
            customdata=capture_customdata,
            mode="markers",
            marker={"size": 10, "color": "rgba(15, 23, 42, 0.003)"},
            hovertemplate="<extra></extra>",
            showlegend=False,
            name="hover-capture-grid",
        )
    )

    if highlighted_windows:
        for index, window in enumerate(highlighted_windows, start=1):
            figure.add_vrect(
                x0=min(float(window["x_min"]), float(window["x_max"])),
                x1=max(float(window["x_min"]), float(window["x_max"])),
                fillcolor="#f59e0b",
                opacity=0.12,
                line_width=0,
                annotation_text=f"Laser {index}",
                annotation_position="top left",
            )

    if highlighted_lines:
        for line in highlighted_lines:
            line_gas = str(line.get("gas", ""))
            if line_gas and line_gas not in visible_set:
                continue
            figure.add_vline(
                x=float(line["x_value"]),
                line_width=1.8,
                line_dash="dot",
                line_color=str(line["color"]),
                opacity=0.95,
                annotation_text=str(line["label"]),
                annotation_position="top",
                annotation_font={"color": str(line["color"]), "size": 11},
            )

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffaf2",
        plot_bgcolor="#fffdf8",
        title={
            "text": title,
            "font": {"family": DISPLAY_FONT, "size": 21},
            "x": 0.02,
            "xanchor": "left",
        },
        font={"family": BODY_FONT, "size": 14, "color": "#1f2937"},
        hovermode="x unified",
        clickmode="event+select",
        dragmode="zoom",
        hoverdistance=-1,
        spikedistance=-1,
        showlegend=False,
        margin={"l": 60, "r": 24, "t": 84, "b": 58},
        uirevision=layout_uirevision,
        meta=layout_meta,
        xaxis={
            "title": "Wellenzahl ν [cm⁻¹]" if x_unit == "cm-1" else "Wellenlänge λ [µm]",
            "autorange": "reversed" if x_unit == "cm-1" else True,
            "showgrid": True,
            "gridcolor": "#eadfc9",
            "zeroline": False,
            "showspikes": True,
            "spikemode": "across",
            "spikecolor": "rgba(31, 59, 47, 0.35)",
            "spikethickness": 1,
            "uirevision": xaxis_uirevision,
        },
        xaxis2={
            "title": secondary_title,
            "overlaying": "x",
            "side": "top",
            "tickmode": "array",
            "tickvals": secondary_ticks,
            "ticktext": secondary_labels,
        },
        yaxis={
            "title": y_title,
            "type": "log" if log_y else "linear",
            "rangemode": "normal" if log_y else "tozero",
            "showgrid": True,
            "gridcolor": "#eadfc9",
            "zeroline": False,
            "showexponent": "last",
            "exponentformat": "power",
            "showspikes": True,
            "spikemode": "across",
            "spikecolor": "rgba(31, 59, 47, 0.2)",
            "spikethickness": 1,
            "uirevision": yaxis_uirevision,
        },
    )
    figure.update_xaxes(range=x_range or default_x_range(result, x_unit), autorange=False)
    if y_range:
        figure.update_yaxes(range=y_range)
    elif log_y:
        figure.update_yaxes(range=auto_log_y_range(visible_total, log_level))
    else:
        figure.update_yaxes(range=auto_linear_y_range(visible_total))
    return figure


def hover_panel(payload: dict[str, Any] | None) -> html.Div:
    if not payload:
        return html.Div(
            className="hover-card",
            children=[
                html.H4("Hover-Details", className="hover-title"),
                html.P(
                    "Wellenlänge, Wellenzahl sowie α und σ der Komponenten erscheinen hier beim Hover über dem Spektrum.",
                    className="section-copy",
                ),
            ],
        )

    header = html.Div(
        className="hover-metrics",
        children=[
            html.Div([html.Span("λ"), html.Strong(f"{payload['wavelength_um'] * 1000.0:.3f} nm")]),
            html.Div([html.Span("ν"), html.Strong(f"{payload['wavenumber_cm1']:.3f} cm⁻¹")]),
            html.Div([html.Span("Σ α [1/cm]"), html.Strong(f"{payload['total_alpha_per_cm']:.3e} 1/cm")]),
            html.Div([html.Span("Σ σ [cm²/Molekül]"), html.Strong(f"{payload['total_sigma_cm2_per_molecule']:.3e} cm²/Molekül")]),
        ],
    )
    rows: list[Any] = []
    for gas, component in payload["components"].items():
        rows.append(
            html.Div(
                className="hover-row",
                children=[
                    html.Span(display_formula(gas), className="hover-gas", style={"color": component.get("color", GAS_LIBRARY[gas]["color"])}),
                    html.Span(f"σ [cm²/Molekül] {component['sigma_cm2_per_molecule']:.3e}"),
                    html.Span(f"α [1/cm] {component['alpha_per_cm']:.3e}"),
                    html.Span(format_concentration(component["concentration"])),
                ],
            )
        )

    return html.Div(
        className="hover-card",
        children=[
            html.H4("Hover-Details", className="hover-title"),
            header,
            html.Div(rows, className="hover-table"),
        ],
    )


def collect_concentrations(
    values: list[float],
    units: list[str],
    selected_gases: list[str] | None = None,
) -> dict[str, float]:
    selected_set = set(selected_gases or [])
    if not selected_set:
        return {}
    concentrations: dict[str, float] = {}
    for gas, raw_value, unit in zip(ALL_GASES, values, units):
        if raw_value in (None, ""):
            continue
        if gas not in selected_set:
            continue
        concentrations[gas] = concentration_to_molar_fraction(float(raw_value), unit)
    return concentrations


def format_laser_window_range(window: Any, range_unit: str) -> str:
    if range_unit == "cm-1":
        wavenumber_min = round_range_value("cm-1", wavelength_um_to_wavenumber_cm1(window.wavelength_max_um))
        wavenumber_max = round_range_value("cm-1", wavelength_um_to_wavenumber_cm1(window.wavelength_min_um))
        return f"{wavenumber_min:.3f}-{wavenumber_max:.3f} cm⁻¹ | {window.tuning_span_nm:.2f} nm"
    return f"{window.wavelength_min_um:.4f}-{window.wavelength_max_um:.4f} µm | {window.tuning_span_nm:.2f} nm"


def plan_worst_case_metrics(plan: LaserPlan) -> dict[str, float]:
    metrics = [metric for window in plan.windows for metric in window.gas_metrics.values()]
    if not metrics:
        return {
            "signal_to_interference": 0.0,
            "delta_alpha_selectivity": 0.0,
            "wms2f_selectivity": 0.0,
            "wms2f_shape_similarity": 0.0,
        }
    return {
        "signal_to_interference": min(metric.signal_to_interference for metric in metrics),
        "delta_alpha_selectivity": min(metric.peak_region_delta_alpha_selectivity for metric in metrics),
        "wms2f_selectivity": min(metric.peak_region_wms2f_selectivity for metric in metrics),
        "wms2f_shape_similarity": min(metric.peak_region_wms2f_shape_similarity for metric in metrics),
    }


def format_plan_worst_case_metrics(plan: LaserPlan) -> str:
    metrics = plan_worst_case_metrics(plan)
    return " | ".join(
        [
            f"S/I {metrics['signal_to_interference']:.2f}",
            f"Δα-Sel {metrics['delta_alpha_selectivity']:.2f}",
            f"2f-Sel {metrics['wms2f_selectivity']:.2f}",
            f"2f-Fit {metrics['wms2f_shape_similarity']:.2f}",
        ]
    )


def search_table_rows(plans: list[LaserPlan], range_unit: str) -> list[dict[str, Any]]:
    rows = []
    for plan in plans:
        ranges = " | ".join(format_laser_window_range(window, range_unit) for window in plan.windows)
        rows.append(
            {
                "rank": plan.rank,
                "score": round(plan.score, 1),
                "lasers": len(plan.windows),
                "covered": ", ".join(plan.covered_targets),
                "missing": ", ".join(plan.missing_targets) or "-",
                "ranges": ranges,
                "robustness": format_plan_worst_case_metrics(plan),
            }
        )
    return rows


def empty_search_plan_details() -> html.Div:
    return html.Div(
        className="search-plan-details",
        children=[
            html.Div(
                className="search-plan-summary",
                children=[
                    html.Span(
                        "Nach Auswahl eines Treffers erscheinen hier kompakte Detailkarten pro Laserfenster.",
                        className="section-copy",
                    )
                ],
            )
        ],
    )


def empty_search_window_plots() -> html.Div:
    return html.Div(className="search-window-plot-grid")


def search_plot_x_range(x_values: np.ndarray, x_unit: str, x_min: float, x_max: float) -> list[float]:
    if x_unit == "cm-1":
        center_value = (x_min + x_max) / 2.0
        center_wavelength_um = float(wavenumber_cm1_to_wavelength_um(center_value))
        min_padding = abs(
            float(wavelength_um_to_wavenumber_cm1(center_wavelength_um - SEARCH_PLOT_PADDING_UM))
            - float(wavelength_um_to_wavenumber_cm1(center_wavelength_um + SEARCH_PLOT_PADDING_UM))
        ) / 2.0
    else:
        min_padding = SEARCH_PLOT_PADDING_UM
    zoom_padding = max((x_max - x_min) * 0.08, min_padding)
    x_range = [
        max(float(np.min(x_values)), x_min - zoom_padding),
        min(float(np.max(x_values)), x_max + zoom_padding),
    ]
    if x_unit == "cm-1":
        return [x_range[1], x_range[0]]
    return x_range


def search_plot_y_range(
    result: ManualSpectrumResult,
    x_values: np.ndarray,
    x_range: list[float],
    log_y: bool,
    visible_gases: list[str] | None = None,
) -> list[float] | None:
    mask = (x_values >= min(x_range)) & (x_values <= max(x_range))
    visible_set = {
        gas
        for gas in (visible_gases or list(result.components.keys()))
        if gas in result.components
    }
    if not visible_set:
        visible_set = set(result.components.keys())
    total_visible_alpha = np.zeros_like(result.total_alpha_per_cm, dtype=float)
    for gas in visible_set:
        total_visible_alpha += result.components[gas].alpha_per_cm
    local_alpha = total_visible_alpha[mask]
    if not local_alpha.size or log_y:
        return None
    local_max = float(np.max(local_alpha))
    local_min = float(np.min(local_alpha))
    span = local_max - local_min
    padding = max(span * 0.15, local_max * 0.08, 1.0e-12)
    return [0.0, local_max + padding]


def build_search_window_plots(
    plan: LaserPlan,
    store: dict[str, Any],
    range_unit: str,
    log_y: bool,
    log_level: int,
    result: ManualSpectrumResult,
    fine_step_cm1: float,
    visible_gases: list[str] | None = None,
) -> html.Div:
    serialized_result = serialize_manual_result(result)
    x_values = spectrum_x_values(result, range_unit)
    plot_cards: list[html.Div] = []

    for index, window in enumerate(plan.windows, start=1):
        window_highlight = {
            "x_min": float(wavelength_um_to_wavenumber_cm1(window.wavelength_max_um)) if range_unit == "cm-1" else window.wavelength_min_um,
            "x_max": float(wavelength_um_to_wavenumber_cm1(window.wavelength_min_um)) if range_unit == "cm-1" else window.wavelength_max_um,
        }
        highlighted_lines = [
            {
                "gas": gas,
                "x_value": float(metric.peak_wavenumber_cm1) if range_unit == "cm-1" else metric.peak_wavelength_um,
                "color": result.components[gas].color,
                "label": display_formula(gas),
            }
            for gas, metric in sorted(window.gas_metrics.items(), key=lambda item: item[1].peak_wavelength_um)
        ]
        x_range = search_plot_x_range(
            x_values,
            range_unit,
            min(window_highlight["x_min"], window_highlight["x_max"]),
            max(window_highlight["x_min"], window_highlight["x_max"]),
        )
        y_range = search_plot_y_range(result, x_values, x_range, log_y, visible_gases)
        title = (
            f"Laser {index} | {', '.join(display_formula(gas) for gas in window.coverage)}"
            if window.coverage
            else f"Laser {index}"
        )
        figure = make_spectrum_figure(
            serialized_result,
            y_mode="alpha",
            log_y=log_y,
            log_level=int(log_level),
            title=title,
            x_unit=range_unit,
            highlighted_windows=[window_highlight],
            highlighted_lines=highlighted_lines,
            x_range=x_range,
            y_range=y_range,
            revision_key=f"search-window:{plan.rank}:{window.window_id}:{range_unit}",
            preserve_ui_state=False,
            visible_gases=visible_gases,
        )
        figure.update_layout(height=360, meta={**(figure.layout.meta or {}), "step_cm1": fine_step_cm1})
        plot_cards.append(
            html.Div(
                className="search-window-plot-card",
                children=[
                    dcc.Graph(
                        figure=figure,
                        className="search-window-graph",
                        clear_on_unhover=False,
                    )
                ],
            )
        )

    return html.Div(className="search-window-plot-grid", children=plot_cards)


def build_search_plan_details(plan: LaserPlan, store: dict[str, Any], range_unit: str) -> html.Div:
    target_concentrations = store.get("target_concentrations", {})
    interference_concentrations = store.get("interference_concentrations", {})
    cards: list[html.Div] = []

    for index, window in enumerate(plan.windows, start=1):
        coverage_pills = [
            html.Span(
                display_formula(gas),
                className="search-laser-pill",
                style={
                    "backgroundColor": GAS_LIBRARY[gas].get("plot_color", GAS_LIBRARY[gas]["color"]),
                },
            )
            for gas in window.coverage
        ]
        rows: list[html.Div] = []
        for gas, metric in sorted(window.gas_metrics.items(), key=lambda item: item[1].peak_wavelength_um):
            concentration = target_concentrations.get(gas)
            role_label = "Ziel"
            if concentration is None:
                concentration = interference_concentrations.get(gas, 0.0)
                role_label = "Stoer"
            rows.append(
                html.Div(
                    className="search-laser-metric-row",
                    children=[
                        html.Span(
                            display_formula(gas),
                            className="search-laser-gas",
                            style={"color": GAS_LIBRARY[gas].get("plot_color", GAS_LIBRARY[gas]["color"])} ,
                        ),
                        html.Span(f"{metric.peak_wavelength_um:.4f} um", className="search-laser-metric-text"),
                        html.Span(f"alpha {metric.peak_alpha_per_cm:.2e}", className="search-laser-metric-text"),
                        html.Span(
                            f"S/I {metric.signal_to_interference:.2f} | dA {metric.peak_region_delta_alpha_selectivity:.2f} | 2f {metric.peak_region_wms2f_selectivity:.2f}/{metric.peak_region_wms2f_shape_similarity:.2f}",
                            className="search-laser-metric-text",
                        ),
                        html.Span(f"{role_label} {format_concentration(float(concentration))}", className="search-laser-metric-text search-laser-metric-meta"),
                    ],
                )
            )

        cards.append(
            html.Div(
                className="search-laser-card",
                children=[
                    html.Div(
                        className="search-laser-card-header",
                        children=[
                            html.Div(
                                children=[
                                    html.H4(f"Laser {index}", className="hover-title"),
                                    html.P(format_laser_window_range(window, range_unit), className="section-copy small"),
                                ]
                            ),
                            html.Div(className="search-laser-coverage", children=coverage_pills),
                        ],
                    ),
                    html.Div(className="search-laser-metric-table", children=rows),
                ],
            )
        )

    summary_bits = [
        html.Strong(f"Rang {plan.rank} | Score {plan.score:.1f}"),
        html.Span(f"Abgedeckt: {', '.join(display_formula(gas) for gas in plan.covered_targets)}"),
        html.Span(
            "Fehlt: " + (", ".join(display_formula(gas) for gas in plan.missing_targets) if plan.missing_targets else "-")
        ),
    ]
    return html.Div(
        className="search-plan-details",
        children=[
            html.Div(className="search-plan-summary", children=summary_bits),
            html.Div(className="search-laser-card-grid", children=cards),
        ],
    )


def build_search_store(
    plans: list[LaserPlan],
    serialized_spectrum: dict[str, Any],
    target_concentrations: dict[str, float],
    interference_concentrations: dict[str, float],
    range_unit: str,
    data_source: str,
) -> dict[str, Any]:
    return {
        "plans": [serialize_laser_plan(plan) for plan in plans],
        "spectrum": serialized_spectrum,
        "target_concentrations": target_concentrations,
        "interference_concentrations": interference_concentrations,
        "range_unit": range_unit,
        "data_source": data_source,
    }


def rebuild_selected_search_result(
    store: dict[str, Any],
    plan: LaserPlan,
) -> tuple[Any, float]:
    coarse_result = deserialize_manual_result(store["spectrum"])
    if not plan.windows:
        return coarse_result, coarse_result.step_cm1

    merged_concentrations = {
        **store.get("interference_concentrations", {}),
        **store.get("target_concentrations", {}),
    }
    if not merged_concentrations:
        return coarse_result, coarse_result.step_cm1

    window_min_um = min(window.wavelength_min_um for window in plan.windows)
    window_max_um = max(window.wavelength_max_um for window in plan.windows)
    padding_um = max((window_max_um - window_min_um) * 0.08, SEARCH_PLOT_PADDING_UM)
    local_min_um = max(float(np.min(coarse_result.wavelength_um)), window_min_um - padding_um)
    local_max_um = min(float(np.max(coarse_result.wavelength_um)), window_max_um + padding_um)
    if local_max_um <= local_min_um:
        return coarse_result, coarse_result.step_cm1

    local_nu_min, local_nu_max = normalize_wavenumber_window("um", local_min_um, local_max_um)
    local_span_cm1 = abs(local_nu_max - local_nu_min)
    fine_step_cm1 = max(
        min(coarse_result.step_cm1, SEARCH_PLOT_FINE_STEP_CM1),
        local_span_cm1 / SEARCH_PLOT_FINE_MAX_POINTS,
    )

    try:
        data_source = store.get("data_source", LIVE_DB_MODE)
        fine_result = build_manual_spectrum(
            concentrations=merged_concentrations,
            temperature_c=coarse_result.temperature_c,
            pressure_hpa=coarse_result.pressure_hpa,
            range_unit="um",
            range_min=local_min_um,
            range_max=local_max_um,
            step_cm1=fine_step_cm1,
            data_source=data_source,
        )
    except Exception:
        return coarse_result, coarse_result.step_cm1
    return fine_result, fine_step_cm1


app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="hero",
            children=[
                html.Div(
                    className="hero-copy",
                    children=[
                        html.P(f"HITRAN-basierte Absorptionsanalyse ({HITRAN_RUNTIME_LABEL})", className="eyebrow"),
                        html.H1("TomExplorer", className="hero-title"),
                        html.P(
                            "Browserbasierte Simulation manueller Spektren und automatische Vorschläge für Laserdurchstimmbereiche in einer Oberfläche",
                            className="hero-text",
                        ),
                    ],
                ),
                html.Div(
                    className="hero-badge",
                    children=[hero_logo()],
                ),
            ],
        ),
        html.Div(
            className="offline-mode-strip",
            children=[
                dcc.Checklist(
                    id="offline-mode",
                    options=[{"label": "Schnelle offline DB verwenden", "value": OFFLINE_DB_MODE}],
                    value=[OFFLINE_DB_MODE],
                    inline=True,
                    className="offline-mode-toggle",
                ),
                html.Span(id="offline-mode-meta", className="offline-mode-meta"),
                html.Div(
                    className="offline-cache-picker",
                    children=[
                        html.Span("Berechnungscache", className="field-label"),
                        dcc.Dropdown(
                            id="manual-offline-candidate",
                            options=[],
                            value=None,
                            placeholder="Gespeicherte Spektren für aktuelle Gaswahl/Bereich",
                            clearable=True,
                            className="mini-dropdown",
                        ),
                        html.Span(id="manual-offline-candidate-meta", className="offline-mode-meta"),
                    ],
                ),
            ],
        ),
        dcc.Tabs(
            className="tabs",
            value="manual",
            children=[
                dcc.Tab(
                    label="Manuelles Spektrum",
                    value="manual",
                    className="tab",
                    selected_className="tab-selected",
                    children=[
                        html.Div(
                            className="tab-grid",
                            children=[
                                html.Div(
                                    className="sidebar",
                                    children=[
                                        html.Div(
                                            className="manual-primary-actions",
                                            children=[
                                                html.Div(
                                                    className="button-lock-wrap",
                                                    children=[
                                                        html.Button("Spektrum berechnen", id="manual-run", className="action-button"),
                                                        html.Div("Bitte warten...", id="manual-run-fetch-lock", className="button-lock", style=BUTTON_LOCK_HIDDEN),
                                                    ],
                                                ),
                                                html.Button("Berechnung abbrechen", id="manual-cancel", className="secondary-button", disabled=True),
                                                dcc.Checklist(
                                                    id="manual-auto-update",
                                                    options=[{"label": "Auto-Update", "value": "auto"}],
                                                    value=["auto"],
                                                    inline=True,
                                                    persistence=True,
                                                    persistence_type="local",
                                                    className="manual-auto-update-toggle",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="sidebar-scroll",
                                            children=[
                                                controls_section(
                                                    "Komponenten",
                                                    "Auswahl über Dropdown, Konzentrationen pro Komponente darunter. Details erscheinen unten im Hover-Feld statt direkt im Plot.",
                                                    [
                                                        dcc.Dropdown(
                                                            id="manual-gases",
                                                            options=gas_options(),
                                                            value=DEFAULT_MANUAL_GASES,
                                                            multi=True,
                                                            maxHeight=520,
                                                            optionHeight=38,
                                                            persistence=True,
                                                            persistence_type="local",
                                                            className="main-dropdown",
                                                        ),
                                                        html.Div(
                                                            id="manual-concentration-rows",
                                                            className="stack",
                                                            children=[
                                                                concentration_row(
                                                                    "manual",
                                                                    gas,
                                                                    *default_concentration_for(gas, DEFAULT_MANUAL_CONCENTRATIONS),
                                                                    visible=gas in DEFAULT_MANUAL_GASES,
                                                                )
                                                                for gas in ALL_GASES
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                controls_section(
                                                    "Randbedingungen",
                                                    "",
                                                    [
                                                        html.Div(
                                                            className="control-grid two-column",
                                                            children=[
                                                                parameter_field(
                                                                    "T [°C]",
                                                                    dcc.Input(
                                                                        id="manual-temperature",
                                                                        type="number",
                                                                        value=35.0,
                                                                        className="number-input",
                                                                        placeholder="35.0",
                                                                        persistence=True,
                                                                        persistence_type="local",
                                                                    ),
                                                                ),
                                                                parameter_field(
                                                                    "p [hPa]",
                                                                    dcc.Input(
                                                                        id="manual-pressure",
                                                                        type="number",
                                                                        value=1035.0,
                                                                        className="number-input",
                                                                        placeholder="1035.0",
                                                                        persistence=True,
                                                                        persistence_type="local",
                                                                    ),
                                                                ),
                                                                parameter_field(
                                                                    "Bereichseinheit",
                                                                    dcc.Dropdown(
                                                                        id="manual-range-unit",
                                                                        options=[
                                                                            {"label": "Wellenlänge [µm]", "value": "um"},
                                                                            {"label": "Wellenzahl [cm⁻¹]", "value": "cm-1"},
                                                                        ],
                                                                        value="um",
                                                                        clearable=False,
                                                                        persistence=True,
                                                                        persistence_type="local",
                                                                        className="main-dropdown",
                                                                    ),
                                                                ),
                                                                parameter_field(
                                                                    "Schrittweite [cm⁻¹]",
                                                                    dcc.Input(
                                                                        id="manual-step",
                                                                        type="number",
                                                                        value=0.01,
                                                                        step="any",
                                                                        min=0,
                                                                        inputMode="decimal",
                                                                        className="number-input",
                                                                        placeholder="0.01",
                                                                        persistence=True,
                                                                        persistence_type="local",
                                                                    ),
                                                                ),
                                                                parameter_field(
                                                                    html.Span("λ Minimum [µm]", id="manual-range-min-label"),
                                                                    dcc.Input(
                                                                        id="manual-range-min",
                                                                        type="number",
                                                                        value=3.22,
                                                                        className="number-input",
                                                                        placeholder="3.22",
                                                                        persistence=True,
                                                                        persistence_type="local",
                                                                    ),
                                                                ),
                                                                parameter_field(
                                                                    html.Span("λ Maximum [µm]", id="manual-range-max-label"),
                                                                    dcc.Input(
                                                                        id="manual-range-max",
                                                                        type="number",
                                                                        value=3.24,
                                                                        className="number-input",
                                                                        placeholder="3.24",
                                                                        persistence=True,
                                                                        persistence_type="local",
                                                                    ),
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="toggle-row",
                                                            children=[
                                                                dcc.RadioItems(
                                                                    id="manual-y-mode",
                                                                    options=[
                                                                        {"label": "Alpha anzeigen", "value": "alpha"},
                                                                        {"label": "Sigma anzeigen", "value": "sigma"},
                                                                    ],
                                                                    value="alpha",
                                                                    inline=True,
                                                                    persistence=True,
                                                                    persistence_type="local",
                                                                ),
                                                                dcc.Checklist(
                                                                    id="manual-log-scale",
                                                                    options=[{"label": "Log y", "value": "log"}],
                                                                    value=[],
                                                                    inline=True,
                                                                    persistence=True,
                                                                    persistence_type="local",
                                                                ),
                                                                html.Div(
                                                                    className="log-level-wrap",
                                                                    children=[
                                                                        html.Span("Log [Dekaden 1-5]", className="field-label"),
                                                                        dcc.Slider(
                                                                            id="manual-log-level",
                                                                            min=1,
                                                                            max=5,
                                                                            step=1,
                                                                            value=5,
                                                                            marks={level: str(level) for level in range(1, 6)},
                                                                            included=False,
                                                                            persistence=True,
                                                                            persistence_type="local",
                                                                        ),
                                                                    ],
                                                                ),
                                                                dcc.Checklist(
                                                                    id="manual-visible-gases",
                                                                    options=[],
                                                                    value=[],
                                                                    persistence=True,
                                                                    persistence_type="local",
                                                                    className="component-visibility-checklist",
                                                                ),
                                                            ],
                                                        ),
                                                        html.P(
                                                            "Die Berechnung läuft temperatur- und druckabhängig direkt über die lokale HAPI/HITRAN-Datenbank. Offline funktioniert das für bereits lokal gecachte Gase und Bereiche; beim ersten Zugriff kann daher ein Download nötig sein.",
                                                            className="section-copy small",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(id="manual-status", className="status-box sidebar-status-box"),
                                        html.Div(
                                            className="sidebar-actions",
                                            children=[
                                                html.Div(
                                                    className="button-row",
                                                    children=[
                                                        html.Div(
                                                            className="button-lock-wrap",
                                                            children=[
                                                                html.Button("Lokalen HITRAN-Cache aktualisieren", id="manual-fetch", className="secondary-button"),
                                                                html.Div("Bitte warten...", id="manual-fetch-manual-lock", className="button-lock", style=BUTTON_LOCK_HIDDEN),
                                                                html.Div("Bitte warten...", id="manual-fetch-search-lock", className="button-lock", style=BUTTON_LOCK_HIDDEN),
                                                            ],
                                                        ),
                                                        html.Button("CSV exportieren", id="manual-export", className="secondary-button"),
                                                    ],
                                                ),
                                                html.P(
                                                    "Waehrend der HITRAN-Aktualisierung bitte keine neuen Berechnungen starten. Die ausgewaehlten Gase und die schnelle Offline-DB werden in dieser Zeit neu aufgebaut.",
                                                    className="section-copy small",
                                                ),
                                                html.Div(id="manual-fetch-status", className="status-box"),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="content",
                                    children=[
                                        html.Div(
                                            className="manual-main-row",
                                            children=[
                                                dcc.Loading(
                                                    type="circle",
                                                    children=[
                                                        dcc.Graph(
                                                            id="manual-graph",
                                                            figure=empty_figure("Spektrum wird nach der ersten Berechnung hier angezeigt."),
                                                            className="main-graph",
                                                            clear_on_unhover=False,
                                                        ),
                                                    ],
                                                ),
                                                html.Div(id="manual-hover-panel", className="manual-hover-panel", children=hover_panel(None)),
                                            ],
                                        ),
                                        pas_lod_controls("manual"),
                                        html.Div(id="manual-source-info", children=source_details_panel(None, None, "um")),
                                        dcc.Store(id="manual-spectrum-store"),
                                        dcc.Store(id="manual-range-unit-store", data="um", storage_type="local"),
                                        dcc.Store(id="manual-cancel-store", data=0),
                                        dcc.Download(id="manual-export-download"),
                                        dcc.ConfirmDialog(id="hitran-update-dialog"),
                                        dcc.Interval(id="startup-hitran-check", interval=400, n_intervals=0, max_intervals=1),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Bandensuche",
                    value="search",
                    className="tab",
                    selected_className="tab-selected",
                    children=[
                        html.Div(
                            className="tab-grid",
                            children=[
                                html.Div(
                                    className="sidebar",
                                    children=[
                                        html.Div(
                                            className="sidebar-scroll",
                                            children=[
                                                controls_section(
                                                    "Zielgase",
                                                    "Minimale Zielkonzentrationen, die nachweisbar sein sollen.",
                                                    [
                                                        dcc.Dropdown(
                                                            id="search-target-gases",
                                                            options=gas_options(),
                                                            value=DEFAULT_TARGET_GASES,
                                                            multi=True,
                                                            maxHeight=520,
                                                            optionHeight=38,
                                                            persistence=True,
                                                            persistence_type="local",
                                                            className="main-dropdown",
                                                        ),
                                                        html.Div(
                                                            id="search-target-rows",
                                                            className="stack",
                                                            children=[
                                                                concentration_row(
                                                                    "search-target",
                                                                    gas,
                                                                    *default_concentration_for(gas, DEFAULT_TARGET_CONCENTRATIONS),
                                                                    visible=gas in DEFAULT_TARGET_GASES,
                                                                )
                                                                for gas in ALL_GASES
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                controls_section(
                                                    "Störgase",
                                                    "Maximal zu erwartende Hintergrund- oder Bulk-Konzentrationen.",
                                                    [
                                                        dcc.Dropdown(
                                                            id="search-interference-gases",
                                                            options=gas_options(),
                                                            value=DEFAULT_INTERFERENCE_GASES,
                                                            multi=True,
                                                            maxHeight=520,
                                                            optionHeight=38,
                                                            persistence=True,
                                                            persistence_type="local",
                                                            className="main-dropdown",
                                                        ),
                                                        html.Div(
                                                            id="search-interference-rows",
                                                            className="stack",
                                                            children=[
                                                                concentration_row(
                                                                    "search-interference",
                                                                    gas,
                                                                    *default_concentration_for(gas, DEFAULT_INTERFERENCE_CONCENTRATIONS),
                                                                    visible=gas in DEFAULT_INTERFERENCE_GASES,
                                                                )
                                                                for gas in ALL_GASES
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                controls_section(
                                                    "Suchparameter",
                                                    "Die Bandensuche bewertet Kandidatenfenster und kombiniert sie zu Laserplänen mit maximal der angegebenen Anzahl an Lasern.",
                                                    [
                                                        html.Div(
                                                            className="control-grid two-column",
                                                            children=[
                                                                parameter_field(
                                                                    "T [°C]",
                                                                    dcc.Input(id="search-temperature", type="number", value=35.0, className="number-input", placeholder="35.0"),
                                                                ),
                                                                parameter_field(
                                                                    "p [hPa]",
                                                                    dcc.Input(id="search-pressure", type="number", value=1035.0, className="number-input", placeholder="1035.0"),
                                                                ),
                                                                parameter_field(
                                                                    "Bereichseinheit",
                                                                    dcc.Dropdown(
                                                                        id="search-range-unit",
                                                                        options=[
                                                                            {"label": "Wellenlänge [µm]", "value": "um"},
                                                                            {"label": "Wellenzahl [cm⁻¹]", "value": "cm-1"},
                                                                        ],
                                                                        value="um",
                                                                        clearable=False,
                                                                        className="main-dropdown",
                                                                    ),
                                                                ),
                                                                parameter_field(
                                                                    html.Span("λ Minimum [µm]", id="search-range-min-label"),
                                                                    dcc.Input(id="search-range-min", type="number", value=2.0, className="number-input", placeholder="2.0"),
                                                                ),
                                                                parameter_field(
                                                                    html.Span("λ Maximum [µm]", id="search-range-max-label"),
                                                                    dcc.Input(id="search-range-max", type="number", value=7.0, className="number-input", placeholder="7.0"),
                                                                ),
                                                                parameter_field(
                                                                    "Durchstimmbereich [nm]",
                                                                    dcc.Input(id="search-tuning-range", type="number", value=20.0, className="number-input", placeholder="20.0"),
                                                                ),
                                                                parameter_field(
                                                                    "Maximale Laserzahl",
                                                                    dcc.Input(id="search-max-lasers", type="number", value=3, className="number-input", placeholder="3"),
                                                                ),
                                                                parameter_field(
                                                                    "Beste Treffer [1-10]",
                                                                    dcc.Input(id="search-result-limit", type="number", value=3, min=1, max=10, step=1, className="number-input", placeholder="3"),
                                                                ),
                                                                parameter_field(
                                                                    "Schrittweite [cm⁻¹]",
                                                                    dcc.Input(id="search-step", type="number", value=0.02, step="any", min=0, inputMode="decimal", className="number-input", placeholder="0.02"),
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="sidebar-actions",
                                            children=[
                                                html.Div(
                                                    className="button-lock-wrap",
                                                    children=[
                                                        html.Button("Bandensuche starten", id="search-run", className="action-button"),
                                                        html.Div("Bitte warten...", id="search-run-fetch-lock", className="button-lock", style=BUTTON_LOCK_HIDDEN),
                                                    ],
                                                ),
                                                html.Div(id="search-status", className="status-box"),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="content",
                                    children=[
                                        html.Div(
                                            className="results-table-wrap",
                                            children=[
                                                dash_table.DataTable(
                                                    id="search-results-table",
                                                    columns=[
                                                        {"name": "Rang", "id": "rank"},
                                                        {"name": "Score", "id": "score"},
                                                        {"name": "Laser", "id": "lasers"},
                                                        {"name": "Abgedeckt", "id": "covered"},
                                                        {"name": "Fehlt", "id": "missing"},
                                                        {"name": "Laserfenster", "id": "ranges"},
                                                        {"name": "Worst-Case", "id": "robustness"},
                                                    ],
                                                    data=[],
                                                    row_selectable="single",
                                                    selected_rows=[],
                                                    style_as_list_view=True,
                                                    style_table={"height": "190px", "maxHeight": "190px", "overflowY": "auto", "overflowX": "auto", "borderRadius": "18px"},
                                                    style_header={"backgroundColor": "#1f3b2f", "color": "#fffdf8", "fontFamily": BODY_FONT, "fontWeight": 600},
                                                    style_cell={"backgroundColor": "#fffdf8", "color": "#1f2937", "fontFamily": BODY_FONT, "padding": "10px", "whiteSpace": "normal", "textAlign": "left"},
                                                    style_data_conditional=[{"if": {"state": "selected"}, "backgroundColor": "#facc15", "color": "#1f2937"}],
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            type="circle",
                                            children=[
                                                html.Div(
                                                    className="graph-stack search-graph-stack",
                                                    children=[
                                                        html.Div(
                                                            className="toggle-row search-graph-toolbar",
                                                            children=[
                                                                dcc.Checklist(
                                                                    id="search-log-scale",
                                                                    options=[{"label": "Log y", "value": "log"}],
                                                                    value=[],
                                                                    inline=True,
                                                                ),
                                                                html.Div(
                                                                    className="log-level-wrap",
                                                                    children=[
                                                                        html.Span("Log [Dekaden 1-5]", className="field-label"),
                                                                        dcc.Slider(
                                                                            id="search-log-level",
                                                                            min=1,
                                                                            max=5,
                                                                            step=1,
                                                                            value=3,
                                                                            marks={level: str(level) for level in range(1, 6)},
                                                                            included=False,
                                                                        ),
                                                                    ],
                                                                ),
                                                                dcc.Checklist(
                                                                    id="search-visible-gases",
                                                                    options=[],
                                                                    value=[],
                                                                    className="component-visibility-checklist",
                                                                ),
                                                            ],
                                                        ),
                                                        dcc.Graph(
                                                            id="search-graph",
                                                            figure=empty_figure("Bandensuche starten und eine Zeile auswählen, um das Spektrum zu prüfen."),
                                                            className="main-graph",
                                                            clear_on_unhover=False,
                                                        ),
                                                        pas_lod_controls("search"),
                                                        html.Div(id="search-source-info", children=source_details_panel(None, None, "um")),
                                                        html.Div(
                                                            id="search-window-figures",
                                                            children=empty_search_window_plots(),
                                                        ),
                                                        html.Div(
                                                            id="search-plan-details",
                                                            children=empty_search_plan_details(),
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(id="search-hover-panel", className="search-hover-panel", children=hover_panel(None)),
                                        dcc.Store(id="search-store"),
                                        dcc.Store(id="search-selected-spectrum-store"),
                                        dcc.Store(id="search-range-unit-store", data="um"),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    [Output(f"manual-concentration-row-{gas}", "className") for gas in ALL_GASES],
    Input("manual-gases", "value"),
)
def update_manual_row_visibility(selected_gases: list[str] | None) -> list[str]:
    return visible_row_classes(selected_gases)


@app.callback(
    [Output(f"search-target-concentration-row-{gas}", "className") for gas in ALL_GASES],
    Input("search-target-gases", "value"),
)
def update_target_row_visibility(selected_gases: list[str] | None) -> list[str]:
    return visible_row_classes(selected_gases)


@app.callback(
    [Output(f"search-interference-concentration-row-{gas}", "className") for gas in ALL_GASES],
    Input("search-interference-gases", "value"),
)
def update_interference_row_visibility(selected_gases: list[str] | None) -> list[str]:
    return visible_row_classes(selected_gases)


@app.callback(
    Output("manual-range-min", "value"),
    Output("manual-range-max", "value"),
    Output("manual-range-min-label", "children"),
    Output("manual-range-max-label", "children"),
    Output("manual-range-unit-store", "data"),
    Input("manual-range-unit", "value"),
    State("manual-range-min", "value"),
    State("manual-range-max", "value"),
    State("manual-range-unit-store", "data"),
)
def sync_manual_range_inputs(
    range_unit: str,
    range_min: float | None,
    range_max: float | None,
    previous_unit: str | None,
) -> tuple[float | None, float | None, str, str, str]:
    source_unit = previous_unit or range_unit
    converted_min, converted_max = convert_range_inputs(source_unit, range_unit, range_min, range_max)
    min_label, max_label = axis_labels_for_unit(range_unit)
    return converted_min, converted_max, min_label, max_label, range_unit


@app.callback(
    Output("manual-offline-candidate", "options"),
    Output("manual-offline-candidate", "value"),
    Output("manual-offline-candidate-meta", "children"),
    Input("manual-gases", "value"),
    Input("manual-range-unit", "value"),
    Input("manual-range-min", "value"),
    Input("manual-range-max", "value"),
    Input("manual-spectrum-store", "data"),
    State("manual-offline-candidate", "value"),
)
def refresh_manual_cache_options(
    selected_gases: list[str] | None,
    range_unit: str,
    range_min: float | None,
    range_max: float | None,
    _spectrum: dict[str, Any] | None,
    current_value: str | None,
) -> tuple[list[dict[str, str]], str | None, str]:
    if not selected_gases or range_min in (None, "") or range_max in (None, ""):
        return [], None, ""
    try:
        entries = list_manual_result_snapshots(
            tuple(sorted(set(selected_gases))),
            str(range_unit),
            float(range_min),
            float(range_max),
        )
    except Exception:
        return [], None, ""

    options = [{"label": manual_cache_option_label(entry), "value": str(entry["id"])} for entry in entries]
    valid_values = {str(item["value"]) for item in options}
    retained_value = current_value if current_value in valid_values else None
    meta = f"{len(options)} gespeicherte Berechnungen gefunden" if options else "Keine gespeicherte Berechnung für aktuelle Gaswahl/Bereich"
    return options, retained_value, meta


@app.callback(
    Output("manual-spectrum-store", "data", allow_duplicate=True),
    Output("manual-status", "children", allow_duplicate=True),
    Output("offline-mode", "value", allow_duplicate=True),
    Output("manual-temperature", "value", allow_duplicate=True),
    Output("manual-pressure", "value", allow_duplicate=True),
    Output("manual-step", "value", allow_duplicate=True),
    Output("manual-gases", "value", allow_duplicate=True),
    *[Output(f"manual-concentration-value-{gas}", "value", allow_duplicate=True) for gas in ALL_GASES],
    *[Output(f"manual-concentration-unit-{gas}", "value", allow_duplicate=True) for gas in ALL_GASES],
    Input("manual-offline-candidate", "value"),
    State("offline-mode", "value"),
    prevent_initial_call=True,
)
def load_manual_cache_candidate(
    snapshot_id: str | None,
    offline_selection: list[str] | None,
) -> tuple[Any, ...]:
    if not snapshot_id:
        raise PreventUpdate
    payload = load_manual_result_snapshot(snapshot_id)
    if not payload:
        raise PreventUpdate

    selected = list(offline_selection or [])
    if OFFLINE_DB_MODE not in selected:
        selected.append(OFFLINE_DB_MODE)
    payload = dict(payload)
    payload["render_revision"] = int(time.time() * 1000)
    temperature_value = payload.get("temperature_c", dash.no_update)
    pressure_value = payload.get("pressure_hpa", dash.no_update)
    step_value = payload.get("step_cm1", dash.no_update)

    components = payload.get("components", {}) if isinstance(payload.get("components", {}), dict) else {}
    selected_gases = [gas for gas in ALL_GASES if gas in components]

    concentration_values: list[float] = []
    concentration_units: list[str] = []
    for gas in ALL_GASES:
        component_payload = components.get(gas)
        if isinstance(component_payload, dict):
            try:
                value, unit = molar_fraction_to_value_unit(float(component_payload.get("concentration", 0.0)))
            except Exception:
                value, unit = default_concentration_for(gas, DEFAULT_MANUAL_CONCENTRATIONS)
        else:
            value, unit = default_concentration_for(gas, DEFAULT_MANUAL_CONCENTRATIONS)
        concentration_values.append(float(value))
        concentration_units.append(str(unit))

    return (
        payload,
        "Spektrum aus lokalem Berechnungscache geladen.",
        selected,
        temperature_value,
        pressure_value,
        step_value,
        selected_gases,
        *concentration_values,
        *concentration_units,
    )


@app.callback(
    Output("offline-mode", "options"),
    Output("offline-mode-meta", "children"),
    Output("offline-mode", "value"),
    Input("startup-hitran-check", "n_intervals"),
    Input("manual-fetch-status", "children"),
    State("offline-mode", "value"),
)
def refresh_offline_mode_state(
    _startup_tick: int,
    _refresh_message: str | None,
    current_value: list[str] | None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    available, meta, option_value = offline_db_state()
    selected_value = current_value or []
    if not available:
        selected_value = []
    return [
        {
            "label": "Schnelle offline DB verwenden",
            "value": option_value,
            "disabled": not available,
        }
    ], meta, selected_value


@app.callback(
    Output("manual-temperature", "disabled"),
    Output("manual-pressure", "disabled"),
    Output("manual-step", "disabled"),
    Output("search-temperature", "disabled"),
    Output("search-pressure", "disabled"),
    Output("search-step", "disabled"),
    Input("offline-mode", "value"),
)
def sync_offline_disabled_state(selection: list[str] | None) -> tuple[bool, bool, bool, bool, bool, bool]:
    disabled = offline_mode_enabled(selection)
    return disabled, disabled, disabled, disabled, disabled, disabled


@app.callback(
    Output("search-range-min", "value"),
    Output("search-range-max", "value"),
    Output("search-range-min-label", "children"),
    Output("search-range-max-label", "children"),
    Output("search-range-unit-store", "data"),
    Input("search-range-unit", "value"),
    State("search-range-min", "value"),
    State("search-range-max", "value"),
    State("search-range-unit-store", "data"),
)
def sync_search_range_inputs(
    range_unit: str,
    range_min: float | None,
    range_max: float | None,
    previous_unit: str | None,
) -> tuple[float | None, float | None, str, str, str]:
    source_unit = previous_unit or range_unit
    converted_min, converted_max = convert_range_inputs(source_unit, range_unit, range_min, range_max)
    min_label, max_label = axis_labels_for_unit(range_unit)
    return converted_min, converted_max, min_label, max_label, range_unit


@app.callback(
    Output("manual-cancel-store", "data"),
    Input("manual-cancel", "n_clicks"),
    State("manual-cancel-store", "data"),
    prevent_initial_call=True,
)
def cancel_manual_run(n_clicks: int | None, token: int | None) -> int:
    if not n_clicks:
        raise PreventUpdate
    MANUAL_CANCEL_EVENT.set()
    return int(token or 0) + 1


@app.callback(
    Output("manual-spectrum-store", "data"),
    Output("manual-status", "children"),
    Output("offline-mode", "value", allow_duplicate=True),
    Input("manual-run", "n_clicks"),
    Input("manual-auto-update", "value"),
    Input("manual-gases", "value"),
    Input("manual-range-unit", "value"),
    Input("manual-range-min", "value"),
    Input("manual-range-max", "value"),
    Input("manual-step", "value"),
    Input("manual-temperature", "value"),
    Input("manual-pressure", "value"),
    Input("offline-mode", "value"),
    Input("manual-cancel-store", "data"),
    *MANUAL_CONCENTRATION_INPUTS,
    State("manual-spectrum-store", "data"),
    running=[
        (Output("manual-run", "disabled"), True, False),
        (Output("manual-run", "children"), "Spektrum wird berechnet...", "Spektrum berechnen"),
        (Output("manual-cancel", "disabled"), False, True),
        (Output("manual-fetch-manual-lock", "style"), BUTTON_LOCK_VISIBLE, BUTTON_LOCK_HIDDEN),
    ],
    prevent_initial_call=True,
)
def update_manual_spectrum(
    _n_clicks: int,
    auto_update_selection: list[str] | None,
    selected_gases: list[str],
    range_unit: str,
    range_min: float,
    range_max: float,
    step_cm1: float,
    temperature_c: float,
    pressure_hpa: float,
    offline_selection: list[str] | None,
    _cancel_token: int,
    *concentration_state_values: Any,
) -> tuple[dict[str, Any] | None, str, list[str]]:
    previous_serialized = concentration_state_values[-1] if concentration_state_values else None
    concentration_state_values = concentration_state_values[:-1]
    trigger_id = getattr(getattr(dash, "ctx", None), "triggered_id", None)
    if trigger_id is None and dash.callback_context.triggered:
        trigger_id = str(dash.callback_context.triggered[0].get("prop_id", "")).split(".", 1)[0]
    if trigger_id == "manual-cancel-store":
        raise PreventUpdate

    MANUAL_CANCEL_EVENT.clear()
    data_source = OFFLINE_DB_MODE if offline_mode_enabled(offline_selection) else LIVE_DB_MODE
    auto_update_enabled = "auto" in (auto_update_selection or [])
    if not auto_update_enabled and trigger_id != "manual-run":
        raise PreventUpdate

    if trigger_id in {"manual-range-unit", "manual-range-min", "manual-range-max"} and previous_serialized:
        try:
            previous_result = deserialize_manual_result(previous_serialized)
            requested_min_um, requested_max_um = normalize_wavelength_window(
                range_unit,
                parse_required_number(range_min, "Minimum"),
                parse_required_number(range_max, "Maximum"),
            )
            prev_min_um = float(np.min(previous_result.wavelength_um))
            prev_max_um = float(np.max(previous_result.wavelength_um))
            tolerance = 2.5e-6
            if abs(requested_min_um - prev_min_um) <= tolerance and abs(requested_max_um - prev_max_um) <= tolerance:
                raise PreventUpdate
        except PreventUpdate:
            raise
        except Exception:
            pass

    try:
        if not selected_gases:
            raise ValueError("Bitte mindestens ein Gas auswaehlen.")
        gas_values = list(concentration_state_values[: len(ALL_GASES)])
        gas_units = list(concentration_state_values[len(ALL_GASES) :])
        gas_tuple = tuple(sorted(set(selected_gases)))
        concentrations = collect_concentrations(gas_values, gas_units, selected_gases)
        parsed_range_min = parse_required_number(range_min, "Minimum")
        parsed_range_max = parse_required_number(range_max, "Maximum")
        parsed_temperature = parse_required_number(temperature_c, "T [°C]")
        parsed_pressure = parse_required_number(pressure_hpa, "p [hPa]")
        requested_step = parse_required_number(step_cm1, "Schrittweite [cm⁻¹]")

        if data_source == LIVE_DB_MODE:
            cached_payload = load_matching_manual_result_snapshot(
                gases=gas_tuple,
                range_unit=range_unit,
                range_min=parsed_range_min,
                range_max=parsed_range_max,
                temperature_c=parsed_temperature,
                pressure_hpa=parsed_pressure,
                step_cm1=requested_step,
                concentrations=concentrations,
            )
            if cached_payload:
                cached_serialized = dict(cached_payload)
                cached_serialized["render_revision"] = int(_n_clicks or 0)
                cached_serialized["display_range_unit"] = range_unit
                selected = list(offline_selection or [])
                if OFFLINE_DB_MODE not in selected:
                    selected.append(OFFLINE_DB_MODE)
                return (
                    cached_serialized,
                    "Spektrum vollständig aus lokalem Berechnungscache geladen (exakter Match für Gase/Bereich/T/p/Schrittweite/Konzentrationen).",
                    selected,
                )

        spectrum_span = abs(normalize_wavenumber_window(range_unit, parsed_range_min, parsed_range_max)[1] - normalize_wavenumber_window(range_unit, parsed_range_min, parsed_range_max)[0])
        is_auto_preview = trigger_id != "manual-run"
        effective_step = requested_step
        if is_auto_preview:
            preview_step = max(requested_step, spectrum_span / 3200.0, 0.02)
            effective_step = preview_step

        t0 = time.perf_counter()
        manual_result = build_manual_spectrum(
            concentrations=concentrations,
            temperature_c=parsed_temperature,
            pressure_hpa=parsed_pressure,
            range_unit=range_unit,
            range_min=parsed_range_min,
            range_max=parsed_range_max,
            step_cm1=effective_step,
            data_source=data_source,
            cancel_event=MANUAL_CANCEL_EVENT,
        )
        t1 = time.perf_counter()
        sampled_result = downsample_manual_result(manual_result)
        serialized = serialize_manual_result(sampled_result)
        serialized["render_revision"] = int(_n_clicks or 0)
        serialized["display_range_unit"] = range_unit
        t2 = time.perf_counter()

        if not is_auto_preview:
            save_manual_result_snapshot(
                serialized_result=serialized,
                gases=gas_tuple,
                range_unit=range_unit,
                range_min=parsed_range_min,
                range_max=parsed_range_max,
                temperature_c=manual_result.temperature_c,
                pressure_hpa=manual_result.pressure_hpa,
                step_cm1=manual_result.step_cm1,
                concentrations=concentrations,
            )

        span_cm1 = abs(sampled_result.wavenumber_cm1.max() - sampled_result.wavenumber_cm1.min())
        suggested_step = recommended_step_cm1(span_cm1, manual_mode=True)
        runtime_note = f" Laufzeit: Build {t1 - t0:.2f}s | RenderPrep {t2 - t1:.2f}s."
        if data_source == OFFLINE_DB_MODE:
            status = (
                f"{len(sampled_result.components)} Komponenten geladen, {len(sampled_result.wavelength_um)} Plotpunkte. "
                f"Quelle: schnelle Offline-DB mit {sampled_result.temperature_c:.1f} °C, {sampled_result.pressure_hpa:.2f} hPa und {sampled_result.step_cm1:.3f} cm⁻¹."
            )
        else:
            status = (
                f"{len(sampled_result.components)} Komponenten geladen, {len(sampled_result.wavelength_um)} Plotpunkte. "
                f"Wenn ein größerer Bereich langsam wird, ist für diese Spannweite etwa {suggested_step:.4f} cm⁻¹ sinnvoll. "
                "Quelle: lokale HAPI/HITRAN-DB."
            )
        if is_auto_preview:
            status = "Vorschau (Auto-Update): gröbere Schrittweite für schnelle Reaktion. Mit 'Spektrum berechnen' wird die volle Auflösung gerechnet. " + status
        status += runtime_note
        status += coverage_gap_notice(manual_result, selected_gases, range_unit, "Fehlende Spektraldatenbereiche:")
        return serialized, status, list(offline_selection or [])
    except InterruptedError:
        if previous_serialized:
            return previous_serialized, "Berechnung abgebrochen. Letztes Spektrum bleibt angezeigt.", list(offline_selection or [])
        return None, "Berechnung abgebrochen.", list(offline_selection or [])
    except Exception as exc:
        return None, format_data_source_error(exc, data_source, range_unit, range_min, range_max), list(offline_selection or [])


@app.callback(
    Output("manual-visible-gases", "options"),
    Output("manual-visible-gases", "value"),
    Input("manual-spectrum-store", "data"),
    State("manual-visible-gases", "value"),
)
def sync_manual_visible_gases(
    serialized_result: dict[str, Any] | None,
    current_selection: list[str] | None,
) -> tuple[list[dict[str, str]], list[str]]:
    options = component_visibility_options(serialized_result)
    return options, normalized_visible_gases(options, current_selection)


@app.callback(
    Output("manual-graph", "figure"),
    Output("manual-source-info", "children"),
    Input("manual-spectrum-store", "data"),
    Input("manual-y-mode", "value"),
    Input("manual-log-scale", "value"),
    Input("manual-log-level", "value"),
    Input("manual-visible-gases", "value"),
    Input("manual-pas-state", "data"),
    Input("manual-range-unit", "value"),
    Input("manual-graph", "relayoutData"),
    State("manual-auto-update", "value"),
    State("manual-graph", "figure"),
)
def render_manual_spectrum(
    serialized_result: dict[str, Any] | None,
    y_mode: str,
    log_scale: list[str],
    log_level: int,
    visible_gases: list[str] | None,
    pas_state: dict[str, Any] | None,
    range_unit: str,
    relayout_data: dict[str, Any] | None,
    auto_update_selection: list[str] | None,
    current_figure: dict[str, Any] | None,
) -> tuple[go.Figure, html.Div]:
    if not serialized_result:
        return empty_figure("Spektrum wird nach der ersten Berechnung hier angezeigt."), source_details_panel(None, None, range_unit)

    trigger_id = getattr(getattr(dash, "ctx", None), "triggered_id", None)
    if trigger_id is None and dash.callback_context.triggered:
        trigger_id = str(dash.callback_context.triggered[0].get("prop_id", "")).split(".", 1)[0]
    auto_update_enabled = "auto" in (auto_update_selection or [])
    if not auto_update_enabled and trigger_id in {
        "manual-y-mode",
        "manual-log-scale",
        "manual-log-level",
        "manual-visible-gases",
        "manual-range-unit",
    }:
        raise PreventUpdate

    log_y = "log" in (log_scale or [])
    x_range, y_range = preserve_manual_ranges(
        current_figure,
        relayout_data,
        log_y,
        y_mode,
        range_unit,
        serialized_result.get("render_revision"),
    )
    figure = make_spectrum_figure(
        serialized_result,
        y_mode=y_mode,
        log_y=log_y,
        log_level=int(log_level),
        title="Manuelles Absorptionsspektrum",
        x_unit=range_unit,
        x_range=x_range,
        y_range=y_range,
        visible_gases=visible_gases,
        pas_state=pas_state if y_mode == "alpha" else None,
    )
    return figure, source_details_panel(serialized_result, visible_gases, range_unit)


@app.callback(
    Output("manual-pas-state", "data"),
    Output("manual-pas-delta-alpha-min", "value"),
    Output("manual-pas-lod", "value"),
    Output("manual-pas-prompt", "children"),
    Output("manual-pas-eps", "value"),
    Output("manual-graph", "style"),
    Input("manual-pas-arm", "n_clicks"),
    Input("manual-graph", "hoverData"),
    Input("manual-graph", "clickData"),
    Input("manual-graph", "relayoutData"),
    Input("manual-spectrum-store", "data"),
    Input("manual-y-mode", "value"),
    Input("manual-range-unit", "value"),
    Input("manual-visible-gases", "value"),
    Input("manual-pas-3sigma", "value"),
    Input("manual-pas-popt", "value"),
    Input("manual-pas-eps", "value"),
    State("manual-pas-state", "data"),
)
def update_manual_pas_lod(
    _arm_clicks: int | None,
    hover_data: dict[str, Any] | None,
    click_data: dict[str, Any] | None,
    relayout_data: dict[str, Any] | None,
    serialized_result: dict[str, Any] | None,
    y_mode: str,
    range_unit: str,
    visible_gases: list[str] | None,
    sigma_3nv: float | None,
    p_opt_mw: float | None,
    eps_value: float | None,
    current_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, str, str, float, dict[str, Any]]:
    state = dict(current_state or {})
    state.setdefault("armed", False)
    state.setdefault("signature", "empty")
    state.setdefault("selected", None)
    state.setdefault("hover", None)
    state.setdefault("pending_x", None)

    eps_sanitized = sanitize_pas_eps(eps_value)
    delta_alpha_min = compute_pas_delta_alpha_min(sigma_3nv, p_opt_mw, eps_sanitized)
    trigger_id = getattr(getattr(dash, "ctx", None), "triggered_id", None)
    trigger_prop = ""
    if dash.callback_context.triggered:
        trigger_prop = str(dash.callback_context.triggered[0].get("prop_id", ""))
    if trigger_id is None and trigger_prop:
        trigger_id = trigger_prop.split(".", 1)[0]

    if trigger_id == "manual-graph" and not state.get("armed"):
        raise PreventUpdate

    if not serialized_result or y_mode != "alpha":
        state = {"armed": False, "signature": "empty", "selected": None, "hover": None, "pending_x": None}
        prompt = "PAS LoD nur bei Alpha-Ansicht verfügbar." if serialized_result else ""
        return state, format_pas_delta_alpha(delta_alpha_min), "", prompt, eps_sanitized, {"cursor": "default"}

    signature = pas_signature(serialized_result, visible_gases)
    if state.get("signature") != signature:
        state = {"armed": False, "signature": signature, "selected": None, "hover": None, "pending_x": None}
    else:
        state["signature"] = signature

    if trigger_id == "manual-pas-arm":
        state["armed"] = True
        state["selected"] = None
        state["hover"] = None
        state["pending_x"] = None

    if state.get("armed") and trigger_id == "manual-graph":
        if trigger_prop.endswith("relayoutData") and relayout_has_explicit_x_range(relayout_data):
            selected_peak = peak_from_x_window(relayout_data, serialized_result, visible_gases, range_unit)
            if selected_peak:
                state["selected"] = selected_peak
                state["armed"] = False
                state["pending_x"] = None
            state["hover"] = None
        elif trigger_prop.endswith("clickData") and click_data:
            points = click_data.get("points") or []
            point = points[0] if points else {}
            click_x = interaction_x_value(point or {}, range_unit)
            if click_x is not None and np.isfinite(click_x):
                pending_x = state.get("pending_x")
                if pending_x in (None, ""):
                    state["pending_x"] = float(click_x)
                    state["hover"] = peak_from_interaction(click_data, serialized_result, visible_gases, range_unit)
                else:
                    selected_peak = peak_from_x_bounds(float(pending_x), float(click_x), serialized_result, visible_gases, range_unit)
                    if not selected_peak:
                        selected_peak = peak_from_interaction(click_data, serialized_result, visible_gases, range_unit)
                    if selected_peak:
                        state["selected"] = selected_peak
                        state["armed"] = False
                    state["pending_x"] = None
                    state["hover"] = None
            else:
                selected_peak = peak_from_interaction(click_data, serialized_result, visible_gases, range_unit)
                if selected_peak:
                    state["selected"] = selected_peak
                    state["armed"] = False
                    state["pending_x"] = None
                state["hover"] = None
        elif trigger_prop.endswith("hoverData"):
            # Avoid continuous graph redraws while hovering; selection is click/drag only.
            state["hover"] = None
    elif not state.get("armed"):
        state["hover"] = None
        state["pending_x"] = None

    lod_text = format_pas_lod(delta_alpha_min, state.get("selected"))
    if state.get("armed"):
        pending_x = state.get("pending_x")
        if pending_x not in (None, "") and np.isfinite(float(pending_x)):
            unit_label = "cm⁻¹" if range_unit == "cm-1" else "µm"
            prompt = (
                f"LoD-Auswahl aktiv: Start bei {float(pending_x):.4f} {unit_label} gesetzt. "
                "Jetzt rechten Rand klicken oder Peakbereich mit linker Maustaste aufziehen."
            )
        else:
            prompt = (
                "LoD-Auswahl aktiv: Entweder Peakbereich mit linker Maustaste aufziehen "
                "oder zwei Mal klicken (links/rechts)."
            )
    elif state.get("selected"):
        selected_peak = state.get("selected") or {}
        dominant_gas = str(selected_peak.get("dominant_gas", ""))
        dominant_conc_ppbv = float(selected_peak.get("dominant_concentration_ppbv", float("nan")))
        if dominant_gas and np.isfinite(dominant_conc_ppbv) and dominant_conc_ppbv > 0.0:
            prompt = (
                f"Peak gewählt: Δα = {float(selected_peak.get('delta_alpha', 0.0)):.3E} 1/cm "
                f"| Gas {display_formula(dominant_gas)} bei {dominant_conc_ppbv:.3g} ppbV"
            )
        else:
            prompt = f"Peak gewählt: Δα = {float(selected_peak.get('delta_alpha', 0.0)):.3E} 1/cm"
    else:
        if trigger_id == "manual-graph":
            prompt = "Im gewählten Bereich wurde kein lokales Peak-Maximum im Summenspektrum gefunden. Bitte enger um den Zielpeak wählen."
        else:
            prompt = ""
    graph_style = {"cursor": "crosshair"} if state.get("armed") else {"cursor": "default"}
    return state, format_pas_delta_alpha(delta_alpha_min), lod_text, prompt, eps_sanitized, graph_style


@app.callback(
    Output("manual-export-download", "data"),
    Input("manual-export", "n_clicks"),
    State("manual-spectrum-store", "data"),
    State("manual-range-unit", "value"),
    State("manual-graph", "relayoutData"),
    State("manual-graph", "figure"),
    prevent_initial_call=True,
)
def export_manual_spectrum_csv(
    _n_clicks: int,
    serialized_result: dict[str, Any] | None,
    range_unit: str,
    relayout_data: dict[str, Any] | None,
    current_figure: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _n_clicks or not serialized_result:
        raise PreventUpdate

    x_range = current_manual_x_range(
        current_figure,
        relayout_data,
        range_unit,
        serialized_result.get("render_revision"),
    )
    csv_content = build_manual_export_csv(serialized_result, range_unit, x_range)
    return dcc.send_string(csv_content, manual_export_filename(serialized_result))


@app.callback(
    Output("search-pas-state", "data"),
    Output("search-pas-delta-alpha-min", "value"),
    Output("search-pas-lod", "value"),
    Output("search-pas-prompt", "children"),
    Output("search-pas-eps", "value"),
    Output("search-graph", "style"),
    Input("search-pas-arm", "n_clicks"),
    Input("search-graph", "hoverData"),
    Input("search-graph", "clickData"),
    Input("search-graph", "relayoutData"),
    Input("search-selected-spectrum-store", "data"),
    Input("search-store", "data"),
    Input("search-range-unit", "value"),
    Input("search-visible-gases", "value"),
    Input("search-pas-3sigma", "value"),
    Input("search-pas-popt", "value"),
    Input("search-pas-eps", "value"),
    State("search-pas-state", "data"),
)
def update_search_pas_lod(
    _arm_clicks: int | None,
    hover_data: dict[str, Any] | None,
    click_data: dict[str, Any] | None,
    relayout_data: dict[str, Any] | None,
    selected_result_store: dict[str, Any] | None,
    store: dict[str, Any] | None,
    range_unit: str,
    visible_gases: list[str] | None,
    sigma_3nv: float | None,
    p_opt_mw: float | None,
    eps_value: float | None,
    current_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, str, str, float, dict[str, Any]]:
    serialized_result = None
    if selected_result_store and selected_result_store.get("spectrum"):
        serialized_result = selected_result_store.get("spectrum")
    elif store and store.get("spectrum"):
        serialized_result = store.get("spectrum")

    state = dict(current_state or {})
    state.setdefault("armed", False)
    state.setdefault("signature", "empty")
    state.setdefault("selected", None)
    state.setdefault("hover", None)
    state.setdefault("pending_x", None)

    eps_sanitized = sanitize_pas_eps(eps_value)
    delta_alpha_min = compute_pas_delta_alpha_min(sigma_3nv, p_opt_mw, eps_sanitized)
    trigger_id = getattr(getattr(dash, "ctx", None), "triggered_id", None)
    trigger_prop = ""
    if dash.callback_context.triggered:
        trigger_prop = str(dash.callback_context.triggered[0].get("prop_id", ""))
    if trigger_id is None and trigger_prop:
        trigger_id = trigger_prop.split(".", 1)[0]

    if trigger_id == "search-graph" and not state.get("armed"):
        raise PreventUpdate

    if not serialized_result:
        state = {"armed": False, "signature": "empty", "selected": None, "hover": None, "pending_x": None}
        return state, format_pas_delta_alpha(delta_alpha_min), "", "", eps_sanitized, {"cursor": "default"}

    signature = pas_signature(serialized_result, visible_gases)
    if state.get("signature") != signature:
        state = {"armed": False, "signature": signature, "selected": None, "hover": None, "pending_x": None}
    else:
        state["signature"] = signature

    if trigger_id == "search-pas-arm":
        state["armed"] = True
        state["selected"] = None
        state["hover"] = None
        state["pending_x"] = None

    if state.get("armed") and trigger_id == "search-graph":
        if trigger_prop.endswith("relayoutData") and relayout_has_explicit_x_range(relayout_data):
            selected_peak = peak_from_x_window(relayout_data, serialized_result, visible_gases, range_unit)
            if selected_peak:
                state["selected"] = selected_peak
                state["armed"] = False
                state["pending_x"] = None
            state["hover"] = None
        elif trigger_prop.endswith("clickData") and click_data:
            points = click_data.get("points") or []
            point = points[0] if points else {}
            click_x = interaction_x_value(point or {}, range_unit)
            if click_x is not None and np.isfinite(click_x):
                pending_x = state.get("pending_x")
                if pending_x in (None, ""):
                    state["pending_x"] = float(click_x)
                    state["hover"] = peak_from_interaction(click_data, serialized_result, visible_gases, range_unit)
                else:
                    selected_peak = peak_from_x_bounds(float(pending_x), float(click_x), serialized_result, visible_gases, range_unit)
                    if not selected_peak:
                        selected_peak = peak_from_interaction(click_data, serialized_result, visible_gases, range_unit)
                    if selected_peak:
                        state["selected"] = selected_peak
                        state["armed"] = False
                    state["pending_x"] = None
                    state["hover"] = None
            else:
                selected_peak = peak_from_interaction(click_data, serialized_result, visible_gases, range_unit)
                if selected_peak:
                    state["selected"] = selected_peak
                    state["armed"] = False
                    state["pending_x"] = None
                state["hover"] = None
        elif trigger_prop.endswith("hoverData"):
            # Avoid continuous graph redraws while hovering; selection is click/drag only.
            state["hover"] = None
    elif not state.get("armed"):
        state["hover"] = None
        state["pending_x"] = None

    lod_text = format_pas_lod(delta_alpha_min, state.get("selected"))
    if state.get("armed"):
        pending_x = state.get("pending_x")
        if pending_x not in (None, "") and np.isfinite(float(pending_x)):
            unit_label = "cm⁻¹" if range_unit == "cm-1" else "µm"
            prompt = (
                f"LoD-Auswahl aktiv: Start bei {float(pending_x):.4f} {unit_label} gesetzt. "
                "Jetzt rechten Rand klicken oder Peakbereich mit linker Maustaste aufziehen."
            )
        else:
            prompt = (
                "LoD-Auswahl aktiv: Entweder Peakbereich mit linker Maustaste aufziehen "
                "oder zwei Mal klicken (links/rechts)."
            )
    elif state.get("selected"):
        selected_peak = state.get("selected") or {}
        dominant_gas = str(selected_peak.get("dominant_gas", ""))
        dominant_conc_ppbv = float(selected_peak.get("dominant_concentration_ppbv", float("nan")))
        if dominant_gas and np.isfinite(dominant_conc_ppbv) and dominant_conc_ppbv > 0.0:
            prompt = (
                f"Peak gewählt: Δα = {float(selected_peak.get('delta_alpha', 0.0)):.3E} 1/cm "
                f"| Gas {display_formula(dominant_gas)} bei {dominant_conc_ppbv:.3g} ppbV"
            )
        else:
            prompt = f"Peak gewählt: Δα = {float(selected_peak.get('delta_alpha', 0.0)):.3E} 1/cm"
    else:
        if trigger_id == "search-graph":
            prompt = "Im gewählten Bereich wurde kein lokales Peak-Maximum im Summenspektrum gefunden. Bitte enger um den Zielpeak wählen."
        else:
            prompt = ""
    graph_style = {"cursor": "crosshair"} if state.get("armed") else {"cursor": "default"}
    return state, format_pas_delta_alpha(delta_alpha_min), lod_text, prompt, eps_sanitized, graph_style


@app.callback(
    Output("manual-fetch-status", "children"),
    Input("manual-fetch", "n_clicks"),
    State("manual-gases", "value"),
    State("manual-range-unit", "value"),
    State("manual-range-min", "value"),
    State("manual-range-max", "value"),
    running=[
        (Output("manual-fetch", "disabled"), True, False),
        (Output("manual-fetch", "children"), "Lokaler HITRAN-Cache wird aktualisiert...", "Lokalen HITRAN-Cache aktualisieren"),
        (Output("manual-run-fetch-lock", "style"), BUTTON_LOCK_VISIBLE, BUTTON_LOCK_HIDDEN),
        (Output("search-run-fetch-lock", "style"), BUTTON_LOCK_VISIBLE, BUTTON_LOCK_HIDDEN),
    ],
)
def refresh_manual_hitran_cache(
    n_clicks: int | None,
    selected_gases: list[str],
    range_unit: str,
    range_min: float,
    range_max: float,
) -> str:
    if not n_clicks:
        return ""

    try:
        message = refresh_hitran_database(
            gases=selected_gases or [],
            range_unit=range_unit,
            range_min=parse_required_number(range_min, "Minimum"),
            range_max=parse_required_number(range_max, "Maximum"),
        )
        return message
    except Exception as exc:
        return str(exc)


@app.callback(
    Output("hitran-update-dialog", "displayed"),
    Output("hitran-update-dialog", "message"),
    Input("startup-hitran-check", "n_intervals"),
)
def show_hitran_update_dialog(_n_intervals: int) -> tuple[bool, str]:
    message = startup_hitran_message()
    if not message:
        return False, ""
    return True, message


@app.callback(
    Output("manual-fetch-status", "children", allow_duplicate=True),
    Input("hitran-update-dialog", "submit_n_clicks"),
    State("manual-gases", "value"),
    State("manual-range-unit", "value"),
    State("manual-range-min", "value"),
    State("manual-range-max", "value"),
    prevent_initial_call=True,
)
def refresh_hitran_after_startup_prompt(
    submit_n_clicks: int | None,
    selected_gases: list[str] | None,
    range_unit: str,
    range_min: float,
    range_max: float,
) -> str:
    if not submit_n_clicks:
        raise PreventUpdate

    gases = selected_gases or cached_hitran_gases()
    if not gases:
        return "Kein lokaler HITRAN-Cache vorhanden und noch keine Gase ausgewaehlt. Bitte zuerst Gase auswaehlen und dann aktualisieren."

    try:
        return refresh_hitran_database(
            gases=gases,
            range_unit=range_unit,
            range_min=parse_required_number(range_min, "Minimum"),
            range_max=parse_required_number(range_max, "Maximum"),
        )
    except Exception as exc:
        return str(exc)


app.clientside_callback(
    ClientsideFunction(namespace="tomexplorer", function_name="manual_hover_children"),
    Output("manual-hover-panel", "children"),
    Input("manual-graph", "hoverData"),
    State("manual-spectrum-store", "data"),
    State("manual-hover-panel", "children"),
)


@app.callback(
    Output("search-results-table", "selected_rows"),
    Output("search-store", "data"),
    Output("search-status", "children"),
    Input("search-run", "n_clicks"),
    State("search-target-gases", "value"),
    State("search-interference-gases", "value"),
    State("search-temperature", "value"),
    State("search-pressure", "value"),
    State("search-range-unit", "value"),
    State("search-range-min", "value"),
    State("search-range-max", "value"),
    State("search-tuning-range", "value"),
    State("search-max-lasers", "value"),
    State("search-result-limit", "value"),
    State("search-step", "value"),
    State("offline-mode", "value"),
    *TARGET_CONCENTRATION_STATES,
    *INTERFERENCE_CONCENTRATION_STATES,
    running=[
        (Output("search-run", "disabled"), True, False),
        (Output("search-run", "children"), "Bandensuche läuft...", "Bandensuche starten"),
        (Output("manual-fetch-search-lock", "style"), BUTTON_LOCK_VISIBLE, BUTTON_LOCK_HIDDEN),
    ],
)
def run_band_search(
    _n_clicks: int,
    selected_target_gases: list[str],
    selected_interference_gases: list[str],
    temperature_c: float,
    pressure_hpa: float,
    range_unit: str,
    range_min_value: float,
    range_max_value: float,
    tuning_range_nm: float,
    max_lasers: float,
    result_limit_value: float,
    step_cm1: float,
    offline_selection: list[str] | None,
    *concentration_state_values: Any,
) -> tuple[list[int], dict[str, Any] | None, str]:
    if not _n_clicks:
        raise PreventUpdate

    data_source = OFFLINE_DB_MODE if offline_mode_enabled(offline_selection) else LIVE_DB_MODE
    try:
        split_one = len(ALL_GASES)
        split_two = split_one * 2
        split_three = split_one * 3
        target_values = list(concentration_state_values[:split_one])
        target_units = list(concentration_state_values[split_one:split_two])
        interference_values = list(concentration_state_values[split_two:split_three])
        interference_units = list(concentration_state_values[split_three:])
        target_concentrations = collect_concentrations(
            target_values,
            target_units,
            selected_target_gases,
        )
        interference_concentrations = collect_concentrations(
            interference_values,
            interference_units,
            selected_interference_gases,
        )
        parsed_range_min = parse_required_number(range_min_value, "Minimum")
        parsed_range_max = parse_required_number(range_max_value, "Maximum")
        plans, search_result = suggest_laser_plans(
            target_concentrations=target_concentrations,
            interference_concentrations=interference_concentrations,
            temperature_c=parse_required_number(temperature_c, "T [°C]"),
            pressure_hpa=parse_required_number(pressure_hpa, "p [hPa]"),
            range_unit=range_unit,
            range_min=parsed_range_min,
            range_max=parsed_range_max,
            tuning_range_nm=parse_required_number(tuning_range_nm, "Durchstimmbereich [nm]"),
            max_lasers=int(parse_required_number(max_lasers, "Maximale Laserzahl")),
            step_cm1=parse_required_number(step_cm1, "Schrittweite [cm⁻¹]"),
            data_source=data_source,
        )
        result_limit = max(1, min(10, int(parse_required_number(result_limit_value, "Beste Treffer [1-10]"))))
        sampled_result = downsample_manual_result(search_result)
        visible_plans = plans[:result_limit]
        store = build_search_store(
            plans=visible_plans,
            serialized_spectrum=serialize_manual_result(sampled_result),
            target_concentrations=target_concentrations,
            interference_concentrations=interference_concentrations,
            range_unit=range_unit,
            data_source=data_source,
        )
        coverage_notice = coverage_gap_notice(
            search_result,
            selected_target_gases,
            range_unit,
            "Achtung: Für diese Zielgase fehlen im Suchbereich Daten in:",
        )
        if not plans:
            return [], store, "Keine geeigneten Laserfenster gefunden. Bereich vergrößern, weniger Zielgase pro Laser erzwingen oder Schrittweite vergrößern." + coverage_notice
        if len(visible_plans) < result_limit:
            status = (
                f"{len(plans)} Vorschläge berechnet. Es wurden nur {len(visible_plans)} ausreichend unterschiedliche Treffer gefunden, obwohl {result_limit} angefordert wurden."
            )
        else:
            status = (
                f"{len(plans)} Vorschläge berechnet. Angezeigt werden die {len(visible_plans)} stärksten Treffer mit der besten Zielgas-Abdeckung und dem höchsten Signal-zu-Interferenz-Verhältnis."
            )
        if data_source == OFFLINE_DB_MODE:
            status += (
                f" Quelle: schnelle Offline-DB ({search_result.temperature_c:.1f} °C, {search_result.pressure_hpa:.2f} hPa, {search_result.step_cm1:.3f} cm⁻¹)."
            )
        status += coverage_notice
        return [0], store, status
    except Exception as exc:
        return [], None, format_data_source_error(exc, data_source, range_unit, range_min_value, range_max_value)


@app.callback(
    Output("search-visible-gases", "options"),
    Output("search-visible-gases", "value"),
    Input("search-store", "data"),
    State("search-visible-gases", "value"),
)
def sync_search_visible_gases(
    store: dict[str, Any] | None,
    current_selection: list[str] | None,
) -> tuple[list[dict[str, str]], list[str]]:
    serialized_result = store.get("spectrum") if store else None
    options = component_visibility_options(serialized_result)
    return options, normalized_visible_gases(options, current_selection)


@app.callback(
    Output("search-results-table", "data"),
    Input("search-range-unit", "value"),
    Input("search-store", "data"),
)
def sync_search_results_table(range_unit: str, store: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not store or not store.get("plans"):
        return []
    plans = [deserialize_laser_plan(plan) for plan in store["plans"]]
    return search_table_rows(plans, range_unit)


@app.callback(
    Output("search-graph", "figure"),
    Output("search-source-info", "children"),
    Output("search-window-figures", "children"),
    Output("search-plan-details", "children"),
    Output("search-selected-spectrum-store", "data"),
    Input("search-results-table", "selected_rows"),
    Input("search-range-unit", "value"),
    Input("search-log-scale", "value"),
    Input("search-log-level", "value"),
    Input("search-visible-gases", "value"),
    Input("search-pas-state", "data"),
    State("search-store", "data"),
    State("search-selected-spectrum-store", "data"),
)
def update_search_plot(
    selected_rows: list[int],
    range_unit: str,
    log_scale: list[str],
    log_level: int,
    visible_gases: list[str] | None,
    pas_state: dict[str, Any] | None,
    store: dict[str, Any] | None,
    selected_result_store: dict[str, Any] | None,
) -> tuple[go.Figure, html.Div, html.Div, html.Div, dict[str, Any] | None]:
    if not store or not store.get("plans"):
        return (
            empty_figure("Bandensuche starten und eine Zeile auswählen, um das Spektrum zu prüfen."),
            source_details_panel(None, None, range_unit),
            empty_search_window_plots(),
            empty_search_plan_details(),
            None,
        )
    row_index = selected_rows[0] if selected_rows else 0
    if row_index >= len(store["plans"]):
        row_index = 0
    plan = deserialize_laser_plan(store["plans"][row_index])
    cache_key = "|".join([str(row_index), *[window.window_id for window in plan.windows]])
    trigger_id = getattr(getattr(dash, "ctx", None), "triggered_id", None)
    if trigger_id is None and dash.callback_context.triggered:
        trigger_id = str(dash.callback_context.triggered[0].get("prop_id", "")).split(".", 1)[0]
    can_reuse_cached_result = (
        trigger_id in {"search-log-scale", "search-log-level", "search-visible-gases", "search-range-unit", "search-pas-state"}
        and selected_result_store is not None
        and selected_result_store.get("cache_key") == cache_key
        and selected_result_store.get("spectrum") is not None
    )
    if can_reuse_cached_result:
        result = deserialize_manual_result(selected_result_store["spectrum"])
        fine_step_cm1 = float(selected_result_store.get("fine_step_cm1", result.step_cm1))
    else:
        result, fine_step_cm1 = rebuild_selected_search_result(store, plan)
    selected_store_payload = {
        "cache_key": cache_key,
        "fine_step_cm1": float(fine_step_cm1),
        "spectrum": serialize_manual_result(result),
    }
    x_unit = range_unit
    highlighted_windows = [
        {
            "x_min": float(wavelength_um_to_wavenumber_cm1(window.wavelength_max_um)) if x_unit == "cm-1" else window.wavelength_min_um,
            "x_max": float(wavelength_um_to_wavenumber_cm1(window.wavelength_min_um)) if x_unit == "cm-1" else window.wavelength_max_um,
        }
        for window in plan.windows
    ]
    highlighted_lines: list[dict[str, Any]] = []
    seen_lines: set[tuple[str, float]] = set()
    for window in plan.windows:
        for gas, metric in window.gas_metrics.items():
            line_key = (gas, round(metric.peak_wavelength_um, 6))
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            highlighted_lines.append(
                {
                    "gas": gas,
                    "x_value": float(metric.peak_wavenumber_cm1) if x_unit == "cm-1" else metric.peak_wavelength_um,
                    "color": result.components[gas].color,
                    "label": display_formula_plot(gas),
                }
            )
    zoom_min = min(min(window["x_min"], window["x_max"]) for window in highlighted_windows)
    zoom_max = max(max(window["x_min"], window["x_max"]) for window in highlighted_windows)
    x_values = spectrum_x_values(result, x_unit)
    x_range = search_plot_x_range(x_values, x_unit, zoom_min, zoom_max)
    log_y = "log" in (log_scale or [])
    y_range = search_plot_y_range(result, x_values, x_range, log_y, visible_gases)
    covered_label = ", ".join(plan.covered_targets[:4])
    if len(plan.covered_targets) > 4:
        covered_label += f" +{len(plan.covered_targets) - 4}"
    title = f"Bandensuche | Rang {plan.rank} | Score {plan.score:.1f} | Ziele: {covered_label}"
    plan_revision_key = "search:" + "|".join(
        [str(plan.rank), x_unit, *[window.window_id for window in plan.windows]]
    )
    figure = make_spectrum_figure(
        serialize_manual_result(result),
        y_mode="alpha",
        log_y=log_y,
        log_level=int(log_level),
        title=title,
        x_unit=x_unit,
        highlighted_windows=highlighted_windows,
        highlighted_lines=highlighted_lines,
        x_range=x_range,
        y_range=y_range,
        revision_key=plan_revision_key,
        preserve_ui_state=False,
        visible_gases=visible_gases,
        pas_state=pas_state,
    )
    figure.update_layout(meta={**(figure.layout.meta or {}), "step_cm1": fine_step_cm1})
    serialized_result = selected_store_payload["spectrum"]
    return (
        figure,
        source_details_panel(serialized_result, visible_gases, range_unit),
        build_search_window_plots(plan, store, range_unit, log_y, int(log_level), result, fine_step_cm1, visible_gases),
        build_search_plan_details(plan, store, range_unit),
        selected_store_payload,
    )


app.clientside_callback(
    ClientsideFunction(namespace="tomexplorer", function_name="search_hover_children"),
    Output("search-hover-panel", "children"),
    Input("search-graph", "hoverData"),
    State("search-store", "data"),
    State("search-hover-panel", "children"),
)


def open_browser_on_startup() -> None:
    if os.environ.get("TOMEXPLORER_NO_BROWSER") == "1":
        return

    host = os.environ.get("TOMEXPLORER_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("TOMEXPLORER_PORT", "8050"))
    except ValueError:
        port = 8050

    def _open() -> None:
        url = f"http://{host}:{port}/"
        deadline = time.monotonic() + 30.0

        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    if response.status < 500:
                        break
            except Exception:
                time.sleep(0.25)

        try:
            if os.name == "nt":
                os.startfile(url)
                return
        except OSError:
            pass
        webbrowser.open_new(url)

    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    host = os.environ.get("TOMEXPLORER_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("TOMEXPLORER_PORT", "8050"))
    except ValueError:
        port = 8050

    open_browser_on_startup()
    app.run(host=host, port=port, debug=False, use_reloader=False)