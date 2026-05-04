from __future__ import annotations

from datetime import datetime
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
    normalize_wavenumber_window,
    offline_library_summary,
    recommended_step_cm1,
    refresh_hitran_database,
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

MANUAL_CONCENTRATION_STATES = [
    State(f"manual-concentration-value-{gas}", "value") for gas in ALL_GASES
] + [
    State(f"manual-concentration-unit-{gas}", "value") for gas in ALL_GASES
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


def normalize_wavelength_window(range_unit: str, range_min: float, range_max: float) -> tuple[float, float]:
    if range_unit == "um":
        return float(min(range_min, range_max)), float(max(range_min, range_max))

    nu_min, nu_max = normalize_wavenumber_window(range_unit, range_min, range_max)
    wavelength_min_um = float(wavenumber_cm1_to_wavelength_um(nu_max))
    wavelength_max_um = float(wavenumber_cm1_to_wavelength_um(nu_min))
    return min(wavelength_min_um, wavelength_max_um), max(wavelength_min_um, wavelength_max_um)


def latest_hitran_cache_age_days() -> float | None:
    headers = list((BASE_DIR / "hitran_cache").glob("*.header"))
    if not headers:
        return None
    newest_mtime = max(path.stat().st_mtime for path in headers)
    return (time.time() - newest_mtime) / 86400.0


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
                        className="number-input",
                    ),
                    dcc.Dropdown(
                        id=f"{prefix}-concentration-unit-{gas}",
                        options=UNIT_OPTIONS,
                        value=unit,
                        clearable=False,
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
) -> go.Figure:
    result = deserialize_manual_result(serialized_result)
    render_revision = serialized_result.get("render_revision")
    x_values = spectrum_x_values(result, x_unit)
    secondary_title, secondary_ticks, secondary_labels = secondary_axis_config(x_unit, x_values)
    customdata = np.column_stack((result.wavelength_um * 1000.0, result.wavenumber_cm1))
    total_y = (
        result.total_alpha_per_cm
        if y_mode == "alpha"
        else result.total_sigma_cm2_per_molecule
    )
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

    for gas, component in result.components.items():
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
            line={"color": "#6b7280", "width": 1.4, "dash": "dash"},
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
                                                                    dcc.Input(id="manual-temperature", type="number", value=35.0, className="number-input", placeholder="35.0"),
                                                                ),
                                                                parameter_field(
                                                                    "p [hPa]",
                                                                    dcc.Input(id="manual-pressure", type="number", value=1035.0, className="number-input", placeholder="1035.0"),
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
                                                                        className="main-dropdown",
                                                                    ),
                                                                ),
                                                                parameter_field(
                                                                    "Schrittweite [cm⁻¹]",
                                                                    dcc.Input(id="manual-step", type="number", value=0.01, step="any", min=0, inputMode="decimal", className="number-input", placeholder="0.01"),
                                                                ),
                                                                parameter_field(
                                                                    html.Span("λ Minimum [µm]", id="manual-range-min-label"),
                                                                    dcc.Input(id="manual-range-min", type="number", value=3.22, className="number-input", placeholder="3.22"),
                                                                ),
                                                                parameter_field(
                                                                    html.Span("λ Maximum [µm]", id="manual-range-max-label"),
                                                                    dcc.Input(id="manual-range-max", type="number", value=3.24, className="number-input", placeholder="3.24"),
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
                                                                ),
                                                                dcc.Checklist(
                                                                    id="manual-log-scale",
                                                                    options=[{"label": "Log y", "value": "log"}],
                                                                    value=[],
                                                                    inline=True,
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
                                                                        ),
                                                                    ],
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
                                        html.Div(
                                            className="sidebar-actions",
                                            children=[
                                                html.Div(
                                                    className="button-row",
                                                    children=[
                                                        html.Div(
                                                            className="button-lock-wrap",
                                                            children=[
                                                                html.Button("Spektrum berechnen", id="manual-run", className="action-button"),
                                                                html.Div("Bitte warten...", id="manual-run-fetch-lock", className="button-lock", style=BUTTON_LOCK_HIDDEN),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="button-lock-wrap",
                                                            children=[
                                                                html.Button("Lokalen HITRAN-Cache aktualisieren", id="manual-fetch", className="secondary-button"),
                                                                html.Div("Bitte warten...", id="manual-fetch-manual-lock", className="button-lock", style=BUTTON_LOCK_HIDDEN),
                                                                html.Div("Bitte warten...", id="manual-fetch-search-lock", className="button-lock", style=BUTTON_LOCK_HIDDEN),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.P(
                                                    "Waehrend der HITRAN-Aktualisierung bitte keine neuen Berechnungen starten. Die ausgewaehlten Gase und die schnelle Offline-DB werden in dieser Zeit neu aufgebaut.",
                                                    className="section-copy small",
                                                ),
                                                html.Div(id="manual-status", className="status-box"),
                                                html.Div(id="manual-fetch-status", className="status-box"),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="content",
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
                                        html.Div(id="manual-hover-panel", children=hover_panel(None)),
                                        dcc.Store(id="manual-spectrum-store"),
                                        dcc.Store(id="manual-range-unit-store", data="um"),
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
                                                    className="graph-stack",
                                                    children=[
                                                        dcc.Graph(
                                                            id="search-graph",
                                                            figure=empty_figure("Bandensuche starten und eine Zeile auswählen, um das Spektrum zu prüfen."),
                                                            className="main-graph",
                                                            clear_on_unhover=False,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(id="search-hover-panel", children=hover_panel(None)),
                                        dcc.Store(id="search-store"),
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
    Output("manual-spectrum-store", "data"),
    Output("manual-status", "children"),
    Input("manual-run", "n_clicks"),
    State("manual-gases", "value"),
    State("manual-range-unit", "value"),
    State("manual-range-min", "value"),
    State("manual-range-max", "value"),
    State("manual-step", "value"),
    State("manual-temperature", "value"),
    State("manual-pressure", "value"),
    State("offline-mode", "value"),
    *MANUAL_CONCENTRATION_STATES,
    running=[
        (Output("manual-run", "disabled"), True, False),
        (Output("manual-run", "children"), "Spektrum wird berechnet...", "Spektrum berechnen"),
        (Output("manual-fetch-manual-lock", "style"), BUTTON_LOCK_VISIBLE, BUTTON_LOCK_HIDDEN),
    ],
)
def update_manual_spectrum(
    _n_clicks: int,
    selected_gases: list[str],
    range_unit: str,
    range_min: float,
    range_max: float,
    step_cm1: float,
    temperature_c: float,
    pressure_hpa: float,
    offline_selection: list[str] | None,
    *concentration_state_values: Any,
) -> tuple[dict[str, Any] | None, str]:
    if not _n_clicks:
        raise PreventUpdate

    try:
        gas_values = list(concentration_state_values[: len(ALL_GASES)])
        gas_units = list(concentration_state_values[len(ALL_GASES) :])
        concentrations = collect_concentrations(gas_values, gas_units, selected_gases)
        data_source = OFFLINE_DB_MODE if offline_mode_enabled(offline_selection) else LIVE_DB_MODE
        parsed_range_min = parse_required_number(range_min, "Minimum")
        parsed_range_max = parse_required_number(range_max, "Maximum")
        manual_result = build_manual_spectrum(
            concentrations=concentrations,
            temperature_c=parse_required_number(temperature_c, "T [°C]"),
            pressure_hpa=parse_required_number(pressure_hpa, "p [hPa]"),
            range_unit=range_unit,
            range_min=parsed_range_min,
            range_max=parsed_range_max,
            step_cm1=parse_required_number(step_cm1, "Schrittweite [cm⁻¹]"),
            data_source=data_source,
        )
        sampled_result = downsample_manual_result(manual_result)
        serialized = serialize_manual_result(sampled_result)
        serialized["render_revision"] = int(_n_clicks or 0)
        serialized["display_range_unit"] = range_unit
        span_cm1 = abs(sampled_result.wavenumber_cm1.max() - sampled_result.wavenumber_cm1.min())
        suggested_step = recommended_step_cm1(span_cm1, manual_mode=True)
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
        return serialized, status
    except Exception as exc:
        return None, format_data_source_error(exc, data_source, range_unit, range_min, range_max)


@app.callback(
    Output("manual-graph", "figure"),
    Input("manual-spectrum-store", "data"),
    Input("manual-y-mode", "value"),
    Input("manual-log-scale", "value"),
    Input("manual-log-level", "value"),
    Input("manual-range-unit", "value"),
    Input("manual-graph", "relayoutData"),
    State("manual-graph", "figure"),
)
def render_manual_spectrum(
    serialized_result: dict[str, Any] | None,
    y_mode: str,
    log_scale: list[str],
    log_level: int,
    range_unit: str,
    relayout_data: dict[str, Any] | None,
    current_figure: dict[str, Any] | None,
) -> go.Figure:
    if not serialized_result:
        return empty_figure("Spektrum wird nach der ersten Berechnung hier angezeigt.")
    log_y = "log" in (log_scale or [])
    x_range, y_range = preserve_manual_ranges(
        current_figure,
        relayout_data,
        log_y,
        y_mode,
        range_unit,
        serialized_result.get("render_revision"),
    )
    return make_spectrum_figure(
        serialized_result,
        y_mode=y_mode,
        log_y=log_y,
        log_level=int(log_level),
        title="Manuelles Absorptionsspektrum",
        x_unit=range_unit,
        x_range=x_range,
        y_range=y_range,
    )


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
        if not plans:
            return [], store, "Keine geeigneten Laserfenster gefunden. Bereich vergrößern, weniger Zielgase pro Laser erzwingen oder Schrittweite vergrößern."
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
        return [0], store, status
    except Exception as exc:
        return [], None, format_data_source_error(exc, data_source, range_unit, range_min_value, range_max_value)


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
    Input("search-results-table", "selected_rows"),
    Input("search-range-unit", "value"),
    State("search-store", "data"),
)
def update_search_plot(selected_rows: list[int], range_unit: str, store: dict[str, Any] | None) -> go.Figure:
    if not store or not store.get("plans"):
        return empty_figure("Bandensuche starten und eine Zeile auswählen, um das Spektrum zu prüfen.")
    row_index = selected_rows[0] if selected_rows else 0
    if row_index >= len(store["plans"]):
        row_index = 0
    plan = deserialize_laser_plan(store["plans"][row_index])
    result, fine_step_cm1 = rebuild_selected_search_result(store, plan)
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
                    "x_value": float(metric.peak_wavenumber_cm1) if x_unit == "cm-1" else metric.peak_wavelength_um,
                    "color": result.components[gas].color,
                    "label": display_formula(gas),
                }
            )
    zoom_min = min(min(window["x_min"], window["x_max"]) for window in highlighted_windows)
    zoom_max = max(max(window["x_min"], window["x_max"]) for window in highlighted_windows)
    x_values = spectrum_x_values(result, x_unit)
    if x_unit == "cm-1":
        center_value = (zoom_min + zoom_max) / 2.0
        center_wavelength_um = float(wavenumber_cm1_to_wavelength_um(center_value))
        min_padding = abs(
            float(wavelength_um_to_wavenumber_cm1(center_wavelength_um - SEARCH_PLOT_PADDING_UM))
            - float(wavelength_um_to_wavenumber_cm1(center_wavelength_um + SEARCH_PLOT_PADDING_UM))
        ) / 2.0
    else:
        min_padding = SEARCH_PLOT_PADDING_UM
    zoom_padding = max((zoom_max - zoom_min) * 0.08, min_padding)
    x_range = [
        max(float(np.min(x_values)), zoom_min - zoom_padding),
        min(float(np.max(x_values)), zoom_max + zoom_padding),
    ]
    if x_unit == "cm-1":
        x_range = [x_range[1], x_range[0]]
    mask = (x_values >= min(x_range)) & (x_values <= max(x_range))
    local_alpha = result.total_alpha_per_cm[mask]
    y_range: list[float] | None = None
    if local_alpha.size:
        local_max = float(np.max(local_alpha))
        local_min = float(np.min(local_alpha))
        span = local_max - local_min
        padding = max(span * 0.15, local_max * 0.08, 1.0e-12)
        y_range = [0.0, local_max + padding]
    covered_label = ", ".join(plan.covered_targets[:4])
    if len(plan.covered_targets) > 4:
        covered_label += f" +{len(plan.covered_targets) - 4}"
    title = f"Bandensuche | Rang {plan.rank} | Score {plan.score:.1f} | Ziele: {covered_label}"
    plan_revision_key = "search:" + "|".join(
        [str(plan.rank), x_unit, *[window.window_id for window in plan.windows]]
    )
    figure = make_spectrum_figure(
        store["spectrum"],
        y_mode="alpha",
        log_y=False,
        title=title,
        x_unit=x_unit,
        highlighted_windows=highlighted_windows,
        highlighted_lines=highlighted_lines,
        x_range=x_range,
        y_range=y_range,
        revision_key=plan_revision_key,
        preserve_ui_state=False,
    )
    figure.update_layout(meta={**(figure.layout.meta or {}), "step_cm1": fine_step_cm1})

    annotation_lines: list[str] = []
    for window in plan.windows:
        for gas, metric in sorted(window.gas_metrics.items()):
            concentration = store.get("target_concentrations", {}).get(gas)
            if concentration is None:
                concentration = store.get("interference_concentrations", {}).get(gas, 0.0)
            annotation_lines.append(
                " | ".join(
                    [
                        display_formula(gas),
                        f"λ {metric.peak_wavelength_um:.4f} µm",
                        f"ν {metric.peak_wavenumber_cm1:.2f} cm⁻¹",
                        f"σ {metric.peak_sigma_cm2_per_molecule:.2e}",
                        f"α {metric.peak_alpha_per_cm:.2e}",
                        f"WC S/I {metric.signal_to_interference:.2f}",
                        f"WC Δα-Sel {metric.peak_region_delta_alpha_selectivity:.2f}",
                        f"WC 2f-Sel {metric.peak_region_wms2f_selectivity:.2f}",
                        f"WC 2f-Fit {metric.peak_region_wms2f_shape_similarity:.2f}",
                        format_concentration(float(concentration)),
                    ]
                )
            )

    if annotation_lines:
        figure.add_annotation(
            x=0.995,
            y=0.995,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            align="left",
            showarrow=False,
            font={"size": 10, "family": BODY_FONT, "color": "#1f2937"},
            bgcolor="rgba(255, 253, 248, 0.88)",
            bordercolor="rgba(148, 163, 184, 0.35)",
            borderwidth=1,
            text="<br>".join(annotation_lines[:8]),
        )
    return figure


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