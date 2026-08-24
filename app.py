from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, savgol_filter
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Force-Length Work Loop Analyser",
    page_icon="📈",
    layout="wide",
)

st.title("Force-Length Work Loop Analyser")

st.markdown(
    """
Upload a comma-delimited `.dat` file containing:

- **Column 1:** Force
- **Column 2:** Length

The application will:

1. Read the force and length data.
2. Calculate a baseline from the final length values.
3. Subtract that baseline from every length value.
4. Detect individual length-change cycles.
5. Apply several smoothing methods to the force signal.
6. Plot force against length for each work loop.
7. Export the complete processed data and individual cycle data as CSV files.
"""
)


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def make_odd(value: int) -> int:
    """Return an odd integer of at least 3."""
    value = max(3, int(value))

    if value % 2 == 0:
        value += 1

    return value


def valid_odd_window(
    requested_window: int,
    number_of_points: int,
) -> Optional"""
    Return a valid odd filter window that does not exceed
    the number of available data points.
    """
    if number_of_points < 3:
        return None

    window = min(
        make_odd(requested_window),
        number_of_points,
    )

    if window % 2 == 0:
        window -= 1

    if window < 3:
        return None

    return window


def clean_column_name(name: str) -> str:
    """
    Convert a descriptive signal name into a CSV-friendly column name.
    """
    name = name.lower()
    name = name.replace("×", "x")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")

    return name


# ============================================================
# FILE IMPORT
# ============================================================

def read_dat_file(uploaded_file) -> pd.DataFrame:
    """
    Read a comma-delimited DAT, CSV, or TXT file.

    The first valid numeric column is treated as Force.
    The second valid numeric column is treated as Length.

    Header rows and other non-numeric rows are removed.
    """
    file_bytes = uploaded_file.getvalue()

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
    ]

    imported_data = None
    last_error = None

    for encoding in encodings:
        try:
            imported_data = pd.read_csv(
                io.BytesIO(file_bytes),
                sep=",",
                header=None,
                comment="#",
                engine="python",
                encoding=encoding,
                skip_blank_lines=True,
            )
            break

        except Exception as error:
            last_error = error

    if imported_data is None:
        raise ValueError(
            f"The file could not be read. Details: {last_error}"
        )

    if imported_data.shape[1] < 2:
        raise ValueError(
            "The file must contain at least two comma-delimited columns."
        )

    imported_data = imported_data.iloc[:, :2].copy()

    imported_data.columns = [
        "Force_raw",
        "Length_raw",
    ]

    imported_data["Force_raw"] = pd.to_numeric(
        imported_data["Force_raw"],
        errors="coerce",
    )

    imported_data["Length_raw"] = pd.to_numeric(
        imported_data["Length_raw"],
        errors="coerce",
    )

    imported_data = imported_data.dropna(
        subset=["Force_raw", "Length_raw"]
    ).reset_index(drop=True)

    if len(imported_data) < 3:
        raise ValueError(
            "Fewer than three valid numeric rows were found."
        )

    imported_data.insert(
        0,
        "Sample",
        np.arange(len(imported_data), dtype=int),
    )

    return imported_data


# ============================================================
# RUN DETECTION
# ============================================================

def identify_true_runs(
    boolean_values: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Identify consecutive runs of True values.

    Returns inclusive start and end indices.
    """
    boolean_values = np.asarray(
        boolean_values,
        dtype=bool,
    )

    if len(boolean_values) == 0:
        return []

    padded_values = np.concatenate(
        [
            np.array([False]),
            boolean_values,
            np.array([False]),
        ]
    )

    changes = np.diff(
        padded_values.astype(int)
    )

    run_starts = np.where(changes == 1)[0]
    run_ends = np.where(changes == -1)[0] - 1

    return list(
        zip(
            run_starts.tolist(),
            run_ends.tolist(),
        )
    )


# ============================================================
# CYCLE DETECTION
# ============================================================

def detect_length_cycles(
    length_values: np.ndarray,
    sustained_increase_points: int,
    positive_change_threshold: float,
    negative_length_threshold: float,
    negative_run_points: int,
    minimum_cycle_points: int,
) -> Tuple[
    List[Dict[str, int]],
    np.ndarray,
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    """
    Detect distinct length-change cycles.

    Cycle start
    -----------
    The first point of a sustained positive increase in length.

    A sustained positive increase must contain at least
    `sustained_increase_points` consecutive point-to-point changes
    greater than `positive_change_threshold`.

    Cycle end
    ---------
    The cycle ends when the length reaches the beginning of the
    second distinct negative-length run.

    A negative-length run must contain at least
    `negative_run_points` consecutive values below
    `negative_length_threshold`.

    After a complete cycle has been detected, the algorithm searches
    for the next sustained positive increase after the previous cycle.
    """
    length_values = np.asarray(
        length_values,
        dtype=float,
    )

    number_of_points = len(length_values)

    if number_of_points < 3:
        return [], np.zeros(number_of_points), [], []

    length_change = np.diff(
        length_values,
        prepend=length_values[0],
    )

    positive_mask = (
        length_change > positive_change_threshold
    )

    all_positive_runs = identify_true_runs(
        positive_mask
    )

    valid_positive_runs = [
        (start, end)
        for start, end in all_positive_runs
        if (end - start + 1) >= sustained_increase_points
    ]

    negative_mask = (
        length_values < negative_length_threshold
    )

    all_negative_runs = identify_true_runs(
        negative_mask
    )

    valid_negative_runs = [
        (start, end)
        for start, end in all_negative_runs
        if (end - start + 1) >= negative_run_points
    ]

    detected_cycles = []
    search_from_index = 0
    cycle_number = 1

    for positive_start, positive_end in valid_positive_runs:

        if positive_start < search_from_index:
            continue

        cycle_start = positive_start

        negative_runs_after_start = [
            (start, end)
            for start, end in valid_negative_runs
            if start > cycle_start
        ]

        if len(negative_runs_after_start) < 2:
            continue

        first_negative_run = negative_runs_after_start[0]
        second_negative_run = negative_runs_after_start[1]

        cycle_end = second_negative_run[0]

        number_of_cycle_points = (
            cycle_end - cycle_start + 1
        )

        if number_of_cycle_points < minimum_cycle_points:
            continue

        detected_cycles.append(
            {
                "Cycle": cycle_number,
                "Start_index": int(cycle_start),
                "End_index": int(cycle_end),
                "Start_sample": int(cycle_start),
                "End_sample": int(cycle_end),
                "Number_of_points": int(
                    number_of_cycle_points
                ),
                "First_negative_start": int(
                    first_negative_run[0]
                ),
                "Second_negative_start": int(
                    second_negative_run[0]
                ),
            }
        )

        search_from_index = cycle_end + 1
        cycle_number += 1

    return (
        detected_cycles,
        length_change,
        valid_positive_runs,
        valid_negative_runs,
    )


# ============================================================
# FORCE SMOOTHING
# ============================================================

def centred_moving_average(
    force_values: np.ndarray,
    window: int,
) -> np.ndarray:
    """Apply a centred moving-average filter."""
    return (
        pd.Series(force_values)
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )


def apply_force_smoothing(
    force_values: np.ndarray,
    moving_average_window: int,
    savgol_window: int,
    savgol_order: int,
    gaussian_sigma: float,
    butterworth_order: int,
    butterworth_cutoff: float,
) -> Dict[str, np.ndarray]:
    """
    Apply multiple smoothing methods to the complete force signal.

    Butterworth cutoff is entered as a proportion of the Nyquist
    frequency and must be between 0 and 1.
    """
    force_values = np.asarray(
        force_values,
        dtype=float,
    )

    number_of_points = len(force_values)

    smoothed_signals = {
        "Raw force": force_values.copy()
    }

    # Moving average
    moving_window = valid_odd_window(
        moving_average_window,
        number_of_points,
    )

    if moving_window is not None:
        smoothed_signals[
            f"Moving average, {moving_window} points"
        ] = centred_moving_average(
            force_values,
            moving_window,
        )

    # Savitzky-Golay
    valid_savgol_window = valid_odd_window(
        savgol_window,
        number_of_points,
    )

    if valid_savgol_window is not None:
        valid_savgol_order = min(
            int(savgol_order),
            valid_savgol_window - 1,
        )

        if valid_savgol_order >= 1:
            smoothed_signals[
                (
                    "Savitzky-Golay, "
                    f"{valid_savgol_window} points, "
                    f"order {valid_savgol_order}"
                )
            ] = savgol_filter(
                force_values,
                window_length=valid_savgol_window,
                polyorder=valid_savgol_order,
                mode="interp",
            )

    # Gaussian
    if gaussian_sigma > 0:
        smoothed_signals[
            f"Gaussian, sigma {gaussian_sigma:g}"
        ] = gaussian_filter1d(
            force_values,
            sigma=float(gaussian_sigma),
            mode="nearest",
        )

    # Butterworth
    if 0 < butterworth_cutoff < 1:
        try:
            filter_b, filter_a = butter(
                N=int(butterworth_order),
                Wn=float(butterworth_cutoff),
                btype="low",
                analog=False,
            )

            required_padding = (
                3 * (max(len(filter_a), len(filter_b)) - 1)
            )

            if number_of_points > required_padding:
                smoothed_signals[
                    (
                        "Butterworth, "
                        f"order {butterworth_order}, "
                        f"cutoff {butterworth_cutoff:g} Nyquist"
                    )
                ] = filtfilt(
                    filter_b,
                    filter_a,
                    force_values,
                )

        except ValueError:
            pass

    return smoothed_signals


# ============================================================
# PLOTTING
# ============================================================

def create_length_diagnostic_plot(
    data: pd.DataFrame,
    cycles: List[Dict[str, int]],
    negative_threshold: float,
) -> go.Figure:
    """Plot normalised length and detected cycle periods."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data["Sample"],
            y=data["Length_normalised"],
            mode="lines",
            name="Normalised length",
            line=dict(
                color="black",
                width=1.5,
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="grey",
        annotation_text="Zero",
    )

    figure.add_hline(
        y=negative_threshold,
        line_dash="dot",
        line_color="red",
        annotation_text="Negative threshold",
    )

    cycle_colours = [
        "rgba(31,119,180,0.18)",
        "rgba(255,127,14,0.18)",
        "rgba(44,160,44,0.18)",
        "rgba(214,39,40,0.18)",
        "rgba(148,103,189,0.18)",
    ]

    for cycle in cycles:
        cycle_number = cycle["Cycle"]
        start_index = cycle["Start_index"]
        end_index = cycle["End_index"]

        figure.add_vrect(
            x0=data.loc[start_index, "Sample"],
            x1=data.loc[end_index, "Sample"],
            fillcolor=cycle_colours[
                (cycle_number - 1) % len(cycle_colours)
            ],
            opacity=1,
            line_width=0,
            annotation_text=f"Cycle {cycle_number}",
            annotation_position="top left",
        )

    figure.update_layout(
        title="Normalised length and detected cycles",
        xaxis_title="Sample",
        yaxis_title="Normalised length",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )

    return figure


def create_force_comparison_plot(
    cycle_data: pd.DataFrame,
    signal_names: List[str],
    cycle_number: int,
) -> go.Figure:
    """Plot raw and smoothed force against sample."""
    figure = go.Figure()

    for signal_name in signal_names:
        column_name = (
            "Force_raw"
            if signal_name == "Raw force"
            else f"Force_{clean_column_name(signal_name)}"
        )

        line_width = (
            1 if signal_name == "Raw force" else 2
        )

        figure.add_trace(
            go.Scatter(
                x=cycle_data["Sample"],
                y=cycle_data[column_name],
                mode="lines",
                name=signal_name,
                line=dict(width=line_width),
            )
        )

    figure.update_layout(
        title=f"Cycle {cycle_number}: force smoothing comparison",
        xaxis_title="Sample",
        yaxis_title="Force",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )

    return figure


def create_work_loop_plot(
    cycle_data: pd.DataFrame,
    signal_names: List[str],
    cycle_number: int,
) -> go.Figure:
    """Plot force against normalised length."""
    figure = go.Figure()

    for signal_name in signal_names:
        column_name = (
            "Force_raw"
            if signal_name == "Raw force"
            else f"Force_{clean_column_name(signal_name)}"
        )

        line_width = (
            1 if signal_name == "Raw force" else 2
        )

        figure.add_trace(
            go.Scatter(
                x=cycle_data["Length_normalised"],
                y=cycle_data[column_name],
                mode="lines",
                name=signal_name,
                line=dict(width=line_width),
                customdata=cycle_data["Sample"],
                hovertemplate=(
                    "Length: %{x:.6g}<br>"
                    "Force: %{y:.6g}<br>"
                    "Sample: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=f"Cycle {cycle_number}: force-length work loop",
        xaxis_title="Normalised length",
        yaxis_title="Force",
        template="plotly_white",
        height=550,
    )

    return figure


# ============================================================
# WORK-LOOP METRICS
# ============================================================

def calculate_work_loop_metrics(
    cycle_data: pd.DataFrame,
    signal_names: List[str],
    cycle_number: int,
) -> List[Dict[str, float]]:
    """
    Calculate work-loop metrics for each force signal.

    The force-length integral is found using the trapezoidal rule.
    """
    length_values = cycle_data[
        "Length_normalised"
    ].to_numpy()

    metric_rows = []

    for signal_name in signal_names:
        column_name = (
            "Force_raw"
            if signal_name == "Raw force"
            else f"Force_{clean_column_name(signal_name)}"
        )

        force_values = cycle_data[
            column_name
        ].to_numpy()

        signed_integral = trapezoid(
            y=force_values,
            x=length_values,
        )

        metric_rows.append(
            {
                "Cycle": cycle_number,
                "Force_signal": signal_name,
                "Start_sample": int(
                    cycle_data["Sample"].iloc[0]
                ),
                "End_sample": int(
                    cycle_data["Sample"].iloc[-1]
                ),
                "Number_of_points": int(
                    len(cycle_data)
                ),
                "Minimum_length": float(
                    np.min(length_values)
                ),
                "Maximum_length": float(
                    np.max(length_values)
                ),
                "Length_range": float(
                    np.ptp(length_values)
                ),
                "Minimum_force": float(
                    np.min(force_values)
                ),
                "Maximum_force": float(
                    np.max(force_values)
                ),
                "Mean_force": float(
                    np.mean(force_values)
                ),
                "Signed_force_length_integral": float(
                    signed_integral
                ),
                "Absolute_force_length_integral": float(
                    abs(signed_integral)
                ),
            }
        )

    return metric_rows


# ============================================================
# LONG-FORM EXPORT
# ============================================================

def create_long_format_export(
    cycle_data: pd.DataFrame,
    signal_names: List[str],
    cycle_number: int,
) -> pd.DataFrame:
    """
    Create long-format data.

    This format is useful for Prism, R, Python, SPSS,
    mixed models, and grouped graphs.
    """
    output_frames = []

    for signal_name in signal_names:
        column_name = (
            "Force_raw"
            if signal_name == "Raw force"
            else f"Force_{clean_column_name(signal_name)}"
        )

        signal_frame = pd.DataFrame(
            {
                "Cycle": cycle_number,
                "Point_within_cycle": np.arange(
                    1,
                    len(cycle_data) + 1,
                ),
                "Original_sample": cycle_data[
                    "Sample"
                ].to_numpy(),
                "Length_raw": cycle_data[
                    "Length_raw"
                ].to_numpy(),
                "Length_normalised": cycle_data[
                    "Length_normalised"
                ].to_numpy(),
                "Force_processing": signal_name,
                "Force": cycle_data[
                    column_name
                ].to_numpy(),
            }
        )

        output_frames.append(signal_frame)

    return pd.concat(
        output_frames,
        ignore_index=True,
    )


# ============================================================
# ZIP EXPORT
# ============================================================

def create_results_zip(
    complete_data: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    metrics_data: pd.DataFrame,
    wide_cycle_data: pd.DataFrame,
    long_cycle_data: pd.DataFrame,
    individual_cycles: Dict[int, pd.DataFrame],
) -> bytes:
    """Create one ZIP file containing all exported CSV files."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        zip_file.writestr(
            "complete_processed_data.csv",
            complete_data.to_csv(index=False),
        )

        zip_file.writestr(
            "cycle_boundaries.csv",
            cycle_summary.to_csv(index=False),
        )

        zip_file.writestr(
            "work_loop_metrics.csv",
            metrics_data.to_csv(index=False),
        )

        zip_file.writestr(
            "all_cycles_wide_format.csv",
            wide_cycle_data.to_csv(index=False),
        )

        zip_file.writestr(
            "all_cycles_long_format.csv",
            long_cycle_data.to_csv(index=False),
        )

        for cycle_number, cycle_data in individual_cycles.items():
            zip_file.writestr(
                f"cycle_{cycle_number}_wide_format.csv",
                cycle_data.to_csv(index=False),
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Analysis settings")

st.sidebar.subheader("Length normalisation")

baseline_points = st.sidebar.number_input(
    "Number of final length points used for baseline",
    min_value=10,
    max_value=1_000_000,
    value=1000,
    step=10,
    help=(
        "The mean of the final selected length values is "
        "subtracted from every length value."
    ),
)

st.sidebar.subheader("Cycle detection")

sustained_increase_points = st.sidebar.number_input(
    "Points required for sustained positive increase",
    min_value=2,
    max_value=100_000,
    value=10,
    step=1,
)

positive_change_threshold = st.sidebar.number_input(
    "Minimum positive length change per sample",
    min_value=0.0,
    value=0.0,
    step=0.000001,
    format="%.8f",
)

negative_length_threshold = st.sidebar.number_input(
    "Negative-length threshold",
    value=0.0,
    step=0.000001,
    format="%.8f",
)

negative_run_points = st.sidebar.number_input(
    "Points required for a negative run",
    min_value=1,
    max_value=100_000,
    value=3,
    step=1,
)

minimum_cycle_points = st.sidebar.number_input(
    "Minimum points in a cycle",
    min_value=3,
    max_value=1_000_000,
    value=50,
    step=1,
)

st.sidebar.subheader("Force smoothing")

moving_average_window = st.sidebar.number_input(
    "Moving-average window",
    min_value=3,
    max_value=100_001,
    value=21,
    step=2,
)

savgol_window = st.sidebar.number_input(
    "Savitzky-Golay window",
    min_value=3,
    max_value=100_001,
    value=21,
    step=2,
)

savgol_order = st.sidebar.number_input(
    "Savitzky-Golay polynomial order",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
)

gaussian_sigma = st.sidebar.number_input(
    "Gaussian sigma",
    min_value=0.1,
    max_value=1000.0,
    value=3.0,
    step=0.1,
)

butterworth_order = st.sidebar.number_input(
    "Butterworth filter order",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
)

butterworth_cutoff = st.sidebar.slider(
    "Butterworth cutoff, proportion of Nyquist",
    min_value=0.001,
    max_value=0.999,
    value=0.050,
    step=0.001,
)

st.sidebar.caption(
    "A lower Butterworth cutoff produces stronger smoothing."
)


# ============================================================
# UPLOAD FILE
# ============================================================

uploaded_file = st.file_uploader(
    "Drag and drop a comma-delimited .dat file",
    type=["dat", "csv", "txt"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info(
        "Upload a DAT, CSV, or TXT file to begin."
    )
    st.stop()


# ============================================================
# IMPORT AND PROCESS DATA
# ============================================================

try:
    data = read_dat_file(uploaded_file)

except Exception as error:
    st.error(
        f"The uploaded file could not be processed: {error}"
    )
    st.stop()


# ============================================================
# NORMALISE LENGTH
# ============================================================

actual_baseline_points = min(
    int(baseline_points),
    len(data),
)

if len(data) < int(baseline_points):
    st.warning(
        f"The file contains {len(data):,} valid data points. "
        f"The baseline will therefore use all "
        f"{actual_baseline_points:,} available points."
    )

length_baseline = float(
    data["Length_raw"]
    .iloc[-actual_baseline_points:]
    .mean()
)

data["Length_normalised"] = (
    data["Length_raw"] - length_baseline
)


# ============================================================
# DETECT CYCLES
# ============================================================

(
    cycles,
    length_change,
    positive_runs,
    negative_runs,
) = detect_length_cycles(
    length_values=data["Length_normalised"].to_numpy(),
    sustained_increase_points=int(
        sustained_increase_points
    ),
    positive_change_threshold=float(
        positive_change_threshold
    ),
    negative_length_threshold=float(
        negative_length_threshold
    ),
    negative_run_points=int(
        negative_run_points
    ),
    minimum_cycle_points=int(
        minimum_cycle_points
    ),
)

data["Length_change_per_sample"] = length_change

data["Cycle"] = pd.Series(
    pd.NA,
    index=data.index,
    dtype="Int64",
)

data["Point_within_cycle"] = pd.Series(
    pd.NA,
    index=data.index,
    dtype="Int64",
)

for cycle in cycles:
    start_index = cycle["Start_index"]
    end_index = cycle["End_index"]

    data.loc[
        start_index:end_index,
        "Cycle",
    ] = cycle["Cycle"]

    data.loc[
        start_index:end_index,
        "Point_within_cycle",
    ] = np.arange(
        1,
        end_index - start_index + 2,
    )


# ============================================================
# APPLY FORCE SMOOTHING
# ============================================================

smoothed_signals = apply_force_smoothing(
    force_values=data["Force_raw"].to_numpy(),
    moving_average_window=int(
        moving_average_window
    ),
    savgol_window=int(savgol_window),
    savgol_order=int(savgol_order),
    gaussian_sigma=float(gaussian_sigma),
    butterworth_order=int(butterworth_order),
    butterworth_cutoff=float(
        butterworth_cutoff
    ),
)

signal_names = list(smoothed_signals.keys())

for signal_name, signal_values in smoothed_signals.items():

    if signal_name == "Raw force":
        continue

    export_column = (
        f"Force_{clean_column_name(signal_name)}"
    )

    data[export_column] = signal_values


# ============================================================
# ANALYSIS SUMMARY
# ============================================================

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

summary_col1.metric(
    "Valid data points",
    f"{len(data):,}",
)

summary_col2.metric(
    "Length baseline",
    f"{length_baseline:.6g}",
)

summary_col3.metric(
    "Cycles detected",
    len(cycles),
)

summary_col4.metric(
    "Force signals",
    len(signal_names),
)

st.caption(
    f"The length baseline was calculated as the mean of the "
    f"final {actual_baseline_points:,} valid length values. "
    f"This value was subtracted from every length value."
)

with st.expander("Preview imported and processed data"):
    st.dataframe(
        data.head(500),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# CYCLE-DETECTION DIAGNOSTIC
# ============================================================

st.header("1. Length normalisation and cycle detection")

diagnostic_figure = create_length_diagnostic_plot(
    data=data,
    cycles=cycles,
    negative_threshold=float(
        negative_length_threshold
    ),
)

st.plotly_chart(
    diagnostic_figure,
    width="stretch",
)

if len(cycles) == 0:
    st.error(
        "No complete work-loop cycles were detected with the "
        "current settings."
    )

    st.markdown(
        """
Possible adjustments:

- Reduce the number of points required for a sustained increase.
- Reduce the minimum positive length-change threshold.
- Increase the negative-length threshold slightly.
- Reduce the number of points required for a negative run.
- Reduce the minimum number of points in a cycle.
- Check whether the final-point baseline correctly positions the length signal around zero.
"""
    )

    st.download_button(
        label="Download processed data without detected cycles",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="processed_data_no_cycles.csv",
        mime="text/csv",
    )

    st.stop()


cycle_summary = pd.DataFrame(cycles)

st.subheader("Detected cycle boundaries")

st.dataframe(
    cycle_summary,
    width="stretch",
    hide_index=True,
)


# ============================================================
# DISPLAY INDIVIDUAL WORK LOOPS
# ============================================================

st.header("2. Force smoothing and work loops")

all_metric_rows = []
wide_cycle_frames = []
long_cycle_frames = []
individual_cycle_data = {}

for cycle in cycles:

    cycle_number = cycle["Cycle"]
    start_index = cycle["Start_index"]
    end_index = cycle["End_index"]

    cycle_data = data.loc[
        start_index:end_index
    ].copy()

    cycle_data["Point_within_cycle"] = np.arange(
        1,
        len(cycle_data) + 1,
    )

    individual_cycle_data[
        cycle_number
    ] = cycle_data.copy()

    wide_cycle_frames.append(
        cycle_data.copy()
    )

    long_cycle_data = create_long_format_export(
        cycle_data=cycle_data,
        signal_names=signal_names,
        cycle_number=cycle_number,
    )

    long_cycle_frames.append(
        long_cycle_data
    )

    all_metric_rows.extend(
        calculate_work_loop_metrics(
            cycle_data=cycle_data,
            signal_names=signal_names,
            cycle_number=cycle_number,
        )
    )

    st.subheader(f"Cycle {cycle_number}")

    information_col1, information_col2, information_col3 = st.columns(3)

    information_col1.metric(
        "Start sample",
        int(cycle_data["Sample"].iloc[0]),
    )

    information_col2.metric(
        "End sample",
        int(cycle_data["Sample"].iloc[-1]),
    )

    information_col3.metric(
        "Number of points",
        len(cycle_data),
    )

    (
        force_tab,
        work_loop_tab,
        data_tab,
    ) = st.tabs(
        [
            "Force smoothing",
            "Work loop",
            "Exportable data",
        ]
    )

    with force_tab:
        force_figure = create_force_comparison_plot(
            cycle_data=cycle_data,
            signal_names=signal_names,
            cycle_number=cycle_number,
        )

        st.plotly_chart(
            force_figure,
            width="stretch",
            key=f"force_plot_{cycle_number}",
        )

    with work_loop_tab:
        work_loop_figure = create_work_loop_plot(
            cycle_data=cycle_data,
            signal_names=signal_names,
            cycle_number=cycle_number,
        )

        st.plotly_chart(
            work_loop_figure,
            width="stretch",
            key=f"work_loop_{cycle_number}",
        )

    with data_tab:
        st.markdown(
            """
**Wide-format data** contains one row per sample and a separate
column for each force-smoothing method.

**Long-format data** contains a separate set of rows for each
force-smoothing method. This is often more convenient for grouped
graphs and statistical software.
"""
        )

        st.dataframe(
            cycle_data,
            width="stretch",
            hide_index=True,
        )

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            st.download_button(
                label=f"Download Cycle {cycle_number}, wide format",
                data=cycle_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    f"cycle_{cycle_number}_wide_format.csv"
                ),
                mime="text/csv",
                key=f"wide_download_{cycle_number}",
            )

        with download_col2:
            st.download_button(
                label=f"Download Cycle {cycle_number}, long format",
                data=long_cycle_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    f"cycle_{cycle_number}_long_format.csv"
                ),
                mime="text/csv",
                key=f"long_download_{cycle_number}",
            )


# ============================================================
# COMBINE RESULTS
# ============================================================

all_cycles_wide = pd.concat(
    wide_cycle_frames,
    ignore_index=True,
)

all_cycles_long = pd.concat(
    long_cycle_frames,
    ignore_index=True,
)

metrics_data = pd.DataFrame(
    all_metric_rows
)


# ============================================================
# METRICS
# ============================================================

st.header("3. Work-loop metrics")

st.dataframe(
    metrics_data,
    width="stretch",
    hide_index=True,
)

st.caption(
    "The force-length integral is calculated using the "
    "trapezoidal rule. Its units are the force unit multiplied "
    "by the length unit. The sign depends on the direction of "
    "the work loop and the force and length sign conventions."
)


# ============================================================
# DOWNLOAD ALL RESULTS
# ============================================================

st.header("4. Export data for your own graphs")

st.markdown(
    """
The exports contain the original force and length values,
normalised length, point-to-point length change, cycle allocation,
point number within each cycle, and every smoothed force signal.
"""
)

download_row1_col1, download_row1_col2 = st.columns(2)

with download_row1_col1:
    st.download_button(
        label="Download complete processed recording",
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="complete_processed_data.csv",
        mime="text/csv",
    )

with download_row1_col2:
    st.download_button(
        label="Download detected cycle boundaries",
        data=cycle_summary.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="cycle_boundaries.csv",
        mime="text/csv",
    )

download_row2_col1, download_row2_col2 = st.columns(2)

with download_row2_col1:
    st.download_button(
        label="Download all cycles, wide format",
        data=all_cycles_wide.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="all_cycles_wide_format.csv",
        mime="text/csv",
    )

with download_row2_col2:
    st.download_button(
        label="Download all cycles, long format",
        data=all_cycles_long.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="all_cycles_long_format.csv",
        mime="text/csv",
    )

download_row3_col1, download_row3_col2 = st.columns(2)

with download_row3_col1:
    st.download_button(
        label="Download work-loop metrics",
        data=metrics_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="work_loop_metrics.csv",
        mime="text/csv",
    )

results_zip = create_results_zip(
    complete_data=data,
    cycle_summary=cycle_summary,
    metrics_data=metrics_data,
    wide_cycle_data=all_cycles_wide,
    long_cycle_data=all_cycles_long,
    individual_cycles=individual_cycle_data,
)

with download_row3_col2:
    st.download_button(
        label="Download all results as one ZIP file",
        data=results_zip,
        file_name="work_loop_analysis_results.zip",
        mime="application/zip",
    )
