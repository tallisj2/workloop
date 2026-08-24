import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, savgol_filter
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
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

1. **Column 1:** Force
2. **Column 2:** Length

The app will normalise length, identify individual length-change cycles,
apply alternative smoothing methods to force, and plot force against length
for each detected work loop.
"""
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_odd(value: int) -> int:
    """
    Ensure that a value is an odd integer.
    """
    value = int(value)

    if value < 3:
        value = 3

    if value % 2 == 0:
        value += 1

    return value


def safe_window_length(requested_window: int, data_length: int) -> int | None:
    """
    Return a valid odd window length that does not exceed the data length.
    Returns None if the data segment is too short.
    """
    if data_length < 3:
        return None

    window = min(make_odd(requested_window), data_length)

    if window % 2 == 0:
        window -= 1

    if window < 3:
        return None

    return window


def read_dat_file(uploaded_file) -> pd.DataFrame:
    """
    Read a comma-delimited .dat file.

    The function:
    - assumes there is no header;
    - uses the first two columns only;
    - converts both columns to numeric;
    - removes rows that cannot be interpreted as numeric data.
    """
    raw_bytes = uploaded_file.getvalue()

    try:
        data = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=",",
            header=None,
            comment="#",
            engine="python",
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=",",
            header=None,
            comment="#",
            engine="python",
            encoding="latin-1",
        )

    if data.shape[1] < 2:
        raise ValueError(
            "The uploaded file must contain at least two comma-delimited columns."
        )

    data = data.iloc[:, :2].copy()
    data.columns = ["Force", "Length"]

    data["Force"] = pd.to_numeric(data["Force"], errors="coerce")
    data["Length"] = pd.to_numeric(data["Length"], errors="coerce")

    data = data.dropna(subset=["Force", "Length"]).reset_index(drop=True)

    if len(data) < 3:
        raise ValueError(
            "Fewer than three valid numeric rows were found in the uploaded file."
        )

    data.insert(0, "Sample", np.arange(len(data), dtype=int))

    return data


def find_boolean_runs(boolean_values: np.ndarray) -> list[tuple[int, int]]:
    """
    Identify consecutive True runs in a Boolean array.

    Returns
    -------
    list of tuples
        Each tuple contains:
        (start_index, end_index)

    Both indices are inclusive.
    """
    boolean_values = np.asarray(boolean_values, dtype=bool)

    if len(boolean_values) == 0:
        return []

    padded = np.concatenate(
        [
            np.array([False]),
            boolean_values,
            np.array([False]),
        ]
    )

    changes = np.diff(padded.astype(int))

    run_starts = np.where(changes == 1)[0]
    run_ends = np.where(changes == -1)[0] - 1

    return list(zip(run_starts, run_ends))


def detect_cycles(
    length_values: np.ndarray,
    sustained_points: int,
    positive_threshold: float,
    negative_threshold: float,
    negative_run_points: int,
    minimum_cycle_points: int,
) -> tuple[list[dict], np.ndarray, list[tuple[int, int]]]:
    """
    Detect distinct length-change cycles.

    Operational definition used by this app
    ---------------------------------------
    Cycle start:
        The first point in a sequence where the point-to-point change in
        length exceeds `positive_threshold` for at least
        `sustained_points` consecutive samples.

    Cycle end:
        The start of the second distinct negative-length run after the
        cycle begins.

    A negative-length run is a sequence containing at least
    `negative_run_points` consecutive samples below
    `negative_threshold`.

    After a cycle has ended, the app searches for the next sustained
    positive increase.
    """
    length_values = np.asarray(length_values, dtype=float)

    if len(length_values) < 3:
        return [], np.full(len(length_values), np.nan), []

    # First derivative expressed as change per sample.
    length_change = np.diff(length_values, prepend=length_values[0])

    # Find sustained positive-change runs.
    positive_mask = length_change > positive_threshold
    positive_runs = find_boolean_runs(positive_mask)

    valid_positive_runs = [
        (start, end)
        for start, end in positive_runs
        if (end - start + 1) >= sustained_points
    ]

    # Find sustained negative-length runs.
    negative_mask = length_values < negative_threshold
    negative_runs = find_boolean_runs(negative_mask)

    valid_negative_runs = [
        (start, end)
        for start, end in negative_runs
        if (end - start + 1) >= negative_run_points
    ]

    cycles = []
    search_from = 0
    cycle_number = 1

    for positive_start, positive_end in valid_positive_runs:

        # Do not use a positive run that belongs to an already detected cycle.
        if positive_start < search_from:
            continue

        cycle_start = positive_start

        # Find negative runs that begin after the cycle has started.
        negative_runs_after_start = [
            run
            for run in valid_negative_runs
            if run[0] > cycle_start
        ]

        # The requested definition requires the second negative occurrence.
        if len(negative_runs_after_start) < 2:
            break

        first_negative_run = negative_runs_after_start[0]
        second_negative_run = negative_runs_after_start[1]

        cycle_end = second_negative_run[0]

        cycle_length = cycle_end - cycle_start + 1

        if cycle_length >= minimum_cycle_points:
            cycles.append(
                {
                    "Cycle": cycle_number,
                    "Start_index": int(cycle_start),
                    "End_index": int(cycle_end),
                    "Number_of_points": int(cycle_length),
                    "First_negative_start": int(first_negative_run[0]),
                    "Second_negative_start": int(second_negative_run[0]),
                }
            )

            cycle_number += 1
            search_from = cycle_end + 1

    return cycles, length_change, valid_negative_runs


def moving_average(signal_values: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a centred moving-average filter.
    """
    return (
        pd.Series(signal_values)
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )


def apply_smoothing(
    force_values: np.ndarray,
    moving_average_window: int,
    savgol_window: int,
    savgol_polynomial: int,
    gaussian_sigma: float,
    butterworth_order: int,
    butterworth_cutoff: float,
) -> dict[str, np.ndarray]:
    """
    Apply multiple smoothing algorithms to the entire force signal.

    Butterworth cutoff is expressed as a proportion of the Nyquist
    frequency, so it must be between 0 and 1.
    """
    force_values = np.asarray(force_values, dtype=float)
    number_of_points = len(force_values)

    smoothed = {
        "Raw force": force_values.copy()
    }

    # --------------------------------------------------------
    # Moving average
    # --------------------------------------------------------

    moving_window = safe_window_length(
        moving_average_window,
        number_of_points,
    )

    if moving_window is not None:
        smoothed[
            f"Moving average ({moving_window} points)"
        ] = moving_average(
            force_values,
            moving_window,
        )

    # --------------------------------------------------------
    # Savitzky-Golay
    # --------------------------------------------------------

    valid_savgol_window = safe_window_length(
        savgol_window,
        number_of_points,
    )

    if valid_savgol_window is not None:
        valid_polynomial = min(
            int(savgol_polynomial),
            valid_savgol_window - 1,
        )

        if valid_polynomial >= 1:
            smoothed[
                f"Savitzky-Golay ({valid_savgol_window} points, "
                f"order {valid_polynomial})"
            ] = savgol_filter(
                force_values,
                window_length=valid_savgol_window,
                polyorder=valid_polynomial,
                mode="interp",
            )

    # --------------------------------------------------------
    # Gaussian
    # --------------------------------------------------------

    if gaussian_sigma > 0:
        smoothed[
            f"Gaussian (sigma {gaussian_sigma:g})"
        ] = gaussian_filter1d(
            force_values,
            sigma=gaussian_sigma,
            mode="nearest",
        )

    # --------------------------------------------------------
    # Low-pass Butterworth
    # --------------------------------------------------------

    try:
        b, a = butter(
            N=int(butterworth_order),
            Wn=float(butterworth_cutoff),
            btype="low",
            analog=False,
        )

        pad_length = 3 * (max(len(a), len(b)) - 1)

        if number_of_points > pad_length:
            smoothed[
                f"Butterworth (order {butterworth_order}, "
                f"cutoff {butterworth_cutoff:g} × Nyquist)"
            ] = filtfilt(
                b,
                a,
                force_values,
            )

    except ValueError:
        # Invalid Butterworth settings are handled in the interface.
        pass

    return smoothed


def create_length_diagnostic_figure(
    data: pd.DataFrame,
    cycles: list[dict],
    negative_threshold: float,
) -> go.Figure:
    """
    Create a length-versus-sample diagnostic plot with detected cycles.
    """
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data["Sample"],
            y=data["Length_normalised"],
            mode="lines",
            name="Normalised length",
            line=dict(color="black", width=1.5),
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

    colours = [
        "rgba(31, 119, 180, 0.15)",
        "rgba(255, 127, 14, 0.15)",
        "rgba(44, 160, 44, 0.15)",
        "rgba(214, 39, 40, 0.15)",
        "rgba(148, 103, 189, 0.15)",
    ]

    for cycle in cycles:
        cycle_number = cycle["Cycle"]
        start_index = cycle["Start_index"]
        end_index = cycle["End_index"]

        figure.add_vrect(
            x0=data.loc[start_index, "Sample"],
            x1=data.loc[end_index, "Sample"],
            fillcolor=colours[(cycle_number - 1) % len(colours)],
            opacity=1,
            line_width=0,
            annotation_text=f"Cycle {cycle_number}",
            annotation_position="top left",
        )

    figure.update_layout(
        title="Cycle-detection diagnostic",
        xaxis_title="Sample",
        yaxis_title="Normalised length",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return figure


def create_force_time_figure(
    cycle_data: pd.DataFrame,
    smoothed_signals: dict[str, np.ndarray],
    start_index: int,
    end_index: int,
    cycle_number: int,
) -> go.Figure:
    """
    Plot force against sample for one detected cycle.
    """
    figure = go.Figure()

    for signal_name, signal_values in smoothed_signals.items():

        local_values = signal_values[start_index:end_index + 1]

        line_width = 1 if signal_name == "Raw force" else 2

        figure.add_trace(
            go.Scatter(
                x=cycle_data["Sample"],
                y=local_values,
                mode="lines",
                name=signal_name,
                line=dict(width=line_width),
            )
        )

    figure.update_layout(
        title=f"Cycle {cycle_number}: force smoothing comparison",
        xaxis_title="Sample",
        yaxis_title="Force",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return figure


def create_work_loop_figure(
    cycle_data: pd.DataFrame,
    smoothed_signals: dict[str, np.ndarray],
    start_index: int,
    end_index: int,
    cycle_number: int,
) -> go.Figure:
    """
    Plot force against normalised length for one detected cycle.
    """
    figure = go.Figure()

    length_values = cycle_data["Length_normalised"].to_numpy()

    for signal_name, signal_values in smoothed_signals.items():

        local_force = signal_values[start_index:end_index + 1]

        line_width = 1 if signal_name == "Raw force" else 2

        figure.add_trace(
            go.Scatter(
                x=length_values,
                y=local_force,
                mode="lines",
                name=signal_name,
                line=dict(width=line_width),
                customdata=cycle_data["Sample"],
                hovertemplate=(
                    "Length: %{x:.6g}<br>"
                    "Force: %{y:.6g}<br>"
                    "Sample: %{customdata}<extra></extra>"
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


def calculate_loop_metrics(
    cycle_data: pd.DataFrame,
    smoothed_signals: dict[str, np.ndarray],
    start_index: int,
    end_index: int,
    cycle_number: int,
) -> list"""
    Calculate descriptive metrics for each smoothing method.

    Signed loop integral:
        Integral of force with respect to length over the recorded path.

    Absolute loop integral:
        Absolute value of the signed loop integral.

    The physical interpretation depends on the force and length units
    and on the sign convention used by the recording system.
    """
    length_values = cycle_data["Length_normalised"].to_numpy()
    metric_rows = []

    for signal_name, signal_values in smoothed_signals.items():

        local_force = signal_values[start_index:end_index + 1]

        signed_integral = trapezoid(
            y=local_force,
            x=length_values,
        )

        metric_rows.append(
            {
                "Cycle": cycle_number,
                "Force_signal": signal_name,
                "Start_sample": int(cycle_data["Sample"].iloc[0]),
                "End_sample": int(cycle_data["Sample"].iloc[-1]),
                "Number_of_points": len(cycle_data),
                "Minimum_length": np.min(length_values),
                "Maximum_length": np.max(length_values),
                "Length_range": np.ptp(length_values),
                "Minimum_force": np.min(local_force),
                "Maximum_force": np.max(local_force),
                "Mean_force": np.mean(local_force),
                "Signed_force_length_integral": signed_integral,
                "Absolute_force_length_integral": abs(signed_integral),
            }
        )

    return metric_rows


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Analysis settings")

st.sidebar.subheader("Length normalisation")

baseline_points = st.sidebar.number_input(
    "Number of final points used for baseline",
    min_value=10,
    max_value=100000,
    value=1000,
    step=10,
    help=(
        "The mean of these final length values is subtracted from every "
        "length value."
    ),
)

st.sidebar.subheader("Cycle detection")

sustained_points = st.sidebar.number_input(
    "Samples required for a sustained positive increase",
    min_value=2,
    max_value=10000,
    value=10,
    step=1,
)

positive_threshold = st.sidebar.number_input(
    "Minimum positive change per sample",
    min_value=0.0,
    value=0.0,
    format="%.8f",
    help=(
        "A point is considered to be increasing when the change in length "
        "is greater than this value."
    ),
)

negative_threshold = st.sidebar.number_input(
    "Negative-length threshold",
    value=0.0,
    format="%.8f",
    help=(
        "Length must fall below this value to count as a negative occurrence."
    ),
)

negative_run_points = st.sidebar.number_input(
    "Minimum samples in a negative run",
    min_value=1,
    max_value=10000,
    value=3,
    step=1,
    help=(
        "This prevents a single noisy sample below zero from being counted "
        "as a negative occurrence."
    ),
)

minimum_cycle_points = st.sidebar.number_input(
    "Minimum number of samples per cycle",
    min_value=3,
    max_value=100000,
    value=50,
    step=1,
)

st.sidebar.subheader("Force smoothing")

moving_average_window = st.sidebar.number_input(
    "Moving-average window",
    min_value=3,
    max_value=10001,
    value=21,
    step=2,
)

savgol_window = st.sidebar.number_input(
    "Savitzky-Golay window",
    min_value=3,
    max_value=10001,
    value=21,
    step=2,
)

savgol_polynomial = st.sidebar.number_input(
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
    "Butterworth cutoff as proportion of Nyquist frequency",
    min_value=0.001,
    max_value=0.999,
    value=0.05,
    step=0.001,
    help=(
        "A smaller value produces stronger smoothing. A value of 0.05 "
        "means 5% of the Nyquist frequency."
    ),
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Drag and drop a comma-delimited .dat file",
    type=["dat", "txt", "csv"],
    accept_multiple_files=False,
    help="The first column must contain force and the second must contain length.",
)

if uploaded_file is None:
    st.info("Upload a `.dat` file to begin the analysis.")
    st.stop()


# ============================================================
# READ AND VALIDATE DATA
# ============================================================

try:
    data = read_dat_file(uploaded_file)
except Exception as error:
    st.error(f"The file could not be read: {error}")
    st.stop()

number_of_baseline_points = min(
    int(baseline_points),
    len(data),
)

if len(data) < int(baseline_points):
    st.warning(
        f"The file contains {len(data):,} valid rows, which is fewer than "
        f"the requested {int(baseline_points):,} baseline points. "
        f"All {len(data):,} rows will therefore be used to calculate the baseline."
    )

length_baseline = data["Length"].iloc[
    -number_of_baseline_points:
].mean()

data["Length_normalised"] = (
    data["Length"] - length_baseline
)


# ============================================================
# DETECT CYCLES
# ============================================================

cycles, length_change, negative_runs = detect_cycles(
    length_values=data["Length_normalised"].to_numpy(),
    sustained_points=int(sustained_points),
    positive_threshold=float(positive_threshold),
    negative_threshold=float(negative_threshold),
    negative_run_points=int(negative_run_points),
    minimum_cycle_points=int(minimum_cycle_points),
)

data["Length_change"] = length_change
data["Cycle"] = pd.Series(pd.NA, index=data.index, dtype="Int64")

for cycle in cycles:
    data.loc[
        cycle["Start_index"]:cycle["End_index"],
        "Cycle",
    ] = cycle["Cycle"]


# ============================================================
# APPLY FORCE SMOOTHING
# ============================================================

smoothed_signals = apply_smoothing(
    force_values=data["Force"].to_numpy(),
    moving_average_window=int(moving_average_window),
    savgol_window=int(savgol_window),
    savgol_polynomial=int(savgol_polynomial),
    gaussian_sigma=float(gaussian_sigma),
    butterworth_order=int(butterworth_order),
    butterworth_cutoff=float(butterworth_cutoff),
)

for signal_name, signal_values in smoothed_signals.items():
    safe_column_name = (
        signal_name
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("×", "x")
    )

    data[f"Force_{safe_column_name}"] = signal_values


# ============================================================
# SUMMARY
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
    "Smoothing methods",
    len(smoothed_signals) - 1,
)

st.caption(
    f"The length baseline is the mean of the final "
    f"{number_of_baseline_points:,} valid length values."
)

with st.expander("Preview imported and processed data"):
    st.dataframe(
        data.head(200),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# DIAGNOSTIC PLOT
# ============================================================

st.header("1. Length normalisation and cycle detection")

diagnostic_figure = create_length_diagnostic_figure(
    data=data,
    cycles=cycles,
    negative_threshold=float(negative_threshold),
)

st.plotly_chart(
    diagnostic_figure,
    width="stretch",
)

if len(cycles) == 0:
    st.error(
        "No complete cycles were detected using the current settings."
    )

    st.markdown(
        """
Try the following adjustments:

- Reduce **Samples required for a sustained positive increase**.
- Reduce **Minimum positive change per sample**.
- Increase the **Negative-length threshold** slightly if the signal does not cross zero cleanly.
- Reduce **Minimum samples in a negative run**.
- Reduce **Minimum number of samples per cycle**.
- Check whether subtracting the final 1,000-point mean has positioned the signal around the intended zero.
"""
    )

    processed_csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download processed data",
        data=processed_csv,
        file_name="processed_force_length_data.csv",
        mime="text/csv",
    )

    st.stop()


cycle_table = pd.DataFrame(cycles)

st.subheader("Detected cycle boundaries")

st.dataframe(
    cycle_table,
    width="stretch",
    hide_index=True,
)


# ============================================================
# WORK LOOP PLOTS
# ============================================================

st.header("2. Force smoothing and work loops")

all_metric_rows = []

for cycle in cycles:

    cycle_number = cycle["Cycle"]
    start_index = cycle["Start_index"]
    end_index = cycle["End_index"]

    cycle_data = data.loc[start_index:end_index].copy()

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
        "Points",
        len(cycle_data),
    )

    force_time_figure = create_force_time_figure(
        cycle_data=cycle_data,
        smoothed_signals=smoothed_signals,
        start_index=start_index,
        end_index=end_index,
        cycle_number=cycle_number,
    )

    work_loop_figure = create_work_loop_figure(
        cycle_data=cycle_data,
        smoothed_signals=smoothed_signals,
        start_index=start_index,
        end_index=end_index,
        cycle_number=cycle_number,
    )

    time_tab, work_loop_tab, data_tab = st.tabs(
        [
            "Force smoothing",
            "Force-length work loop",
            "Cycle data",
        ]
    )

    with time_tab:
        st.plotly_chart(
            force_time_figure,
            width="stretch",
            key=f"force_time_{cycle_number}",
        )

    with work_loop_tab:
        st.plotly_chart(
            work_loop_figure,
            width="stretch",
            key=f"work_loop_{cycle_number}",
        )

    with data_tab:
        cycle_output_columns = [
            "Sample",
            "Force",
            "Length",
            "Length_normalised",
            "Length_change",
            "Cycle",
        ]

        smoothed_columns = [
            column
            for column in data.columns
            if column.startswith("Force_")
        ]

        cycle_output = cycle_data[
            cycle_output_columns + smoothed_columns
        ]

        st.dataframe(
            cycle_output,
            width="stretch",
            hide_index=True,
        )

        cycle_csv = cycle_output.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=f"Download Cycle {cycle_number} data",
            data=cycle_csv,
            file_name=f"work_loop_cycle_{cycle_number}.csv",
            mime="text/csv",
            key=f"download_cycle_{cycle_number}",
        )

    cycle_metric_rows = calculate_loop_metrics(
        cycle_data=cycle_data,
        smoothed_signals=smoothed_signals,
        start_index=start_index,
        end_index=end_index,
        cycle_number=cycle_number,
    )

    all_metric_rows.extend(cycle_metric_rows)


# ============================================================
# RESULTS AND DOWNLOADS
# ============================================================

st.header("3. Work-loop metrics")

metrics_data = pd.DataFrame(all_metric_rows)

st.dataframe(
    metrics_data,
    width="stretch",
    hide_index=True,
)

st.caption(
    "The force-length integral is calculated using the trapezoidal rule. "
    "Its units are the force unit multiplied by the length unit. "
    "Its sign depends on the direction of the loop and the sign convention "
    "used by the recording system."
)

st.header("4. Download results")

download_col1, download_col2, download_col3 = st.columns(3)

processed_csv = data.to_csv(index=False).encode("utf-8")

cycle_summary_csv = cycle_table.to_csv(index=False).encode("utf-8")

metrics_csv = metrics_data.to_csv(index=False).encode("utf-8")

with download_col1:
    st.download_button(
        label="Download all processed data",
        data=processed_csv,
        file_name="processed_force_length_data.csv",
        mime="text/csv",
    )

with download_col2:
    st.download_button(
        label="Download cycle boundaries",
        data=cycle_summary_csv,
        file_name="detected_cycle_boundaries.csv",
        mime="text/csv",
    )

with download_col3:
    st.download_button(
        label="Download work-loop metrics",
        data=metrics_csv,
        file_name="work_loop_metrics.csv",
        mime="text/csv",
    )
