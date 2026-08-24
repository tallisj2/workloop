import io
import re
import zipfile

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

st.write(
    "Upload a comma-delimited DAT file with force in column 1 "
    "and length in column 2."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def odd_window(requested, number_of_points):

    if number_of_points < 3:
        return None

    window = max(3, int(requested))

    if window % 2 == 0:
        window += 1

    if number_of_points % 2 == 1:
        maximum_window = number_of_points
    else:
        maximum_window = number_of_points - 1

    window = min(window, maximum_window)

    if window < 3:
        return None

    return window


def safe_name(text):

    cleaned_name = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        text,
    )

    return cleaned_name.strip("_")


# ============================================================
# FILE IMPORT
# ============================================================

def read_data(uploaded_file):

    raw_file = uploaded_file.getvalue()

    imported_data = None
    last_error = None

    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    for encoding in encodings:

        try:

            imported_data = pd.read_csv(
                io.BytesIO(raw_file),
                sep=",",
                header=None,
                comment="#",
                encoding=encoding,
                engine="python",
                skip_blank_lines=True,
            )

            break

        except Exception as error:

            last_error = error

    if imported_data is None:

        raise ValueError(
            "The file could not be read. "
            + str(last_error)
        )

    if imported_data.shape[1] < 2:

        raise ValueError(
            "The file must contain at least two "
            "comma-delimited columns."
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
        subset=[
            "Force_raw",
            "Length_raw",
        ]
    ).reset_index(drop=True)

    if len(imported_data) < 3:

        raise ValueError(
            "Fewer than three valid numeric rows were found."
        )

    imported_data.insert(
        0,
        "Sample",
        np.arange(
            len(imported_data),
            dtype=int,
        ),
    )

    return imported_data


# ============================================================
# IDENTIFY RUNS OF TRUE VALUES
# ============================================================

def true_runs(boolean_mask):

    boolean_mask = np.asarray(
        boolean_mask,
        dtype=bool,
    )

    padded_mask = np.concatenate(
        [
            np.array([False]),
            boolean_mask,
            np.array([False]),
        ]
    )

    changes = np.diff(
        padded_mask.astype(int)
    )

    run_starts = np.where(
        changes == 1
    )[0]

    run_ends = (
        np.where(changes == -1)[0] - 1
    )

    runs = list(
        zip(
            run_starts.tolist(),
            run_ends.tolist(),
        )
    )

    return runs


# ============================================================
# CYCLE DETECTION
# ============================================================

def detect_cycles(
    length_values,
    sustained_points,
    positive_threshold,
    negative_threshold,
    negative_run_points,
    minimum_cycle_points,
):

    length_values = np.asarray(
        length_values,
        dtype=float,
    )

    length_change = np.diff(
        length_values,
        prepend=length_values[0],
    )

    positive_mask = (
        length_change > positive_threshold
    )

    all_positive_runs = true_runs(
        positive_mask
    )

    valid_positive_runs = []

    for run_start, run_end in all_positive_runs:

        number_of_points = (
            run_end - run_start + 1
        )

        if number_of_points >= sustained_points:

            valid_positive_runs.append(
                (
                    run_start,
                    run_end,
                )
            )

    negative_mask = (
        length_values < negative_threshold
    )

    all_negative_runs = true_runs(
        negative_mask
    )

    valid_negative_runs = []

    for run_start, run_end in all_negative_runs:

        number_of_points = (
            run_end - run_start + 1
        )

        if number_of_points >= negative_run_points:

            valid_negative_runs.append(
                (
                    run_start,
                    run_end,
                )
            )

    cycles = []

    search_from_index = 0

    for positive_start, positive_end in valid_positive_runs:

        if positive_start < search_from_index:
            continue

        negative_runs_after_start = []

        for negative_start, negative_end in valid_negative_runs:

            if negative_start > positive_start:

                negative_runs_after_start.append(
                    (
                        negative_start,
                        negative_end,
                    )
                )

        if len(negative_runs_after_start) < 2:
            continue

        first_negative_run = (
            negative_runs_after_start[0]
        )

        second_negative_run = (
            negative_runs_after_start[1]
        )

        cycle_end = second_negative_run[0]

        number_of_cycle_points = (
            cycle_end - positive_start + 1
        )

        if number_of_cycle_points < minimum_cycle_points:
            continue

        cycle_number = len(cycles) + 1

        cycles.append(
            {
                "Cycle": cycle_number,
                "Start_index": int(
                    positive_start
                ),
                "End_index": int(
                    cycle_end
                ),
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

    return cycles, length_change


# ============================================================
# FORCE SMOOTHING
# ============================================================

def smooth_force(
    force_values,
    moving_window,
    savgol_window,
    savgol_order,
    gaussian_sigma,
    butterworth_order,
    butterworth_cutoff,
):

    force_values = np.asarray(
        force_values,
        dtype=float,
    )

    number_of_points = len(
        force_values
    )

    smoothed_signals = {
        "Raw force": force_values.copy()
    }

    # --------------------------------------------------------
    # Moving average
    # --------------------------------------------------------

    valid_moving_window = odd_window(
        moving_window,
        number_of_points,
    )

    if valid_moving_window is not None:

        moving_average_values = (
            pd.Series(force_values)
            .rolling(
                window=valid_moving_window,
                center=True,
                min_periods=1,
            )
            .mean()
            .to_numpy()
        )

        moving_average_name = (
            "Moving average "
            + str(valid_moving_window)
        )

        smoothed_signals[
            moving_average_name
        ] = moving_average_values

    # --------------------------------------------------------
    # Savitzky-Golay filter
    # --------------------------------------------------------

    valid_savgol_window = odd_window(
        savgol_window,
        number_of_points,
    )

    if valid_savgol_window is not None:

        valid_savgol_order = min(
            int(savgol_order),
            valid_savgol_window - 1,
        )

        savgol_values = savgol_filter(
            force_values,
            window_length=valid_savgol_window,
            polyorder=valid_savgol_order,
            mode="interp",
        )

        savgol_name = (
            "Savitzky-Golay "
            + str(valid_savgol_window)
            + " order "
            + str(valid_savgol_order)
        )

        smoothed_signals[
            savgol_name
        ] = savgol_values

    # --------------------------------------------------------
    # Gaussian filter
    # --------------------------------------------------------

    gaussian_values = gaussian_filter1d(
        force_values,
        sigma=float(gaussian_sigma),
        mode="nearest",
    )

    gaussian_name = (
        "Gaussian sigma "
        + str(gaussian_sigma)
    )

    smoothed_signals[
        gaussian_name
    ] = gaussian_values

    # --------------------------------------------------------
    # Butterworth low-pass filter
    # --------------------------------------------------------

    try:

        filter_b, filter_a = butter(
            N=int(butterworth_order),
            Wn=float(butterworth_cutoff),
            btype="low",
        )

        required_padding = (
            3
            * (
                max(
                    len(filter_a),
                    len(filter_b),
                )
                - 1
            )
        )

        if number_of_points > required_padding:

            butterworth_values = filtfilt(
                filter_b,
                filter_a,
                force_values,
            )

            butterworth_name = (
                "Butterworth order "
                + str(butterworth_order)
                + " cutoff "
                + str(butterworth_cutoff)
            )

            smoothed_signals[
                butterworth_name
            ] = butterworth_values

    except ValueError:

        pass

    return smoothed_signals


# ============================================================
# ZIP FILE CREATION
# ============================================================

def create_zip(dataframes):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        for filename, dataframe in dataframes.items():

            csv_text = dataframe.to_csv(
                index=False
            )

            archive.writestr(
                filename,
                csv_text,
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header(
    "Analysis settings"
)

st.sidebar.subheader(
    "Length normalisation"
)

baseline_points = st.sidebar.number_input(
    "Final points used for length baseline",
    min_value=10,
    max_value=1000000,
    value=1000,
    step=10,
    help=(
        "The average of the final selected length "
        "points is subtracted from every length value."
    ),
)

st.sidebar.subheader(
    "Cycle detection"
)

sustained_points = st.sidebar.number_input(
    "Points required for sustained positive increase",
    min_value=2,
    max_value=100000,
    value=10,
    step=1,
)

positive_threshold = st.sidebar.number_input(
    "Minimum positive length change per sample",
    min_value=0.0,
    value=0.0,
    step=0.000001,
    format="%.8f",
)

negative_threshold = st.sidebar.number_input(
    "Negative-length threshold",
    value=0.0,
    step=0.000001,
    format="%.8f",
)

negative_run_points = st.sidebar.number_input(
    "Points required for a negative run",
    min_value=1,
    max_value=100000,
    value=3,
    step=1,
)

minimum_cycle_points = st.sidebar.number_input(
    "Minimum points in a cycle",
    min_value=3,
    max_value=1000000,
    value=50,
    step=1,
)

st.sidebar.subheader(
    "Force smoothing"
)

moving_window = st.sidebar.number_input(
    "Moving-average window",
    min_value=3,
    max_value=100001,
    value=21,
    step=2,
)

savgol_window = st.sidebar.number_input(
    "Savitzky-Golay window",
    min_value=3,
    max_value=100001,
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
    "Butterworth cutoff as proportion of Nyquist",
    min_value=0.001,
    max_value=0.999,
    value=0.050,
    step=0.001,
    help=(
        "A lower value produces stronger smoothing."
    ),
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Drag and drop a .dat, .csv, or .txt file",
    type=[
        "dat",
        "csv",
        "txt",
    ],
    accept_multiple_files=False,
)

if uploaded_file is None:

    st.info(
        "Upload a file to begin."
    )

    st.stop()


# ============================================================
# IMPORT FILE
# ============================================================

try:

    data = read_data(
        uploaded_file
    )

except Exception as error:

    st.error(
        str(error)
    )

    st.stop()


# ============================================================
# NORMALISE LENGTH
# ============================================================

number_of_baseline_points = min(
    int(baseline_points),
    len(data),
)

length_baseline = float(
    data["Length_raw"]
    .tail(number_of_baseline_points)
    .mean()
)

data["Length_normalised"] = (
    data["Length_raw"]
    - length_baseline
)


# ============================================================
# DETECT CYCLES
# ============================================================

cycles, length_change = detect_cycles(
    length_values=data[
        "Length_normalised"
    ].to_numpy(),
    sustained_points=int(
        sustained_points
    ),
    positive_threshold=float(
        positive_threshold
    ),
    negative_threshold=float(
        negative_threshold
    ),
    negative_run_points=int(
        negative_run_points
    ),
    minimum_cycle_points=int(
        minimum_cycle_points
    ),
)

data[
    "Length_change_per_sample"
] = length_change

data["Cycle"] = pd.Series(
    pd.NA,
    index=data.index,
    dtype="Int64",
)

data[
    "Point_within_cycle"
] = pd.Series(
    pd.NA,
    index=data.index,
    dtype="Int64",
)

for cycle in cycles:

    cycle_start = cycle[
        "Start_index"
    ]

    cycle_end = cycle[
        "End_index"
    ]

    data.loc[
        cycle_start:cycle_end,
        "Cycle",
    ] = cycle["Cycle"]

    data.loc[
        cycle_start:cycle_end,
        "Point_within_cycle",
    ] = np.arange(
        1,
        cycle_end - cycle_start + 2,
    )


# ============================================================
# SMOOTH FORCE
# ============================================================

smoothed_signals = smooth_force(
    force_values=data[
        "Force_raw"
    ].to_numpy(),
    moving_window=int(
        moving_window
    ),
    savgol_window=int(
        savgol_window
    ),
    savgol_order=int(
        savgol_order
    ),
    gaussian_sigma=float(
        gaussian_sigma
    ),
    butterworth_order=int(
        butterworth_order
    ),
    butterworth_cutoff=float(
        butterworth_cutoff
    ),
)

signal_columns = {
    "Raw force": "Force_raw"
}

for signal_name, signal_values in smoothed_signals.items():

    if signal_name == "Raw force":
        continue

    column_name = (
        "Force_"
        + safe_name(signal_name)
    )

    data[column_name] = signal_values

    signal_columns[
        signal_name
    ] = column_name


# ============================================================
# SUMMARY
# ============================================================

summary_column_1, summary_column_2, summary_column_3 = (
    st.columns(3)
)

summary_column_1.metric(
    "Valid data points",
    "{:,}".format(len(data)),
)

summary_column_2.metric(
    "Length baseline",
    "{:.6g}".format(
        length_baseline
    ),
)

summary_column_3.metric(
    "Cycles detected",
    len(cycles),
)

st.caption(
    "The length baseline was calculated from the final "
    + "{:,}".format(number_of_baseline_points)
    + " valid length points."
)

with st.expander(
    "Preview processed data"
):

    st.dataframe(
        data.head(500),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# LENGTH DIAGNOSTIC GRAPH
# ============================================================

length_figure = go.Figure()

length_figure.add_trace(
    go.Scatter(
        x=data["Sample"],
        y=data[
            "Length_normalised"
        ],
        mode="lines",
        name="Normalised length",
        line=dict(
            color="black",
            width=1.5,
        ),
    )
)

length_figure.add_hline(
    y=0,
    line_dash="dash",
    line_color="grey",
)

length_figure.add_hline(
    y=float(
        negative_threshold
    ),
    line_dash="dot",
    line_color="red",
)

for cycle in cycles:

    length_figure.add_vrect(
        x0=cycle[
            "Start_index"
        ],
        x1=cycle[
            "End_index"
        ],
        opacity=0.15,
        line_width=0,
        annotation_text=(
            "Cycle "
            + str(cycle["Cycle"])
        ),
    )

length_figure.update_layout(
    title=(
        "Normalised length and detected cycles"
    ),
    xaxis_title="Sample",
    yaxis_title="Normalised length",
    template="plotly_white",
    height=500,
)

st.plotly_chart(
    length_figure,
    use_container_width=True,
)


# ============================================================
# HANDLE NO DETECTED CYCLES
# ============================================================

if len(cycles) == 0:

    st.warning(
        "No complete cycles were detected. "
        "Adjust the cycle-detection settings "
        "in the sidebar."
    )

    st.write(
        "Try reducing the sustained positive increase "
        "points, reducing the minimum cycle points, or "
        "adjusting the negative-length threshold."
    )

    st.download_button(
        label=(
            "Download processed data without cycles"
        ),
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "processed_data_no_cycles.csv"
        ),
        mime="text/csv",
    )

    st.stop()


# ============================================================
# CYCLE SUMMARY
# ============================================================

cycle_summary = pd.DataFrame(
    cycles
)

st.subheader(
    "Detected cycle boundaries"
)

st.dataframe(
    cycle_summary,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# PROCESS INDIVIDUAL CYCLES
# ============================================================

wide_cycle_frames = []

long_cycle_frames = []

metric_rows = []

zip_files = {
    "complete_processed_data.csv": data,
    "cycle_boundaries.csv": cycle_summary,
}


for cycle in cycles:

    cycle_number = cycle[
        "Cycle"
    ]

    cycle_start = cycle[
        "Start_index"
    ]

    cycle_end = cycle[
        "End_index"
    ]

    cycle_data = data.loc[
        cycle_start:cycle_end
    ].copy()

    cycle_data[
        "Point_within_cycle"
    ] = np.arange(
        1,
        len(cycle_data) + 1,
    )

    wide_cycle_frames.append(
        cycle_data
    )

    zip_files[
        "cycle_"
        + str(cycle_number)
        + "_wide.csv"
    ] = cycle_data

    long_format_parts = []

    for signal_name, signal_column in signal_columns.items():

        long_format_part = pd.DataFrame(
            {
                "Cycle": cycle_number,
                "Point_within_cycle": (
                    cycle_data[
                        "Point_within_cycle"
                    ].to_numpy()
                ),
                "Original_sample": (
                    cycle_data[
                        "Sample"
                    ].to_numpy()
                ),
                "Length_raw": (
                    cycle_data[
                        "Length_raw"
                    ].to_numpy()
                ),
                "Length_normalised": (
                    cycle_data[
                        "Length_normalised"
                    ].to_numpy()
                ),
                "Force_processing": (
                    signal_name
                ),
                "Force": (
                    cycle_data[
                        signal_column
                    ].to_numpy()
                ),
            }
        )

        long_format_parts.append(
            long_format_part
        )

        length_for_integral = (
            cycle_data[
                "Length_normalised"
            ].to_numpy()
        )

        force_for_integral = (
            cycle_data[
                signal_column
            ].to_numpy()
        )

        signed_integral = float(
            trapezoid(
                y=force_for_integral,
                x=length_for_integral,
            )
        )

        metric_rows.append(
            {
                "Cycle": cycle_number,
                "Force_processing": signal_name,
                "Signed_force_length_integral": (
                    signed_integral
                ),
                "Absolute_force_length_integral": (
                    abs(signed_integral)
                ),
                "Minimum_force": float(
                    np.min(force_for_integral)
                ),
                "Maximum_force": float(
                    np.max(force_for_integral)
                ),
                "Mean_force": float(
                    np.mean(force_for_integral)
                ),
                "Minimum_length": float(
                    np.min(length_for_integral)
                ),
                "Maximum_length": float(
                    np.max(length_for_integral)
                ),
                "Length_range": float(
                    np.ptp(length_for_integral)
                ),
                "Number_of_points": int(
                    len(cycle_data)
                ),
            }
        )

    cycle_long_data = pd.concat(
        long_format_parts,
        ignore_index=True,
    )

    long_cycle_frames.append(
        cycle_long_data
    )

    zip_files[
        "cycle_"
        + str(cycle_number)
        + "_long.csv"
    ] = cycle_long_data

    # --------------------------------------------------------
    # Force versus sample graph
    # --------------------------------------------------------

    force_figure = go.Figure()

    # --------------------------------------------------------
    # Force versus length work-loop graph
    # --------------------------------------------------------

    work_loop_figure = go.Figure()

    for signal_name, signal_column in signal_columns.items():

        if signal_name == "Raw force":
            line_width = 1
        else:
            line_width = 2

        force_figure.add_trace(
            go.Scatter(
                x=cycle_data[
                    "Sample"
                ],
                y=cycle_data[
                    signal_column
                ],
                mode="lines",
                name=signal_name,
                line=dict(
                    width=line_width
                ),
            )
        )

        work_loop_figure.add_trace(
            go.Scatter(
                x=cycle_data[
                    "Length_normalised"
                ],
                y=cycle_data[
                    signal_column
                ],
                mode="lines",
                name=signal_name,
                line=dict(
                    width=line_width
                ),
                customdata=cycle_data[
                    "Sample"
                ],
                hovertemplate=(
                    "Length: %{x:.6g}<br>"
                    "Force: %{y:.6g}<br>"
                    "Sample: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

    force_figure.update_layout(
        title=(
            "Cycle "
            + str(cycle_number)
            + ": force smoothing comparison"
        ),
        xaxis_title="Sample",
        yaxis_title="Force",
        template="plotly_white",
        height=500,
    )

    work_loop_figure.update_layout(
        title=(
            "Cycle "
            + str(cycle_number)
            + ": force-length work loop"
        ),
        xaxis_title=(
            "Normalised length"
        ),
        yaxis_title="Force",
        template="plotly_white",
        height=550,
    )

    st.subheader(
        "Cycle "
        + str(cycle_number)
    )

    information_column_1, information_column_2, information_column_3 = (
        st.columns(3)
    )

    information_column_1.metric(
        "Start sample",
        int(
            cycle_data[
                "Sample"
            ].iloc[0]
        ),
    )

    information_column_2.metric(
        "End sample",
        int(
            cycle_data[
                "Sample"
            ].iloc[-1]
        ),
    )

    information_column_3.metric(
        "Number of points",
        len(cycle_data),
    )

    force_tab, work_loop_tab, export_tab = st.tabs(
        [
            "Force smoothing",
            "Work loop",
            "Export data",
        ]
    )

    with force_tab:

        st.plotly_chart(
            force_figure,
            use_container_width=True,
            key=(
                "force_graph_"
                + str(cycle_number)
            ),
        )

    with work_loop_tab:

        st.plotly_chart(
            work_loop_figure,
            use_container_width=True,
            key=(
                "work_loop_graph_"
                + str(cycle_number)
            ),
        )

    with export_tab:

        st.write(
            "Wide format contains one row per sample "
            "and one force column for each smoothing method."
        )

        st.download_button(
            label=(
                "Download Cycle "
                + str(cycle_number)
                + " wide-format CSV"
            ),
            data=cycle_data.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                "cycle_"
                + str(cycle_number)
                + "_wide.csv"
            ),
            mime="text/csv",
            key=(
                "wide_download_"
                + str(cycle_number)
            ),
        )

        st.write(
            "Long format contains separate rows for "
            "each force-processing method."
        )

        st.download_button(
            label=(
                "Download Cycle "
                + str(cycle_number)
                + " long-format CSV"
            ),
            data=cycle_long_data.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                "cycle_"
                + str(cycle_number)
                + "_long.csv"
            ),
            mime="text/csv",
            key=(
                "long_download_"
                + str(cycle_number)
            ),
        )


# ============================================================
# COMBINE CYCLE DATA
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
    metric_rows
)

zip_files[
    "all_cycles_wide.csv"
] = all_cycles_wide

zip_files[
    "all_cycles_long.csv"
] = all_cycles_long

zip_files[
    "work_loop_metrics.csv"
] = metrics_data


# ============================================================
# EXPORT ALL RESULTS
# ============================================================

st.header(
    "Export all results"
)

st.write(
    "The exported files contain the original force, "
    "original length, normalised length, detected cycle, "
    "point within each cycle, and every smoothed force signal."
)

export_column_1, export_column_2 = (
    st.columns(2)
)

with export_column_1:

    st.download_button(
        label=(
            "Download complete processed recording"
        ),
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "complete_processed_data.csv"
        ),
        mime="text/csv",
    )

    st.download_button(
        label=(
            "Download all cycles in wide format"
        ),
        data=all_cycles_wide.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "all_cycles_wide.csv"
        ),
        mime="text/csv",
    )

    st.download_button(
        label=(
            "Download cycle boundaries"
        ),
        data=cycle_summary.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "cycle_boundaries.csv"
        ),
        mime="text/csv",
    )

with export_column_2:

    st.download_button(
        label=(
            "Download all cycles in long format"
        ),
        data=all_cycles_long.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "all_cycles_long.csv"
        ),
        mime="text/csv",
    )

    st.download_button(
        label=(
            "Download work-loop metrics"
        ),
        data=metrics_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "work_loop_metrics.csv"
        ),
        mime="text/csv",
    )

    st.download_button(
        label=(
            "Download all results as a ZIP file"
        ),
        data=create_zip(
            zip_files
        ),
        file_name=(
            "work_loop_results.zip"
        ),
        mime="application/zip",
    )


# ============================================================
# METRICS
# ============================================================

st.subheader(
    "Work-loop metrics"
)

st.dataframe(
    metrics_data,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "The force-length integral is calculated using "
    "the trapezoidal rule. Its units are the force "
    "unit multiplied by the length unit. The sign "
    "depends on the direction of the loop and the "
    "force and length sign conventions."
)
