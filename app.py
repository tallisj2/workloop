import io
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import streamlit as st


st.set_page_config(
    page_title="Work Loop Analyser",
    page_icon="📈",
    layout="wide",
)

st.title("Force-Length Work Loop Analyser")

st.write(
    "Upload a raw DAT file. The first two numeric values on each "
    "data row are treated as force voltage and length."
)


# ============================================================
# FILE IMPORT
# ============================================================

def extract_data(uploaded_file):

    raw_file = uploaded_file.getvalue()
    decoded_text = None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]:

        try:

            decoded_text = raw_file.decode(
                encoding
            )

            break

        except UnicodeDecodeError:

            continue

    if decoded_text is None:

        raise ValueError(
            "The file could not be decoded as text."
        )

    number_pattern = re.compile(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?"
    )

    extracted_rows = []
    skipped_lines = 0

    for line_number, source_line in enumerate(
        decoded_text.splitlines(),
        start=1,
    ):

        stripped_line = source_line.strip()

        if stripped_line == "":
            continue

        numeric_values = number_pattern.findall(
            stripped_line
        )

        if len(numeric_values) >= 2:

            try:

                extracted_rows.append(
                    (
                        float(numeric_values[0]),
                        float(numeric_values[1]),
                        line_number,
                    )
                )

            except ValueError:

                skipped_lines += 1

        else:

            skipped_lines += 1

    if len(extracted_rows) < 3:

        raise ValueError(
            "Fewer than three rows containing two numeric "
            "values were found."
        )

    data = pd.DataFrame(
        extracted_rows,
        columns=[
            "Force_V",
            "Length_raw",
            "Source_line",
        ],
    )

    data.insert(
        0,
        "Sample",
        np.arange(
            len(data),
            dtype=int,
        ),
    )

    return data, skipped_lines


# ============================================================
# VALID FILTER WINDOW
# ============================================================

def odd_window(
    requested_window,
    number_of_points,
):

    if number_of_points < 3:
        return None

    window = max(
        3,
        int(requested_window),
    )

    if window % 2 == 0:
        window += 1

    if number_of_points % 2 == 1:
        maximum_window = number_of_points
    else:
        maximum_window = number_of_points - 1

    window = min(
        window,
        maximum_window,
    )

    if window < 3:
        return None

    return window


# ============================================================
# CYCLE DETECTION
# ============================================================

def detect_cycles(
    length_values,
    smoothing_sigma,
    positive_peak_level,
    negative_trough_level,
    boundary_level,
    minimum_peak_distance,
    minimum_cycle_points,
    maximum_cycle_points,
    end_hold_points,
):

    length_values = np.asarray(
        length_values,
        dtype=float,
    )

    detection_length = gaussian_filter1d(
        length_values,
        sigma=float(
            smoothing_sigma
        ),
        mode="nearest",
    )

    positive_peaks, positive_properties = find_peaks(
        detection_length,
        height=float(
            positive_peak_level
        ),
        distance=int(
            minimum_peak_distance
        ),
    )

    negative_troughs, negative_properties = find_peaks(
        -detection_length,
        height=abs(
            float(
                negative_trough_level
            )
        ),
        distance=int(
            minimum_peak_distance
        ),
    )

    detected_cycles = []
    previous_cycle_end = -1

    for positive_peak in positive_peaks:

        if positive_peak <= previous_cycle_end:
            continue

        later_troughs = negative_troughs[
            negative_troughs > positive_peak
        ]

        if len(later_troughs) == 0:
            continue

        negative_trough = int(
            later_troughs[0]
        )

        possible_start_points = np.where(
            detection_length[
                :positive_peak + 1
            ]
            <= boundary_level
        )[0]

        if len(possible_start_points) == 0:

            cycle_start = 0

        else:

            cycle_start = int(
                possible_start_points[-1]
            )

        if cycle_start <= previous_cycle_end:

            cycle_start = (
                previous_cycle_end + 1
            )

        cycle_end = None

        final_possible_end = (
            len(detection_length)
            - int(end_hold_points)
        )

        for index in range(
            negative_trough + 1,
            final_possible_end,
        ):

            crossed_upward = (
                detection_length[index - 1]
                < boundary_level
                and detection_length[index]
                >= boundary_level
            )

            future_values = detection_length[
                index:
                index + int(end_hold_points)
            ]

            remains_above_boundary = np.all(
                future_values >= boundary_level
            )

            if (
                crossed_upward
                and remains_above_boundary
            ):

                cycle_end = index
                break

        if cycle_end is None:
            continue

        number_of_cycle_points = (
            cycle_end
            - cycle_start
            + 1
        )

        if (
            number_of_cycle_points
            < int(minimum_cycle_points)
        ):
            continue

        if (
            int(maximum_cycle_points) > 0
            and number_of_cycle_points
            > int(maximum_cycle_points)
        ):
            continue

        detected_cycles.append(
            {
                "Cycle": len(detected_cycles) + 1,
                "Start_index": int(
                    cycle_start
                ),
                "Positive_peak_index": int(
                    positive_peak
                ),
                "Negative_trough_index": int(
                    negative_trough
                ),
                "End_index": int(
                    cycle_end
                ),
                "Number_of_points": int(
                    number_of_cycle_points
                ),
            }
        )

        previous_cycle_end = cycle_end

    return (
        detected_cycles,
        detection_length,
        positive_peaks,
        negative_troughs,
    )


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

    signals = {
        "Raw calibrated force": (
            force_values.copy()
        )
    }

    valid_moving_window = odd_window(
        moving_window,
        number_of_points,
    )

    if valid_moving_window is not None:

        signals[
            "Moving average"
        ] = (
            pd.Series(force_values)
            .rolling(
                window=valid_moving_window,
                center=True,
                min_periods=1,
            )
            .mean()
            .to_numpy()
        )

    valid_savgol_window = odd_window(
        savgol_window,
        number_of_points,
    )

    if valid_savgol_window is not None:

        valid_savgol_order = min(
            int(savgol_order),
            valid_savgol_window - 1,
        )

        signals[
            "Savitzky-Golay"
        ] = savgol_filter(
            force_values,
            window_length=(
                valid_savgol_window
            ),
            polyorder=(
                valid_savgol_order
            ),
            mode="interp",
        )

    signals[
        "Gaussian"
    ] = gaussian_filter1d(
        force_values,
        sigma=float(
            gaussian_sigma
        ),
        mode="nearest",
    )

    try:

        filter_b, filter_a = butter(
            N=int(
                butterworth_order
            ),
            Wn=float(
                butterworth_cutoff
            ),
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

            signals[
                "Butterworth"
            ] = filtfilt(
                filter_b,
                filter_a,
                force_values,
            )

    except ValueError:

        pass

    return signals


# ============================================================
# SAFE COLUMN NAME
# ============================================================

def safe_name(text):

    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        text,
    ).strip("_")


# ============================================================
# ZIP FILE
# ============================================================

def make_zip(dataframes):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        for filename, dataframe in dataframes.items():

            archive.writestr(
                filename,
                dataframe.to_csv(
                    index=False
                ),
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# SIDEBAR: FORCE CALIBRATION
# ============================================================

st.sidebar.header(
    "Force calibration"
)

force_calibration = st.sidebar.number_input(
    "Force calibration, mN per volt",
    min_value=0.000001,
    value=1.0,
    step=0.1,
    format="%.6f",
)

force_offset_voltage = st.sidebar.number_input(
    "Force zero offset, volts",
    value=0.0,
    step=0.001,
    format="%.6f",
)


# ============================================================
# SIDEBAR: LENGTH NORMALISATION
# ============================================================

st.sidebar.header(
    "Length normalisation"
)

baseline_location = st.sidebar.radio(
    "Baseline location",
    options=[
        "Final points",
        "First points",
    ],
    index=0,
)

baseline_points = st.sidebar.number_input(
    "Points used for length baseline",
    min_value=10,
    max_value=1000000,
    value=1000,
    step=10,
)


# ============================================================
# SIDEBAR: CYCLE DETECTION
# ============================================================

st.sidebar.header(
    "Cycle detection"
)

length_detection_sigma = st.sidebar.number_input(
    "Length smoothing sigma for detection",
    min_value=0.1,
    max_value=1000.0,
    value=10.0,
    step=0.5,
)

positive_peak_level = st.sidebar.number_input(
    "Minimum positive peak",
    value=0.50,
    step=0.05,
    format="%.4f",
)

negative_trough_level = st.sidebar.number_input(
    "Maximum negative trough",
    value=-0.50,
    step=0.05,
    format="%.4f",
)

cycle_boundary_level = st.sidebar.number_input(
    "Cycle start and end level",
    value=0.0,
    step=0.01,
    format="%.4f",
)

minimum_peak_distance = st.sidebar.number_input(
    "Minimum samples between peaks",
    min_value=1,
    max_value=1000000,
    value=1000,
    step=10,
)

end_hold_points = st.sidebar.number_input(
    "Points held above cycle end level",
    min_value=1,
    max_value=10000,
    value=1,
    step=1,
)

minimum_cycle_points = st.sidebar.number_input(
    "Minimum points per cycle",
    min_value=10,
    max_value=1000000,
    value=500,
    step=10,
)

maximum_cycle_points = st.sidebar.number_input(
    "Maximum points per cycle, 0 disables",
    min_value=0,
    max_value=1000000,
    value=5000,
    step=10,
)


# ============================================================
# SIDEBAR: FORCE SMOOTHING
# ============================================================

st.sidebar.header(
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
    "Gaussian force sigma",
    min_value=0.1,
    max_value=1000.0,
    value=3.0,
    step=0.1,
)

butterworth_order = st.sidebar.number_input(
    "Butterworth order",
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
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Drag and drop a DAT, TXT, or CSV file",
    type=[
        "dat",
        "txt",
        "csv",
    ],
)

if uploaded_file is None:

    st.info(
        "Upload a file to begin."
    )

    st.stop()


# ============================================================
# IMPORT AND PROCESS DATA
# ============================================================

try:

    data, skipped_lines = extract_data(
        uploaded_file
    )

except Exception as error:

    st.error(
        str(error)
    )

    st.stop()


number_of_baseline_points = min(
    int(baseline_points),
    len(data),
)

if baseline_location == "Final points":

    length_baseline = float(
        data["Length_raw"]
        .tail(
            number_of_baseline_points
        )
        .mean()
    )

else:

    length_baseline = float(
        data["Length_raw"]
        .head(
            number_of_baseline_points
        )
        .mean()
    )

data["Length_normalised"] = (
    data["Length_raw"]
    - length_baseline
)

data["Force_mN"] = (
    (
        data["Force_V"]
        - float(
            force_offset_voltage
        )
    )
    * float(
        force_calibration
    )
)


# ============================================================
# DETECT CYCLES
# ============================================================

(
    cycles,
    detection_length,
    positive_peaks,
    negative_troughs,
) = detect_cycles(
    length_values=data[
        "Length_normalised"
    ].to_numpy(),
    smoothing_sigma=float(
        length_detection_sigma
    ),
    positive_peak_level=float(
        positive_peak_level
    ),
    negative_trough_level=float(
        negative_trough_level
    ),
    boundary_level=float(
        cycle_boundary_level
    ),
    minimum_peak_distance=int(
        minimum_peak_distance
    ),
    minimum_cycle_points=int(
        minimum_cycle_points
    ),
    maximum_cycle_points=int(
        maximum_cycle_points
    ),
    end_hold_points=int(
        end_hold_points
    ),
)

data[
    "Length_detection_smoothed"
] = detection_length

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
        "Force_mN"
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

signal_columns = {}

for signal_name, signal_values in smoothed_signals.items():

    column_name = (
        "Force_mN_"
        + safe_name(
            signal_name
        )
    )

    data[column_name] = (
        signal_values
    )

    signal_columns[
        signal_name
    ] = column_name


# ============================================================
# SELECT FIGURE SIGNALS
# ============================================================

st.sidebar.header(
    "Figure display"
)

default_display_signals = [
    "Raw calibrated force"
]

if "Butterworth" in signal_columns:

    default_display_signals.append(
        "Butterworth"
    )

displayed_signals = st.sidebar.multiselect(
    "Force signals shown on figures",
    options=list(
        signal_columns.keys()
    ),
    default=default_display_signals,
)

if len(displayed_signals) == 0:

    displayed_signals = [
        "Raw calibrated force"
    ]


# ============================================================
# SUMMARY
# ============================================================

summary_1, summary_2, summary_3, summary_4 = st.columns(
    4
)

summary_1.metric(
    "Numeric rows",
    "{:,}".format(
        len(data)
    ),
)

summary_2.metric(
    "Positive peaks",
    len(
        positive_peaks
    ),
)

summary_3.metric(
    "Negative troughs",
    len(
        negative_troughs
    ),
)

summary_4.metric(
    "Complete cycles",
    len(
        cycles
    ),
)

st.caption(
    "Force (mN) = (recorded voltage - zero offset) "
    "× calibration (mN/V)."
)


# ============================================================
# DIAGNOSTIC GRAPH
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
            width=1.3,
        ),
    )
)

length_figure.add_trace(
    go.Scatter(
        x=data["Sample"],
        y=data[
            "Length_detection_smoothed"
        ],
        mode="lines",
        name="Smoothed detection length",
        line=dict(
            color="royalblue",
            width=1,
        ),
        opacity=0.50,
    )
)

length_figure.add_hline(
    y=float(
        positive_peak_level
    ),
    line_dash="dot",
    line_color="green",
    annotation_text=(
        "Positive peak threshold"
    ),
)

length_figure.add_hline(
    y=float(
        negative_trough_level
    ),
    line_dash="dot",
    line_color="orange",
    annotation_text=(
        "Negative trough threshold"
    ),
)

length_figure.add_hline(
    y=float(
        cycle_boundary_level
    ),
    line_dash="dot",
    line_color="blue",
    annotation_text=(
        "Cycle boundary level"
    ),
)

length_figure.add_trace(
    go.Scatter(
        x=positive_peaks,
        y=detection_length[
            positive_peaks
        ],
        mode="markers",
        name="Detected positive peaks",
        marker=dict(
            color="green",
            size=9,
        ),
    )
)

length_figure.add_trace(
    go.Scatter(
        x=negative_troughs,
        y=detection_length[
            negative_troughs
        ],
        mode="markers",
        name="Detected negative troughs",
        marker=dict(
            color="orange",
            size=9,
        ),
    )
)

for cycle in cycles:

    length_figure.add_vline(
        x=cycle[
            "Start_index"
        ],
        line_dash="dash",
        line_color="blue",
    )

    length_figure.add_vline(
        x=cycle[
            "End_index"
        ],
        line_dash="dash",
        line_color="purple",
    )

    length_figure.add_vrect(
        x0=cycle[
            "Start_index"
        ],
        x1=cycle[
            "End_index"
        ],
        opacity=0.10,
        line_width=0,
        annotation_text=(
            "Cycle "
            + str(cycle["Cycle"])
        ),
    )

length_figure.update_layout(
    title=(
        "Length peaks, troughs, and detected complete cycles"
    ),
    xaxis_title="Sample",
    yaxis_title="Normalised length",
    template="plotly_white",
    height=550,
)

st.plotly_chart(
    length_figure,
    use_container_width=True,
)


if len(cycles) == 0:

    st.warning(
        "No complete cycles were detected. Check whether "
        "green peak markers and orange trough markers appear."
    )

    st.download_button(
        label="Download processed data",
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="complete_processed_data.csv",
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

    wide_filename = (
        "cycle_"
        + str(cycle_number)
        + "_wide.csv"
    )

    zip_files[
        wide_filename
    ] = cycle_data

    long_parts = []

    for signal_name, signal_column in signal_columns.items():

        long_part = pd.DataFrame(
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
                "Force_V": (
                    cycle_data[
                        "Force_V"
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
                "Force_mN": (
                    cycle_data[
                        signal_column
                    ].to_numpy()
                ),
            }
        )

        long_parts.append(
            long_part
        )

        length_values = cycle_data[
            "Length_normalised"
        ].to_numpy()

        force_values = cycle_data[
            signal_column
        ].to_numpy()

        signed_work = float(
            trapezoid(
                y=force_values,
                x=length_values,
            )
        )

        metric_rows.append(
            {
                "Cycle": cycle_number,
                "Force_processing": signal_name,
                "Signed_work_mN_length_units": signed_work,
                "Absolute_work_mN_length_units": abs(
                    signed_work
                ),
                "Minimum_force_mN": float(
                    np.min(force_values)
                ),
                "Maximum_force_mN": float(
                    np.max(force_values)
                ),
                "Mean_force_mN": float(
                    np.mean(force_values)
                ),
                "Length_range": float(
                    np.ptp(length_values)
                ),
                "Number_of_points": int(
                    len(cycle_data)
                ),
            }
        )

    cycle_long_data = pd.concat(
        long_parts,
        ignore_index=True,
    )

    long_cycle_frames.append(
        cycle_long_data
    )

    long_filename = (
        "cycle_"
        + str(cycle_number)
        + "_long.csv"
    )

    zip_files[
        long_filename
    ] = cycle_long_data

    force_figure = go.Figure()
    work_loop_figure = go.Figure()

    for signal_name in displayed_signals:

        signal_column = signal_columns[
            signal_name
        ]

        if signal_name == "Raw calibrated force":
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
            )
        )

    force_figure.update_layout(
        title=(
            "Cycle "
            + str(cycle_number)
            + ": selected force signals"
        ),
        xaxis_title="Sample",
        yaxis_title="Force (mN)",
        template="plotly_white",
        height=500,
    )

    work_loop_figure.update_layout(
        title=(
            "Cycle "
            + str(cycle_number)
            + ": force-length work loop"
        ),
        xaxis_title="Normalised length",
        yaxis_title="Force (mN)",
        template="plotly_white",
        height=550,
    )

    st.subheader(
        "Cycle "
        + str(cycle_number)
    )

    force_tab, loop_tab, export_tab = st.tabs(
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
                "force_"
                + str(cycle_number)
            ),
        )

    with loop_tab:

        st.plotly_chart(
            work_loop_figure,
            use_container_width=True,
            key=(
                "loop_"
                + str(cycle_number)
            ),
        )

    with export_tab:

        st.download_button(
            label=(
                "Download Cycle "
                + str(cycle_number)
                + " wide-format CSV"
            ),
            data=cycle_data.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=wide_filename,
            mime="text/csv",
            key=(
                "wide_"
                + str(cycle_number)
            ),
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
            file_name=long_filename,
            mime="text/csv",
            key=(
                "long_"
                + str(cycle_number)
            ),
        )


# ============================================================
# COMBINE INDIVIDUAL OUTPUTS
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
# AVERAGE NORMALISED WORK LOOP
# ============================================================

st.header(
    "Average normalised work loop"
)

st.write(
    "Selected cycles are resampled to a common cycle phase. "
    "Length is normalised within each cycle to -100% to +100% "
    "of its own half-range. Force remains in mN."
)

available_cycles = [
    int(value)
    for value in cycle_summary[
        "Cycle"
    ].tolist()
]

selected_average_cycles = st.multiselect(
    "Cycles included in the average",
    options=available_cycles,
    default=available_cycles,
)

average_force_method = st.selectbox(
    "Force signal used for the average",
    options=list(
        signal_columns.keys()
    ),
    index=(
        list(
            signal_columns.keys()
        ).index("Butterworth")
        if "Butterworth" in signal_columns
        else 0
    ),
)

phase_points = st.number_input(
    "Number of common phase points",
    min_value=51,
    max_value=5001,
    value=501,
    step=50,
)

variability_display = st.selectbox(
    "Variability shown on phase graph",
    options=[
        "Standard deviation",
        "Standard error",
        "95% confidence interval",
        "None",
    ],
    index=0,
)


if len(selected_average_cycles) >= 1:

    common_phase = np.linspace(
        0.0,
        100.0,
        int(phase_points),
    )

    force_matrix = []
    normalised_length_matrix = []
    average_long_parts = []
    physical_work_rows = []

    average_force_column = signal_columns[
        average_force_method
    ]

    for cycle_number in selected_average_cycles:

        cycle_row = cycle_summary.loc[
            cycle_summary["Cycle"]
            == cycle_number
        ].iloc[0]

        cycle_start = int(
            cycle_row["Start_index"]
        )

        cycle_end = int(
            cycle_row["End_index"]
        )

        selected_cycle_data = data.loc[
            cycle_start:cycle_end
        ].copy()

        original_phase = np.linspace(
            0.0,
            100.0,
            len(selected_cycle_data),
        )

        interpolated_force = np.interp(
            common_phase,
            original_phase,
            selected_cycle_data[
                average_force_column
            ].to_numpy(),
        )

        interpolated_length = np.interp(
            common_phase,
            original_phase,
            selected_cycle_data[
                "Length_normalised"
            ].to_numpy(),
        )

        length_minimum = float(
            np.min(
                interpolated_length
            )
        )

        length_maximum = float(
            np.max(
                interpolated_length
            )
        )

        length_midpoint = (
            length_maximum
            + length_minimum
        ) / 2.0

        length_half_range = (
            length_maximum
            - length_minimum
        ) / 2.0

        if length_half_range == 0:

            st.warning(
                "Cycle "
                + str(cycle_number)
                + " has no length range and was excluded."
            )

            continue

        length_percentage = (
            100.0
            * (
                interpolated_length
                - length_midpoint
            )
            / length_half_range
        )

        force_matrix.append(
            interpolated_force
        )

        normalised_length_matrix.append(
            length_percentage
        )

        average_long_parts.append(
            pd.DataFrame(
                {
                    "Cycle": cycle_number,
                    "Cycle_phase_percent": (
                        common_phase
                    ),
                    "Length_percent_half_range": (
                        length_percentage
                    ),
                    "Force_mN": (
                        interpolated_force
                    ),
                    "Force_processing": (
                        average_force_method
                    ),
                }
            )
        )

        physical_work = float(
            trapezoid(
                y=selected_cycle_data[
                    average_force_column
                ].to_numpy(),
                x=selected_cycle_data[
                    "Length_normalised"
                ].to_numpy(),
            )
        )

        physical_work_rows.append(
            {
                "Cycle": cycle_number,
                "Force_processing": (
                    average_force_method
                ),
                "Original_length_range": float(
                    np.ptp(
                        selected_cycle_data[
                            "Length_normalised"
                        ].to_numpy()
                    )
                ),
                "Signed_work_mN_length_units": (
                    physical_work
                ),
                "Absolute_work_mN_length_units": (
                    abs(physical_work)
                ),
            }
        )


    if len(force_matrix) >= 1:

        force_array = np.vstack(
            force_matrix
        )

        normalised_length_array = np.vstack(
            normalised_length_matrix
        )

        number_of_cycles = (
            force_array.shape[0]
        )

        mean_force = np.mean(
            force_array,
            axis=0,
        )

        mean_normalised_length = np.mean(
            normalised_length_array,
            axis=0,
        )

        if number_of_cycles > 1:

            sd_force = np.std(
                force_array,
                axis=0,
                ddof=1,
            )

            sem_force = (
                sd_force
                / np.sqrt(
                    number_of_cycles
                )
            )

        else:

            sd_force = np.zeros_like(
                mean_force
            )

            sem_force = np.zeros_like(
                mean_force
            )

        ci95_force = (
            1.96
            * sem_force
        )

        average_summary = pd.DataFrame(
            {
                "Cycle_phase_percent": (
                    common_phase
                ),
                "Mean_length_percent_half_range": (
                    mean_normalised_length
                ),
                "Mean_force_mN": (
                    mean_force
                ),
                "SD_force_mN": (
                    sd_force
                ),
                "SEM_force_mN": (
                    sem_force
                ),
                "CI95_lower_force_mN": (
                    mean_force
                    - ci95_force
                ),
                "CI95_upper_force_mN": (
                    mean_force
                    + ci95_force
                ),
                "Number_of_cycles": (
                    number_of_cycles
                ),
                "Force_processing": (
                    average_force_method
                ),
            }
        )

        average_individual_data = pd.concat(
            average_long_parts,
            ignore_index=True,
        )

        physical_work_data = pd.DataFrame(
            physical_work_rows
        )


        # ----------------------------------------------------
        # AVERAGE WORK LOOP FIGURE
        # ----------------------------------------------------

        average_loop_figure = go.Figure()

        for cycle_number in selected_average_cycles:

            cycle_plot_data = (
                average_individual_data.loc[
                    average_individual_data[
                        "Cycle"
                    ]
                    == cycle_number
                ]
            )

            if len(cycle_plot_data) == 0:
                continue

            average_loop_figure.add_trace(
                go.Scatter(
                    x=cycle_plot_data[
                        "Length_percent_half_range"
                    ],
                    y=cycle_plot_data[
                        "Force_mN"
                    ],
                    mode="lines",
                    name=(
                        "Cycle "
                        + str(cycle_number)
                    ),
                    line=dict(
                        width=1
                    ),
                    opacity=0.30,
                )
            )

        average_loop_figure.add_trace(
            go.Scatter(
                x=average_summary[
                    "Mean_length_percent_half_range"
                ],
                y=average_summary[
                    "Mean_force_mN"
                ],
                mode="lines",
                name="Mean loop",
                line=dict(
                    color="black",
                    width=4,
                ),
            )
        )

        average_loop_figure.update_layout(
            title=(
                "Individual and mean "
                "amplitude-normalised work loops"
            ),
            xaxis_title=(
                "Normalised length "
                "(% of individual half-range)"
            ),
            yaxis_title="Force (mN)",
            template="plotly_white",
            height=600,
        )


        # ----------------------------------------------------
        # MEAN FORCE AGAINST PHASE
        # ----------------------------------------------------

        phase_figure = go.Figure()

        if variability_display == "Standard deviation":

            lower_band = (
                mean_force
                - sd_force
            )

            upper_band = (
                mean_force
                + sd_force
            )

            variability_name = (
                "Mean ± SD"
            )

        elif variability_display == "Standard error":

            lower_band = (
                mean_force
                - sem_force
            )

            upper_band = (
                mean_force
                + sem_force
            )

            variability_name = (
                "Mean ± SEM"
            )

        elif variability_display == "95% confidence interval":

            lower_band = (
                mean_force
                - ci95_force
            )

            upper_band = (
                mean_force
                + ci95_force
            )

            variability_name = (
                "Mean ± 95% CI"
            )

        else:

            lower_band = None
            upper_band = None
            variability_name = None

        if lower_band is not None:

            phase_figure.add_trace(
                go.Scatter(
                    x=common_phase,
                    y=upper_band,
                    mode="lines",
                    line=dict(
                        width=0
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            phase_figure.add_trace(
                go.Scatter(
                    x=common_phase,
                    y=lower_band,
                    mode="lines",
                    line=dict(
                        width=0
                    ),
                    fill="tonexty",
                    fillcolor=(
                        "rgba(31,119,180,0.20)"
                    ),
                    name=variability_name,
                    hoverinfo="skip",
                )
            )

        phase_figure.add_trace(
            go.Scatter(
                x=common_phase,
                y=mean_force,
                mode="lines",
                name="Mean force",
                line=dict(
                    color=(
                        "rgb(31,119,180)"
                    ),
                    width=3,
                ),
            )
        )

        phase_figure.update_layout(
            title=(
                "Mean force across "
                "normalised cycle phase"
            ),
            xaxis_title=(
                "Cycle phase (%)"
            ),
            yaxis_title="Force (mN)",
            template="plotly_white",
            height=550,
        )


        # ----------------------------------------------------
        # LENGTH NORMALISATION CHECK
        # ----------------------------------------------------

        length_phase_figure = go.Figure()

        for cycle_number in selected_average_cycles:

            cycle_plot_data = (
                average_individual_data.loc[
                    average_individual_data[
                        "Cycle"
                    ]
                    == cycle_number
                ]
            )

            if len(cycle_plot_data) == 0:
                continue

            length_phase_figure.add_trace(
                go.Scatter(
                    x=cycle_plot_data[
                        "Cycle_phase_percent"
                    ],
                    y=cycle_plot_data[
                        "Length_percent_half_range"
                    ],
                    mode="lines",
                    name=(
                        "Cycle "
                        + str(cycle_number)
                    ),
                    line=dict(
                        width=1
                    ),
                    opacity=0.25,
                )
            )

        length_phase_figure.add_trace(
            go.Scatter(
                x=common_phase,
                y=mean_normalised_length,
                mode="lines",
                name=(
                    "Mean normalised length"
                ),
                line=dict(
                    color="black",
                    width=3,
                ),
            )
        )

        length_phase_figure.update_layout(
            title=(
                "Normalised length "
                "across cycle phase"
            ),
            xaxis_title=(
                "Cycle phase (%)"
            ),
            yaxis_title=(
                "Normalised length "
                "(% of individual half-range)"
            ),
            template="plotly_white",
            height=500,
        )


        average_tab, phase_tab, length_tab, average_export_tab = st.tabs(
            [
                "Average work loop",
                "Mean force by phase",
                "Length normalisation check",
                "Average outputs",
            ]
        )

        with average_tab:

            st.plotly_chart(
                average_loop_figure,
                use_container_width=True,
            )

            st.caption(
                "Thin lines are individual amplitude-normalised "
                "loops. The thick black line is the phase-matched "
                "mean loop."
            )

        with phase_tab:

            st.plotly_chart(
                phase_figure,
                use_container_width=True,
            )

        with length_tab:

            st.plotly_chart(
                length_phase_figure,
                use_container_width=True,
            )

        with average_export_tab:

            st.download_button(
                label=(
                    "Download interpolated individual loops"
                ),
                data=average_individual_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "average_input_loops_long.csv"
                ),
                mime="text/csv",
            )

            st.download_button(
                label=(
                    "Download mean loop and variability"
                ),
                data=average_summary.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "average_normalised_work_loop.csv"
                ),
                mime="text/csv",
            )

            st.download_button(
                label=(
                    "Download physical work "
                    "for selected cycles"
                ),
                data=physical_work_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "selected_cycle_physical_work.csv"
                ),
                mime="text/csv",
            )

            st.dataframe(
                physical_work_data,
                use_container_width=True,
                hide_index=True,
            )

        zip_files[
            "average_input_loops_long.csv"
        ] = average_individual_data

        zip_files[
            "average_normalised_work_loop.csv"
        ] = average_summary

        zip_files[
            "selected_cycle_physical_work.csv"
        ] = physical_work_data

    else:

        st.warning(
            "No selected cycle contained "
            "a valid length range."
        )

else:

    st.info(
        "Select at least one cycle "
        "to create the average outputs."
    )


# ============================================================
# EXPORT ALL RESULTS
# ============================================================

st.header(
    "Export all results"
)

export_column_1, export_column_2 = st.columns(
    2
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
            "Download everything as ZIP"
        ),
        data=make_zip(
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
    "Physical work is calculated from calibrated force "
    "and the original non-percentage length data. The "
    "percentage-normalised length is used only to compare "
    "and average loop shape."
)
