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
    page_title="Work Loop Analyser",
    page_icon="📈",
    layout="wide",
)

st.title("Force-Length Work Loop Analyser")

st.write(
    "Upload a raw DAT file. The first two numeric values on "
    "each data row are interpreted as force in volts and length."
)


# ============================================================
# IMPORT RAW DATA
# ============================================================

def extract_numeric_data(uploaded_file):

    raw_file = uploaded_file.getvalue()

    decoded_text = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ):

        try:

            decoded_text = raw_file.decode(
                encoding
            )

            break

        except UnicodeDecodeError:

            pass

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

    source_lines = decoded_text.splitlines()

    for line_number, source_line in enumerate(
        source_lines,
        start=1,
    ):

        numeric_values = number_pattern.findall(
            source_line.strip()
        )

        if len(numeric_values) >= 2:

            try:

                force_voltage = float(
                    numeric_values[0]
                )

                length_value = float(
                    numeric_values[1]
                )

                extracted_rows.append(
                    (
                        force_voltage,
                        length_value,
                        line_number,
                    )
                )

            except ValueError:

                skipped_lines += 1

        elif source_line.strip():

            skipped_lines += 1

    if len(extracted_rows) < 3:

        raise ValueError(
            "Fewer than three rows containing at least "
            "two numeric values were found."
        )

    extracted_data = pd.DataFrame(
        extracted_rows,
        columns=[
            "Force_V",
            "Length_raw",
            "Source_line",
        ],
    )

    extracted_data.insert(
        0,
        "Sample",
        np.arange(
            len(extracted_data),
            dtype=int,
        ),
    )

    return extracted_data, skipped_lines


# ============================================================
# VALIDATE FILTER WINDOW
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
        largest_window = number_of_points
    else:
        largest_window = number_of_points - 1

    window = min(
        window,
        largest_window,
    )

    if window < 3:
        return None

    return window


# ============================================================
# IDENTIFY SUSTAINED UPWARD CROSSINGS
# ============================================================

def sustained_upward_crossings(
    length_values,
    lower_level,
    upper_level,
    sustained_points,
    refractory_points,
):

    detected_crossings = []

    number_of_points = len(
        length_values
    )

    last_crossing = -int(
        refractory_points
    )

    index = 1

    while index < (
        number_of_points
        - sustained_points
    ):

        was_below_lower_level = (
            length_values[index - 1]
            <= lower_level
        )

        future_values = length_values[
            index:index + sustained_points
        ]

        remains_above_upper_level = np.all(
            future_values >= upper_level
        )

        generally_rising = (
            future_values[-1]
            > future_values[0]
        )

        if (
            was_below_lower_level
            and remains_above_upper_level
            and generally_rising
        ):

            points_since_last_crossing = (
                index - last_crossing
            )

            if (
                points_since_last_crossing
                >= refractory_points
            ):

                detected_crossings.append(
                    index
                )

                last_crossing = index

                index += refractory_points

                continue

        index += 1

    return detected_crossings


# ============================================================
# DETECT COMPLETE CYCLES
# ============================================================

def detect_cycles(
    length_values,
    lower_level,
    upper_level,
    sustained_points,
    minimum_cycle_points,
    maximum_cycle_points,
):

    refractory_points = max(
        int(minimum_cycle_points),
        int(sustained_points),
    )

    cycle_starts = sustained_upward_crossings(
        length_values=length_values,
        lower_level=lower_level,
        upper_level=upper_level,
        sustained_points=int(
            sustained_points
        ),
        refractory_points=refractory_points,
    )

    detected_cycles = []

    for crossing_index in range(
        len(cycle_starts) - 1
    ):

        cycle_start = cycle_starts[
            crossing_index
        ]

        cycle_end = cycle_starts[
            crossing_index + 1
        ]

        number_of_cycle_points = (
            cycle_end
            - cycle_start
            + 1
        )

        if (
            number_of_cycle_points
            < minimum_cycle_points
        ):

            continue

        if (
            maximum_cycle_points > 0
            and number_of_cycle_points
            > maximum_cycle_points
        ):

            continue

        cycle_number = (
            len(detected_cycles) + 1
        )

        detected_cycles.append(
            {
                "Cycle": cycle_number,
                "Start_index": int(
                    cycle_start
                ),
                "End_index": int(
                    cycle_end
                ),
                "Number_of_points": int(
                    number_of_cycle_points
                ),
            }
        )

    return detected_cycles, cycle_starts


# ============================================================
# SMOOTH FORCE DATA
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

    smoothed_signals = {
        "Raw calibrated force": (
            force_values.copy()
        )
    }

    number_of_points = len(
        force_values
    )

    # --------------------------------------------------------
    # Moving average
    # --------------------------------------------------------

    valid_moving_window = odd_window(
        moving_window,
        number_of_points,
    )

    if valid_moving_window is not None:

        smoothed_signals[
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

    # --------------------------------------------------------
    # Savitzky-Golay
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

        smoothed_signals[
            "Savitzky-Golay"
        ] = savgol_filter(
            force_values,
            window_length=valid_savgol_window,
            polyorder=valid_savgol_order,
            mode="interp",
        )

    # --------------------------------------------------------
    # Gaussian
    # --------------------------------------------------------

    smoothed_signals[
        "Gaussian"
    ] = gaussian_filter1d(
        force_values,
        sigma=float(
            gaussian_sigma
        ),
        mode="nearest",
    )

    # --------------------------------------------------------
    # Butterworth
    # --------------------------------------------------------

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

            smoothed_signals[
                "Butterworth"
            ] = filtfilt(
                filter_b,
                filter_a,
                force_values,
            )

    except ValueError:

        pass

    return smoothed_signals


# ============================================================
# SAFE COLUMN NAMES
# ============================================================

def safe_name(text):

    cleaned_name = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        text,
    )

    return cleaned_name.strip("_")


# ============================================================
# ZIP EXPORT
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
# FORCE CALIBRATION SETTINGS
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
    help=(
        "Enter the number of millinewtons represented "
        "by one volt."
    ),
)

force_offset_voltage = st.sidebar.number_input(
    "Force zero offset, volts",
    value=0.0,
    step=0.001,
    format="%.6f",
    help=(
        "This voltage is subtracted from every recorded "
        "force voltage before conversion to millinewtons."
    ),
)


# ============================================================
# LENGTH NORMALISATION SETTINGS
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
# CYCLE DETECTION SETTINGS
# ============================================================

st.sidebar.header(
    "Cycle detection"
)

st.sidebar.caption(
    "Each complete cycle runs from one sustained "
    "upward baseline crossing to the next sustained "
    "upward baseline crossing."
)

lower_crossing_level = st.sidebar.number_input(
    "Lower crossing level",
    value=-0.02,
    step=0.001,
    format="%.6f",
    help=(
        "The length signal must first be at or below "
        "this value."
    ),
)

upper_crossing_level = st.sidebar.number_input(
    "Upper crossing level",
    value=0.02,
    step=0.001,
    format="%.6f",
    help=(
        "The signal must then remain above this value "
        "to confirm an upward crossing."
    ),
)

sustained_points = st.sidebar.number_input(
    "Points sustained above upper level",
    min_value=2,
    max_value=10000,
    value=10,
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
# SMOOTHING SETTINGS
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
    "Gaussian sigma",
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
    accept_multiple_files=False,
)

if uploaded_file is None:

    st.info(
        "Upload a file to begin."
    )

    st.stop()


# ============================================================
# EXTRACT RAW DATA
# ============================================================

try:

    data, skipped_lines = extract_numeric_data(
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


# ============================================================
# CALIBRATE FORCE
# ============================================================

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

cycles, crossing_starts = detect_cycles(
    length_values=data[
        "Length_normalised"
    ].to_numpy(),
    lower_level=float(
        lower_crossing_level
    ),
    upper_level=float(
        upper_crossing_level
    ),
    sustained_points=int(
        sustained_points
    ),
    minimum_cycle_points=int(
        minimum_cycle_points
    ),
    maximum_cycle_points=int(
        maximum_cycle_points
    ),
)

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
# SMOOTH CALIBRATED FORCE
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

    data[column_name] = signal_values

    signal_columns[
        signal_name
    ] = column_name


# ============================================================
# SELECT SIGNALS DISPLAYED ON FIGURES
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

summary_column_1, summary_column_2, summary_column_3, summary_column_4 = (
    st.columns(4)
)

summary_column_1.metric(
    "Numeric rows",
    "{:,}".format(
        len(data)
    ),
)

summary_column_2.metric(
    "Skipped lines",
    "{:,}".format(
        skipped_lines
    ),
)

summary_column_3.metric(
    "Upward crossings",
    len(crossing_starts),
)

summary_column_4.metric(
    "Complete cycles",
    len(cycles),
)

st.caption(
    "Force conversion: force (mN) = "
    "(recorded voltage - zero offset) × "
    "calibration (mN/V)."
)


# ============================================================
# CHECK PROCESSED DATA
# ============================================================

with st.expander(
    "Check extracted and calibrated data"
):

    st.dataframe(
        data[
            [
                "Sample",
                "Source_line",
                "Force_V",
                "Force_mN",
                "Length_raw",
                "Length_normalised",
            ]
        ].head(500),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label=(
            "Download extracted and calibrated data"
        ),
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "complete_processed_data.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# CYCLE DETECTION FIGURE
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
    y=float(
        lower_crossing_level
    ),
    line_dash="dot",
    line_color="orange",
    annotation_text=(
        "Lower crossing level"
    ),
)

length_figure.add_hline(
    y=float(
        upper_crossing_level
    ),
    line_dash="dot",
    line_color="green",
    annotation_text=(
        "Upper crossing level"
    ),
)

for crossing_start in crossing_starts:

    length_figure.add_vline(
        x=crossing_start,
        line_dash="dash",
        line_color="blue",
        opacity=0.5,
    )

for cycle in cycles:

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
        "Normalised length and detected cycle boundaries"
    ),
    xaxis_title="Sample",
    yaxis_title="Normalised length",
    template="plotly_white",
    height=520,
)

st.plotly_chart(
    length_figure,
    use_container_width=True,
)


# ============================================================
# STOP IF NO COMPLETE CYCLES
# ============================================================

if len(cycles) == 0:

    st.warning(
        "No complete cycles were detected. The blue "
        "dashed lines show detected upward crossings. "
        "Adjust the lower and upper crossing levels, "
        "sustained points, or cycle-length limits."
    )

    st.download_button(
        label=(
            "Download processed data"
        ),
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "complete_processed_data.csv"
        ),
        mime="text/csv",
    )

    st.stop()


# ============================================================
# CYCLE BOUNDARIES
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
# PROCESS CYCLES
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
                "Force_mN": (
                    cycle_data[
                        signal_column
                    ].to_numpy()
                ),
            }
        )

        long_format_parts.append(
            long_format_part
        )

        signed_integral = float(
            trapezoid(
                y=cycle_data[
                    signal_column
                ].to_numpy(),
                x=cycle_data[
                    "Length_normalised"
                ].to_numpy(),
            )
        )

        metric_rows.append(
            {
                "Cycle": cycle_number,
                "Force_processing": (
                    signal_name
                ),
                "Signed_work_mN_length_units": (
                    signed_integral
                ),
                "Absolute_work_mN_length_units": (
                    abs(signed_integral)
                ),
                "Minimum_force_mN": float(
                    cycle_data[
                        signal_column
                    ].min()
                ),
                "Maximum_force_mN": float(
                    cycle_data[
                        signal_column
                    ].max()
                ),
                "Mean_force_mN": float(
                    cycle_data[
                        signal_column
                    ].mean()
                ),
                "Length_range": float(
                    np.ptp(
                        cycle_data[
                            "Length_normalised"
                        ].to_numpy()
                    )
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
    # CREATE SELECTED FORCE FIGURE
    # --------------------------------------------------------

    force_figure = go.Figure()

    # --------------------------------------------------------
    # CREATE WORK-LOOP FIGURE
    # --------------------------------------------------------

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
            + ": work loop"
        ),
        xaxis_title=(
            "Normalised length"
        ),
        yaxis_title="Force (mN)",
        template="plotly_white",
        height=550,
    )

    st.subheader(
        "Cycle "
        + str(cycle_number)
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
                "force_"
                + str(cycle_number)
            ),
        )

    with work_loop_tab:

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
                "Download wide-format CSV"
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
                "wide_"
                + str(cycle_number)
            ),
        )

        st.download_button(
            label=(
                "Download long-format CSV"
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
                "long_"
                + str(cycle_number)
            ),
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
# EXPORT RESULTS
# ============================================================

st.header(
    "Export all results"
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
            "Download all cycles, wide format"
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
            "Download all cycles, long format"
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
# METRICS TABLE
# ============================================================

st.subheader(
    "Work-loop metrics"
)

st.dataframe(
    metrics_data,
    use_container_width=True,
    hide_index=True,
)
