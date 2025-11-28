# streamlit_parquet_app.py
# ---------------------------------------------------------
# Parquet Data Explorer with Incremental Multi-Run Support
# - Runtime calculation for components
# - Uses per-run msgID_list.xlsx mapping
# - Incremental runtime_summary.xlsx (add only missing runs)
# - Dynamic ordered metrics and pie chart
# - Overlay explorer per run
# ---------------------------------------------------------

import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import base64
from typing import Dict

# -------------------------
# Parquet Cache Helpers
# -------------------------
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def load_cache(name: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def save_cache(name: str, df: pd.DataFrame):
    if not df.empty:
        df.to_parquet(CACHE_DIR / f"{name}.parquet", index=False)

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(page_title="Runtime Dashboard - 25X Direct Air Capture Unit", layout="wide")

# Inject ZEF theme CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;   /* very light grey */
            color: #111111;              /* black text */
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #ff4500;
        }
        .stTextInput > div > input {
            background-color: #ffffff;
            color: #111111;
        }
        .stSidebar {
            background-color: #f0f0f0;
        }
        .css-1d391kg {
            background-color: #f0f0f0;
        }
        .stButton button {
            background-color: #ff4500;
            color: white;
            border-radius: 5px;
        }
        .stButton button:hover {
            background-color: #ff6347;
        }
    </style>
""", unsafe_allow_html=True)

# Convert logo image to base64
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo64 = img_to_base64("zef_logo.png")

# Centered logo
st.markdown(
    f"""
    <div style="text-align: center; margin-top: 05px;">
        <img src="data:image/png;base64,{logo64}" style="width:300px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Centered title
st.markdown(
    """
    <h1 style='text-align: center; margin-top: 5px;'>
        Runtime Dashboard - 25X Direct Air Capture Unit
    </h1>
    """,
   unsafe_allow_html=True
)

st.markdown("""
    <style>
    div.stButton > button {
        height: 60px;                  /* increase height */
        background-color: #DC2D28 !important;
        font-size: 50px !important;    /* force larger text */
        font-weight: bold !important;  /* make text bold */
        border-radius: 10px;           /* rounded corners */
        padding: 10px 20px;            /* extra padding for spacing */
    }
    div.stButton > button:hover {
        background-color: #E8462F !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Configuration (sidebar)
# -------------------------
default_parent = str(Path.cwd() / "Parquet_Exports")   # ✅ default now points to Parquet_Exports
with st.sidebar:

    st.header("Configuration")
    # ✅ Update label text to reflect Parquet_Exports instead of Run_*
    PARENT_RUNS_FOLDER = Path(st.text_input("Parquet exports folder", value=default_parent))
    st.markdown(f"**Using:** `{PARENT_RUNS_FOLDER}`")
    st.markdown("---")
    st.header("Plot Settings")
    title = st.text_input("Overlay Plot Title", "Data Analysis")
    xlab = st.text_input("X-axis label", "Time (s)")
    ylab = st.text_input("Y-axis label", "Data")

# -------------------------
# Parquet Cache Helpers
# -------------------------
CACHE_DIR = PARENT_RUNS_FOLDER / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def load_cache(name: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def save_cache(name: str, df: pd.DataFrame):
    if not df.empty:
        df.to_parquet(CACHE_DIR / f"{name}.parquet", index=False)

# ✅ Folder existence check remains the same
if not PARENT_RUNS_FOLDER.exists():
    st.error(f"Parquet exports folder not found: {PARENT_RUNS_FOLDER}")
    st.stop()

# ✅ runtime_summary.xlsx still lives inside Parquet_Exports

# Load Parquet cache for durations_df
durations_df = load_cache("db_file_duration")
if durations_df.empty:
    st.info("No cache found yet. Overlap filtering will be done from the second db file.")
    durations_df = pd.DataFrame(columns=["db File","Start Time Stamp (s)","End Time Stamp (s)"])


# -------------------------
# Helpers
# -------------------------

def to_seconds_from_datetime(series: pd.Series) -> pd.Series:
    """Convert a datetime64 series to seconds since epoch."""
    ts = series
    mask = ts.notna()
    out = pd.Series(np.nan, index=ts.index, dtype="float64")
    if mask.any():
        out.loc[mask] = ts.loc[mask].values.astype("datetime64[ns]").astype("int64") / 1e9
    return out


def read_parquet_safe(path: Path) -> pd.DataFrame:
    """Safely read a parquet file, return empty DataFrame on failure."""
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Failed to read parquet file {path.name}: {e}")
        return pd.DataFrame()


def read_messages_parquet(path: Path) -> pd.DataFrame:
    """
    Read a messages parquet file and normalize the timestamp column to seconds.
    """
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Expected a 'timestamp' column in {path.name}.")
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = to_seconds_from_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df

def read_messages_with_overlap(messages_file: Path) -> pd.DataFrame:
    return read_parquet_safe(messages_file)

# Collect all commands and messages files in Parquet_Exports
def find_parquet_files(parent_folder: Path, filename: str):
    # Look for filename in DATAxxxx/* and DATAxxxx/tables/*
    files = list(parent_folder.glob(f"*/{filename}"))
    files += list(parent_folder.glob(f"*/tables/{filename}"))
    return sorted(files)


def per_series_downsample_long(long_df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """
    Downsample a long-format DataFrame per signal so that the total number of points
    does not exceed max_points. Preserves relative distribution across signals.
    """
    if max_points <= 0 or len(long_df) <= max_points:
        return long_df

    out = []
    sizes = long_df.groupby("signal").size().to_dict()
    total = sum(sizes.values()) if sizes else 0

    for sig, size in sizes.items():
        quota = max(1, int(max_points * (size / total))) if total else max_points
        g = long_df[long_df["signal"] == sig]
        if len(g) <= quota:
            out.append(g)
        else:
            step = max(1, len(g) // quota)
            out.append(g.iloc[::step, :])

    return pd.concat(out, axis=0).sort_values(["signal", "timestamp"]) if out else long_df


def extract_db_file_name(path: Path) -> str:
    return path.parent.name   # Already in expected format DATAYYYY-MM-DD-HH-MM-SS


def get_non_overlapping_window(db_new_start_ts: float,
                               db_new_end_ts: float,
                               durations_df: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    Given a new file's start/end timestamps and the existing durations_df
    (from runtime_summary.xlsx), return the non-overlapping window.
    If fully overlapped, return (None, None).
    """
    # If no prior files, keep full window
    if durations_df.empty:
        return db_new_start_ts, db_new_end_ts

    # Find all overlaps
    for _, row in durations_df.iterrows():
        prev_start = row["Start Time Stamp (s)"]
        prev_end   = row["End Time Stamp (s)"]

        # Case 1: new window fully inside an old one → skip entirely
        if prev_start <= db_new_start_ts and db_new_end_ts <= prev_end:
            return None, None

        # Case 2: overlap at the beginning
        if prev_start <= db_new_start_ts <= prev_end < db_new_end_ts:
            return prev_end, db_new_end_ts

        # Case 3: overlap at the end
        if db_new_start_ts < prev_start <= db_new_end_ts <= prev_end:
            return db_new_start_ts, prev_start

        # Case 4: partial overlap in the middle (rare, but possible)
        if db_new_start_ts < prev_start and db_new_end_ts > prev_end:
            # Trim out the overlapped middle; keep only after prev_end
            return prev_end, db_new_end_ts

    # No overlap found → keep full window
    return db_new_start_ts, db_new_end_ts

def compute_db_file_duration_raw(messages_file: Path) -> pd.DataFrame:
    """
    Compute start and end timestamps for a given messages parquet file (RAW).
    Does NOT apply overlap filtering. Used to build durations_df in-memory.
    """
    db_file = extract_db_file_name(messages_file)
    df_msg = read_parquet_safe(messages_file)  # raw read, no overlap

    if df_msg.empty or "timestamp" not in df_msg.columns:
        return pd.DataFrame([{
            "db File": db_file,
            "Start Time Stamp (s)": None,
            "End Time Stamp (s)": None
        }])

    df_msg["timestamp"] = pd.to_numeric(df_msg["timestamp"], errors="coerce")
    df_msg = df_msg.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df_msg.empty:
        return pd.DataFrame([{
            "db File": db_file,
            "Start Time Stamp (s)": None,
            "End Time Stamp (s)": None
        }])

    start_ts = df_msg["timestamp"].iloc[0]
    end_ts = df_msg["timestamp"].iloc[-1]

    return pd.DataFrame([{
        "db File": db_file,
        "Start Time Stamp (s)": start_ts,
        "End Time Stamp (s)": end_ts
    }])


# -------------------------------
# Runtime Computation Function
# -------------------------------

# Define all components and their ON conditions
components = {
    # For Big Fan: both 518 and 526 > 0
    "Big Fan": ("DAC", [518, 526], [lambda v: v > 0, lambda v: v > 0]),
    "Recycle Pump 1": ("DAC", [548], [lambda v: v > 0]),
    "Recycle Pump 2": ("DAC", [549], [lambda v: v > 0]),
    "Feed Pump": ("DAC", [550], [lambda v: v > 0]),
    "Heater 1": ("DAC", [514], [lambda v: v == 255]),
    "Heater 2": ("DAC", [515], [lambda v: v == 255]),
    "Heater 3": ("DAC", [516], [lambda v: v == 255]),
    "Heater 4": ("DAC", [517], [lambda v: v == 255]),
    "Heater 5": ("DAC", [520], [lambda v: v == 255]),
    "Heater 6": ("DAC", [521], [lambda v: v == 255]),
    # FM Components
    "Fan Drying System": ("FM", [1058], [lambda v: v > 0]),
    "Condenser Stage 1": ("FM", [1026], [lambda v: v > 0]),
    "Freezer Stage 2A": ("FM", [1028], [lambda v: v > 0]),
    "Freezer Stage 2B": ("FM", [1029], [lambda v: v > 0]),
    "Compressor First Stage": ("FM", [1102], [lambda v: v > 0]),
    "Compressor Second Stage": ("FM", [1103], [lambda v: v > 0]),
    "Stack 1": ("AEC", [338], [lambda v: v > 24]),
    "Stack 2": ("AEC", [340], [lambda v: v > 24]),
    "Stack 3": ("AEC", [342], [lambda v: v > 24]),
    "Stack 4": ("AEC", [344], [lambda v: v > 24]),
    # DS components
    "Heater 1 (DS)": ("DS", [1282], [lambda v: v > 0]),
    "Heater 2 (DS)": ("DS", [1283], [lambda v: v > 0]),
    "Heater 3 (DS)": ("DS", [1284], [lambda v: v > 0]),
    "Heater 4 (DS)": ("DS", [1285], [lambda v: v > 0]),
    "Heater 5 (DS)": ("DS", [1288], [lambda v: v > 0]),
    "Heater 6 (DS)": ("DS", [1289], [lambda v: v > 0]),
    "Distillation Column": ("DS", [1285], [lambda v: v > 0]), # Same as heater 4
    # For Feed Pump (DS): both 1286 and 1294 > 0
    "Feed Pump (DS)": ("DS", [1286, 1294], [lambda v: v > 0, lambda v: v > 0]),
}
# ...existing code...

def compute_all_runtimes(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    db_file = extract_db_file_name(messages_file)
    df_msg = read_messages_with_overlap(messages_file)
    if df_msg.empty or "timestamp" not in df_msg.columns:
        return pd.DataFrame()

    db_new_start_ts = df_msg["timestamp"].min()
    db_new_end_ts   = df_msg["timestamp"].max()
    adj_start, adj_end = get_non_overlapping_window(db_new_start_ts, db_new_end_ts, durations_df)
    if adj_start is None or adj_end is None or adj_start >= adj_end:
        records = []
        for comp_name, (subsystem, arb_ids, on_conditions) in components.items():
            records.append({
                "db File": db_file,
                "Component": comp_name,
                "Subsystem": subsystem,
                "Start Time Stamp (s)": None,
                "End Time Stamp (s)": None,
                "Hours of operation (hr)": 0.0
            })
        return pd.DataFrame(records)

    df_msg = df_msg[(df_msg["timestamp"] >= adj_start) & (df_msg["timestamp"] <= adj_end)]
    if df_msg.empty or "timestamp" not in df_msg.columns or "arbitration_id" not in df_msg.columns:
        return pd.DataFrame()

    records = []
    for comp_name, (subsystem, arb_ids, on_conditions) in components.items():
        # Ensure lists for uniformity
        if not isinstance(arb_ids, list):
            arb_ids = [arb_ids]
        if not isinstance(on_conditions, list):
            on_conditions = [on_conditions]

        # Gather all arbitration id dataframes
        dfs = []
        for arb_id in arb_ids:
            df_comp = df_msg[df_msg["arbitration_id"] == arb_id].copy()
            data_col = "formatted_data" if "formatted_data" in df_comp.columns else (
                "data" if "data" in df_comp.columns else None
            )
            if data_col is None or df_comp.empty:
                break
            df_comp["timestamp"] = pd.to_numeric(df_comp["timestamp"], errors="coerce")
            df_comp[data_col] = pd.to_numeric(df_comp[data_col], errors="coerce")
            df_comp = df_comp.dropna(subset=["timestamp", data_col]).sort_values("timestamp")
            dfs.append(df_comp.set_index("timestamp")[data_col])

        if len(dfs) != len(arb_ids):
            records.append({
                "db File": db_file,
                "Component": comp_name,
                "Subsystem": subsystem,
                "Start Time Stamp (s)": None,
                "End Time Stamp (s)": None,
                "Hours of operation (hr)": 0.0
            })
            continue

        # Align all signals on timestamp (inner join)
        df_all = pd.concat(dfs, axis=1, join="outer").sort_index().ffill()
        df_all = df_all.dropna()  # Drop rows where any signal is still NaN
        df_all.columns = [f"sig_{i}" for i in range(len(arb_ids))]
        # Apply ON conditions: all must be True
        mask = np.ones(len(df_all), dtype=bool)
        for i, cond in enumerate(on_conditions):
            mask &= cond(df_all.iloc[:, i].values)

        times = df_all.index.values
        n = len(times)
        i = 0
        found = False
        while i < n:
            while i < n and not mask[i]:
                i += 1
            if i >= n:
                break
            start_ts = times[i]
            j = i + 1
            while j < n and mask[j]:
                j += 1
            end_ts = times[j] if j < n else times[n - 1]
            if end_ts > start_ts:
                hours = (end_ts - start_ts) / 3600.0
                records.append({
                    "db File": db_file,
                    "Component": comp_name,
                    "Subsystem": subsystem,
                    "Start Time Stamp (s)": start_ts,
                    "End Time Stamp (s)": end_ts,
                    "Hours of operation (hr)": hours
                })
                found = True
            i = j + 1
        if not found:
            records.append({
                "db File": db_file,
                "Component": comp_name,
                "Subsystem": subsystem,
                "Start Time Stamp (s)": None,
                "End Time Stamp (s)": None,
                "Hours of operation (hr)": 0.0
            })
    return pd.DataFrame(records)



# -------------------------------
# Range Computation Function (Data for histograms)
# -------------------------------

def compute_range_cycles(
    messages_file: Path,
    arb_id: int,
    ranges: list[tuple[str, callable]],
    label_col: str,
    durations_df: pd.DataFrame,   # <-- add durations_df as a parameter
    transform_fn: callable = None
) -> pd.DataFrame:

    records = []
    db_file = extract_db_file_name(messages_file)

    df_msg = read_messages_with_overlap(messages_file)
    if df_msg.empty or "arbitration_id" not in df_msg.columns:
        return pd.DataFrame()

    # --- NEW: trim overlap window ---
    db_new_start_ts = df_msg["timestamp"].min()
    db_new_end_ts   = df_msg["timestamp"].max()
    adj_start, adj_end = get_non_overlapping_window(db_new_start_ts, db_new_end_ts, durations_df)

    if adj_start is None or adj_end is None or adj_start >= adj_end:
        return pd.DataFrame()   # fully overlapped → skip

    df_msg = df_msg[(df_msg["timestamp"] >= adj_start) & (df_msg["timestamp"] <= adj_end)]
    if df_msg.empty:
        return pd.DataFrame()
    # --------------------------------

    data_col = "formatted_data" if "formatted_data" in df_msg.columns else (
        "data" if "data" in df_msg.columns else None
    )
    if data_col is None:
        return pd.DataFrame()

    df_comp = df_msg[df_msg["arbitration_id"] == arb_id].copy()
    if df_comp.empty:
        return pd.DataFrame()

    df_comp["timestamp"] = pd.to_numeric(df_comp["timestamp"], errors="coerce")
    df_comp[data_col] = pd.to_numeric(df_comp[data_col], errors="coerce")
    df_comp = df_comp.dropna(subset=["timestamp", data_col]).sort_values("timestamp")

    # Apply optional transform (e.g., current density)
    if transform_fn is not None:
        df_comp[data_col] = transform_fn(df_comp[data_col])

    values = df_comp[data_col].values
    times = df_comp["timestamp"].values
    n = len(values)

    for label, cond in ranges:
        mask_values = cond(values)
        i = 0
        while i < n:
            while i < n and not mask_values[i]:
                i += 1
            if i >= n:
                break
            start_idx = i
            start_ts = times[start_idx]
            j = start_idx + 1
            while j < n and mask_values[j]:
                j += 1
            end_idx = j if j < n else n - 1
            end_ts = times[end_idx]
            if end_ts > start_ts:
                hours = (end_ts - start_ts) / 3600.0
                records.append({
                    "db File": db_file,
                    label_col: label,
                    "Start Time Stamp (s)": start_ts,
                    "End Time Stamp (s)": end_ts,
                    "Hours of operation (hr)": hours
                })
            i = end_idx + 1

    return pd.DataFrame(records)



def compute_reboiler_temperature_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    reboiler_ranges = [
        ("<=100°C", lambda t: t <= 100),
        ("100°C<T<=110°C", lambda t: (t > 100) & (t <= 110)),
        ("110°C<T<=120°C", lambda t: (t > 110) & (t <= 120)),
        ("120°C<T<=130°C", lambda t: (t > 120) & (t <= 130)),
        ("130°C<T<=135°C", lambda t: (t > 130) & (t <= 135)),
        ("135°C<T<=140°C", lambda t: (t > 135) & (t <= 140)),
        ("140°C<T<=145°C", lambda t: (t > 140) & (t <= 145)),
        ("145°C<T<=150°C", lambda t: (t > 145) & (t <= 150)),
        ("150°C<T<=155°C", lambda t: (t > 150) & (t <= 155)),
        (">155°C", lambda t: t > 155),
    ]
    return compute_range_cycles(messages_file, 595, reboiler_ranges, "Temperature range", durations_df=durations_df)


def compute_sorbent_temperature_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    sorbent_ranges = [
        ("<=60°C", lambda t: t <= 60),
        ("60°C<T<=70°C", lambda t: (t > 60) & (t <= 70)),
        ("70°C<T<=80°C", lambda t: (t > 70) & (t <= 80)),
        ("80°C<T<=85°C", lambda t: (t > 80) & (t <= 85)),
        ("85°C<T<=90°C", lambda t: (t > 85) & (t <= 90)),
        ("90°C<T<=95°C", lambda t: (t > 90) & (t <= 95)),
        ("95°C<T<=100°C", lambda t: (t > 95) & (t <= 100)),
        ("100°C<T<=105°C", lambda t: (t > 100) & (t <= 105)),
        (">105°C", lambda t: t > 105),
    ]
    return compute_range_cycles(messages_file, 553, sorbent_ranges, "Temperature range", durations_df=durations_df)


# -------------------------------
# RPM Ranges for Rotating Equipment
# -------------------------------

RPM_RANGES = [
    ("<=10", lambda x: x <= 10),
    ("10<RPM<=20", lambda x: (x > 10) & (x <= 20)),
    ("20<RPM<=30", lambda x: (x > 20) & (x <= 30)),
    ("30<RPM<=40", lambda x: (x > 30) & (x <= 40)),
    ("40<RPM<=50", lambda x: (x > 40) & (x <= 50)),
    ("50<RPM<=60", lambda x: (x > 50) & (x <= 60)),
    ("60<RPM<=70", lambda x: (x > 60) & (x <= 70)),
    ("70<RPM<=80", lambda x: (x > 70) & (x <= 80)),
    ("80<RPM<=90", lambda x: (x > 80) & (x <= 90)),
    (">90", lambda x: x > 90),
]

# -------------------------------
# Pressure Ranges for Compressors
# -------------------------------
PRESSURE_RANGES = [
    ("<1 bar", lambda p: p < 1),
    ("1–10 bar", lambda p: (p >= 1) & (p <= 10)),
    ("10–20 bar", lambda p: (p > 10) & (p <= 20)),
    ("20–30 bar", lambda p: (p > 20) & (p <= 30)),
    ("30–40 bar", lambda p: (p > 30) & (p <= 40)),
    ("40–50 bar", lambda p: (p > 40) & (p <= 50)),
    (">50 bar", lambda p: p > 50),
]

# -------------------------------
# Current Ranges for AEC Stacks
# -------------------------------
CURRENT_RANGES = [
    ("<=1 A", lambda i: i <= 1),
    ("1 A < Current <= 2 A", lambda i: (i > 1) & (i <= 2)),
    ("2 A < Current <= 3 A", lambda i: (i > 2) & (i <= 3)),
    ("3 A < Current <= 4 A", lambda i: (i > 3) & (i <= 4)),
    (">4 A", lambda i: i > 4),
]

# -------------------------------
# Current Density Ranges for AEC Stacks
# -------------------------------
CURRENT_DENSITY_RANGES = [
    ("<=0.014 A/cm²", lambda j: j <= 0.014),
    ("0.014 A/cm² < Current Density <= 0.027 A/cm²", lambda j: (j > 0.014) & (j <= 0.027)),
    ("0.027 A/cm² < Current Density <= 0.041 A/cm²", lambda j: (j > 0.027) & (j <= 0.041)),
    ("0.041 A/cm² < Current Density <= 0.054 A/cm²", lambda j: (j > 0.041) & (j <= 0.054)),
    (">0.054 A/cm²", lambda j: j > 0.054),
]

# -------------------------------
# Temperature Ranges for Hydrogen and Oxygen
# -------------------------------
TEMPERATURE_RANGES = [
        ("<=10°C", lambda t: t <= 10),
        ("10°C<T<=20°C", lambda t: (t > 10) & (t <= 20)),
        ("20°C<T<=30°C", lambda t: (t > 20) & (t <= 30)),
        ("30°C<T<=40°C", lambda t: (t > 30) & (t <= 40)),
        ("40°C<T<=50°C", lambda t: (t > 40) & (t <= 50)),
        ("50°C<T<=60°C", lambda t: (t > 50) & (t <= 60)),
        (">60°C", lambda t: t > 60),
    ]


def compute_recycle_pump_1_rpm_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    return compute_range_cycles(messages_file, 548, RPM_RANGES, "RPM Category", durations_df=durations_df)

def compute_recycle_pump_2_rpm_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    return compute_range_cycles(messages_file, 549, RPM_RANGES, "RPM Category", durations_df=durations_df)

def compute_compressor_first_stage_pressure_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize first stage compressor delivery pressure (arb_id=1106)."""
    return compute_range_cycles(messages_file, 1106, PRESSURE_RANGES, "Pressure Range", durations_df=durations_df)

def compute_compressor_second_stage_pressure_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize second stage compressor delivery pressure (arb_id=1108)."""
    return compute_range_cycles(messages_file, 1108, PRESSURE_RANGES, "Pressure Range", durations_df=durations_df)

def compute_compressor_first_stage_temperature_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize first stage compressor delivery temperature (arb_id=1083)."""
    return compute_range_cycles(messages_file, 1083, TEMPERATURE_RANGES, "Temperature Range", durations_df=durations_df)

def compute_compressor_second_stage_temperature_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize second stage compressor delivery temperature (arb_id=1081)."""
    return compute_range_cycles(messages_file, 1081, TEMPERATURE_RANGES, "Temperature Range", durations_df=durations_df)

def compute_aec_stack_1_current_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize AEC stack current (arb_id=346)."""
    return compute_range_cycles(messages_file, 346, CURRENT_RANGES, "Current Range", durations_df=durations_df)

def compute_aec_stack_2_current_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize AEC stack current (arb_id=348)."""
    return compute_range_cycles(messages_file, 348, CURRENT_RANGES, "Current Range", durations_df=durations_df)

def compute_aec_stack_3_current_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize AEC stack current (arb_id=350)."""
    return compute_range_cycles(messages_file, 350, CURRENT_RANGES, "Current Range", durations_df=durations_df)

def compute_aec_stack_4_current_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize AEC stack current (arb_id=352)."""
    return compute_range_cycles(messages_file, 352, CURRENT_RANGES, "Current Range", durations_df=durations_df)

def compute_aec_stack_1_current_density_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    return compute_range_cycles(
        messages_file,
        arb_id=346,
        ranges=CURRENT_DENSITY_RANGES,
        label_col="Current Density Range",
        durations_df=durations_df,
        transform_fn=lambda v: v / 74.0 # taking 74 cm² electrode area
    )

def compute_aec_stack_2_current_density_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    return compute_range_cycles(
        messages_file,
        arb_id=348,
        ranges=CURRENT_DENSITY_RANGES,
        label_col="Current Density Range",
        durations_df=durations_df,
        transform_fn=lambda v: v / 74.0 # taking 74 cm² electrode area
    )

def compute_aec_stack_3_current_density_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    return compute_range_cycles(
        messages_file,
        arb_id=350,
        ranges=CURRENT_DENSITY_RANGES,
        label_col="Current Density Range",
        durations_df=durations_df,
        transform_fn=lambda v: v / 74.0 # taking 74 cm² electrode area
    )

def compute_aec_stack_4_current_density_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    return compute_range_cycles(
        messages_file,
        arb_id=352,
        ranges=CURRENT_DENSITY_RANGES,
        label_col="Current Density Range",
        durations_df=durations_df,
        transform_fn=lambda v: v / 74.0 # taking 74 cm² electrode area
    )

def compute_aec_oxygen_temperature_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize oxygen flash temperature (arb_id=326)."""
    return compute_range_cycles(messages_file, 326, TEMPERATURE_RANGES, "Temperature Range", durations_df=durations_df)  

def compute_aec_hydrogen_temperature_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize hydrogen flash temperature (arb_id=334)."""
    return compute_range_cycles(messages_file, 334, TEMPERATURE_RANGES, "Temperature Range", durations_df=durations_df)

def compute_aec_oxygen_pressure_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize oxygen pressure (arb_id=358)."""
    return compute_range_cycles(messages_file, 358, PRESSURE_RANGES, "Pressure Range", durations_df=durations_df)


# -------------------------------
# Histogram Plotting Function
# -------------------------------

def plot_categorical_histogram(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    custom_order: list,
    title: str,
    mean_value: float = None,
    mean_label: str = None,
    filename: str = "histogram.csv",
    axis_title_style: dict = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    bargap: float = 0.2
):
    """
    Generic categorical histogram plotter.
    - df: input DataFrame
    - category_col: column with categories (e.g. 'Temperature range', 'RPM Category')
    - value_col: column with numeric values (e.g. 'Hours of operation (hr)')
    - custom_order: desired order of categories
    - title: chart title
    - mean_value: optional mean to annotate
    - mean_label: label for mean annotation (e.g. 'Average: {val:.2f} °C')
    - filename: CSV export filename
    - axis_title_style: dict for font styling
    - xaxis_title, yaxis_title: axis labels
    - bargap: spacing between bars
    """
    if df.empty:
        st.info(f"No data available for {title}.")
        return

    st.markdown(f"## {title}")

    # Default axis style if none provided
    if axis_title_style is None:
        axis_title_style = dict(
            family="Segoe UI, Roboto, sans-serif",
            size=14,
            color="#2F2B2C"
        )

    # Aggregate
    summary = (
        df.groupby(category_col, as_index=False)[value_col].sum()
        .set_index(category_col)
        .reindex(custom_order, fill_value=0)
        .reset_index()
    )

    # Enforce categorical order
    summary[category_col] = pd.Categorical(summary[category_col], categories=custom_order, ordered=True)

    # Plot
    fig = px.bar(summary, x=category_col, y=value_col, text=value_col)

    if mean_value is not None and mean_label is not None:
        fig.add_annotation(
            x=0.5, y=1.05, xref="paper", yref="paper",
            text=mean_label.format(val=mean_value),
            showarrow=False, font=axis_title_style, align="center"
        )

    fig.update_traces(texttemplate="%{text:.2f} hr", textposition="outside")
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=60, b=60),
        bargap=bargap,
        xaxis_title=xaxis_title or category_col,
        yaxis_title=yaxis_title or value_col,
        xaxis_title_font=axis_title_style,
        yaxis_title_font=axis_title_style
    )

    st.plotly_chart(fig, use_container_width=True)

    # Export
    st.download_button(
        f"Download {title} Data (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


# -------------------------
# Compute Active State Table (Active Time DAC, Active Time MS, Active Time AEC, Active Time FM)
# -------------------------

def compute_active_state_for_file(messages_file: Path, durations_df: pd.DataFrame) -> pd.DataFrame:
    """
    For a given messages.parquet file, compute:
    - Active Time DAC (hr): sum of non-overlapping durations where arb_id=595 or 561 > -20 (DAC subsystem)
    - Active Time MS (hr): sum of non-overlapping durations where arb_id=916 > 4 (MS subsystem)
    - Active Time AEC (hr): sum of non-overlapping durations where arb_id=390 > 4 (AEC subsystem)
    - Active Time FM (hr): sum of non-overlapping durations where arb_id=1106 or 1087 > 0 (FM subsystem)
    """
    db_file = extract_db_file_name(messages_file)

    df_msg = read_messages_with_overlap(messages_file)
    if df_msg.empty or "timestamp" not in df_msg.columns or "arbitration_id" not in df_msg.columns:
        return pd.DataFrame([{
            "db File": db_file,
            "Active Time DAC (hr)": 0.0,
            "Active Time MS (hr)": 0.0,
            "Active Time AEC (hr)": 0.0,
            "Active Time FM (hr)": 0.0
        }])

    import operator

    def sum_active_time_or(arb_ids, threshold, op):
        # Accepts a list of arb_ids, computes OR condition
        dfs = []
        for arb_id in arb_ids:
            df_vref = df_msg[df_msg["arbitration_id"] == arb_id].copy()
            if df_vref.empty:
                continue
            data_col = "formatted_data" if "formatted_data" in df_vref.columns else ("data" if "data" in df_vref.columns else None)
            if data_col is None:
                continue
            df_vref[data_col] = pd.to_numeric(df_vref[data_col], errors="coerce")
            df_vref["timestamp"] = pd.to_numeric(df_vref["timestamp"], errors="coerce")
            df_vref = df_vref.dropna(subset=["timestamp", data_col]).sort_values("timestamp")
            dfs.append(df_vref.set_index("timestamp")[data_col])
        if not dfs:
            return 0.0
        # Outer join and forward fill
        df_all = pd.concat(dfs, axis=1, join="outer").sort_index().ffill()
        df_all = df_all.dropna()
        # OR condition: any column meets the condition
        mask = np.zeros(len(df_all), dtype=bool)
        for i in range(len(dfs)):
            mask |= op(df_all.iloc[:, i].values, threshold)
        times = df_all.index.values
        n = len(times)
        i = 0
        total = 0.0
        while i < n:
            while i < n and not mask[i]:
                i += 1
            if i >= n:
                break
            start_ts = times[i]
            j = i + 1
            while j < n and mask[j]:
                j += 1
            end_ts = times[j] if j < n else times[n - 1]
            if end_ts > start_ts:
                adj_start, adj_end = get_non_overlapping_window(start_ts, end_ts, durations_df)
                if adj_start is not None and adj_end is not None and adj_end > adj_start:
                    total += (adj_end - adj_start) / 3600.0
            i = j + 1
        return total

    def sum_active_time(arb_id, threshold, op):
        return sum_active_time_or([arb_id], threshold, op)

    # DAC: 595 OR 561 > -20
    active_time_dac_hr = sum_active_time_or([595, 561], -20, operator.gt)
    # MS: 916 > 4
    active_time_ms_hr = sum_active_time(916, 4, operator.gt)
    # AEC: 390 > 4
    active_time_aec_hr = sum_active_time(390, 4, operator.gt)
    # FM: 1106 OR 1087 > 0
    active_time_fm_hr = sum_active_time_or([1106, 1087], 0, operator.gt)

    return pd.DataFrame([{
        "db File": db_file,
        "Active Time DAC (hr)": active_time_dac_hr,
        "Active Time MS (hr)": active_time_ms_hr,
        "Active Time AEC (hr)": active_time_aec_hr,
        "Active Time FM (hr)": active_time_fm_hr
    }])

# -------------------------
# Compute total_gui_time from db_file_duration
# -------------------------

db_file_duration_df = load_cache("db_file_duration")
if not db_file_duration_df.empty and "End Time Stamp (s)" in db_file_duration_df.columns and "Start Time Stamp (s)" in db_file_duration_df.columns:
    min_start = db_file_duration_df["Start Time Stamp (s)"].min()
    max_end = db_file_duration_df["End Time Stamp (s)"].max()
    total_gui_time = (max_end - min_start) / 3600.0 if pd.notnull(min_start) and pd.notnull(max_end) else 0.0
else:
    total_gui_time = 0.0


# -------------------------
# Computing the Big Fan PWM
# -------------------------
def compute_big_fan_pwm_for_file(messages_file: Path) -> pd.DataFrame:
    """
    Compute Big Fan PWM ON/OFF cycles from a messages parquet file (arbitration_id=526).
    Returns DataFrame with db File identifier instead of Run.
    """
    records = []
    db_file = extract_db_file_name(messages_file)  # derive db File name from filename

    df_msg = read_messages_with_overlap(messages_file)
    if df_msg.empty or "arbitration_id" not in df_msg.columns:
        return pd.DataFrame()

    data_col = "formatted_data" if "formatted_data" in df_msg.columns else ("data" if "data" in df_msg.columns else None)
    if data_col is None:
        return pd.DataFrame()

    df_pwm = df_msg[df_msg["arbitration_id"] == 526].copy()
    if df_pwm.empty:
        return pd.DataFrame()

    # Parse numeric values
    df_pwm["timestamp"] = pd.to_numeric(df_pwm["timestamp"], errors="coerce")
    df_pwm[data_col] = pd.to_numeric(df_pwm[data_col], errors="coerce")
    df_pwm = df_pwm.dropna(subset=["timestamp", data_col]).sort_values("timestamp")

    values = df_pwm[data_col].values
    times = df_pwm["timestamp"].values
    n = len(values)
    i = 0
    found = False

    while i < n:
        # Find start of ON (PWM > 0)
        while i < n and values[i] <= 0:
            i += 1
        if i >= n:
            break
        start_ts = times[i]
        j = i + 1
        while j < n and values[j] > 0:
            j += 1
        end_ts = times[j if j < n else n - 1]
        pwm_val = np.mean(values[i:j]) if j > i else values[i]
        if end_ts > start_ts:
            hours = (end_ts - start_ts) / 3600.0
            records.append({
                "db File": db_file,
                "PWM": pwm_val,
                "Start Time Stamp (s)": start_ts,
                "End Time Stamp (s)": end_ts,
                "Hours of operation (hr)": hours
            })
            found = True
        i = j + 1

    if not found:
        records.append({
            "db File": db_file,
            "PWM": 0,
            "Start Time Stamp (s)": None,
            "End Time Stamp (s)": None,
            "Hours of operation (hr)": 0.0
        })

    return pd.DataFrame(records)

# -------------------------
# Weather classification (Temperature = arb_id 561, RH = arb_id 593)
# -------------------------
def compute_weather_classification_for_file(messages_file: Path) -> pd.DataFrame:
    """
    Compute average temperature and relative humidity from a messages parquet file
    and classify weather conditions. Returns DataFrame with db File identifier.
    """
    db_file = extract_db_file_name(messages_file)  # derive db File name from filename

    df_msg = read_messages_with_overlap(messages_file)
    if df_msg.empty or "arbitration_id" not in df_msg.columns:
        return pd.DataFrame()

    data_col = "formatted_data" if "formatted_data" in df_msg.columns else ("data" if "data" in df_msg.columns else None)
    if data_col is None:
        return pd.DataFrame()

    avg_temp, avg_rh = None, None

    # Temperature (arb_id = 561)
    df_temp = df_msg[df_msg["arbitration_id"] == 561].copy()
    if not df_temp.empty:
        df_temp[data_col] = pd.to_numeric(df_temp[data_col], errors="coerce")
        avg_temp = df_temp[data_col].mean()

    # Relative Humidity (arb_id = 593)
    df_rh = df_msg[df_msg["arbitration_id"] == 593].copy()
    if not df_rh.empty:
        df_rh[data_col] = pd.to_numeric(df_rh[data_col], errors="coerce")
        avg_rh = df_rh[data_col].mean()

    if avg_temp is None or avg_rh is None:
        return pd.DataFrame()

    # Classification rules
    if avg_temp >= 24 and avg_rh >= 60:
        classification = "Hot and humid"
    elif avg_temp >= 24 and avg_rh <= 40:
        classification = "Hot and dry"
    elif avg_temp <= 18 and avg_rh >= 60:
        classification = "Cold and humid"
    elif avg_temp <= 18 and avg_rh <= 40:
        classification = "Cold and dry"
    elif avg_temp <= 18 and (avg_rh > 40 and avg_rh < 60):
        classification = "Cold and Moderate humidity"
    elif avg_temp >= 24 and (avg_rh > 40 and avg_rh < 60):
        classification = "Hot and Moderate humidity"
    elif (avg_temp > 18 and avg_temp < 24) and avg_rh <= 40:
        classification = "Dry and Moderate temperature"
    elif (avg_temp > 18 and avg_temp < 24) and avg_rh >= 60:
        classification = "Humid and Moderate temperature"
    else:
        classification = "Moderate"

    return pd.DataFrame([{
        "db File": db_file,
        "Average Temperature (°C)": avg_temp,
        "Average Relative Humidity (%)": avg_rh,
        "Weather Classification": classification
    }])


# -------------------------
# Mean Values
# -------------------------
def compute_mean(messages_file: Path) -> pd.DataFrame:
    """
    Compute average values for Reboiler Temp, Sorbent Temp, Big Fan PWM,
    Recycle Pump RPMs, Compressor pressures, AEC stack currents,
    and AEC stack current densities.
    """
    db_file = extract_db_file_name(messages_file)  # derive db File name from filename

    df_msg = read_messages_with_overlap(messages_file)
    if df_msg.empty or "arbitration_id" not in df_msg.columns:
        return pd.DataFrame()

    data_col = "formatted_data" if "formatted_data" in df_msg.columns else ("data" if "data" in df_msg.columns else None)
    if data_col is None:
        return pd.DataFrame()

    # Initialize variables (DO NOT CHANGE VARIABLE NAMES)
    avg_reboiler_temp, avg_sorbent_temp, avg_big_fan_pwm = None, None, None
    avg_rec_pump_1_rpm, avg_rec_pump_2_rpm = None, None
    avg_comp_first_stage_pressure, avg_comp_second_stage_pressure = None, None
    avg_comp_first_stage_temperature, avg_comp_second_stage_temperature = None, None
    avg_aec_stack_1_current = avg_aec_stack_2_current = None
    avg_aec_stack_3_current = avg_aec_stack_4_current = None
    avg_aec_stack_1_current_density = avg_aec_stack_2_current_density = None
    avg_aec_stack_3_current_density = avg_aec_stack_4_current_density = None
    avg_aec_oxygen_temp = None
    avg_aec_hydrogen_temp = None
    avg_aec_oxygen_pressure = None

    # Reboiler Temperature (arb_id = 595)
    df_reboiler_temp = df_msg[df_msg["arbitration_id"] == 595].copy()
    if not df_reboiler_temp.empty:
        df_reboiler_temp[data_col] = pd.to_numeric(df_reboiler_temp[data_col], errors="coerce")
        avg_reboiler_temp = df_reboiler_temp[data_col].mean()

    # Sorbent Top Temperature (arb_id = 553)
    df_sorbent_temp = df_msg[df_msg["arbitration_id"] == 553].copy()
    if not df_sorbent_temp.empty:
        df_sorbent_temp[data_col] = pd.to_numeric(df_sorbent_temp[data_col], errors="coerce")
        avg_sorbent_temp = df_sorbent_temp[data_col].mean()

    # Big Fan PWM (arb_id = 526)
    df_big_fan_pwm = df_msg[df_msg["arbitration_id"] == 526].copy()
    if not df_big_fan_pwm.empty:
        df_big_fan_pwm[data_col] = pd.to_numeric(df_big_fan_pwm[data_col], errors="coerce")
        avg_big_fan_pwm = df_big_fan_pwm[data_col].mean()

    # Recycle Pump 1 RPM (arb_id = 548)
    df_rec_pump_1_rpm = df_msg[df_msg["arbitration_id"] == 548].copy()
    if not df_rec_pump_1_rpm.empty:
        df_rec_pump_1_rpm[data_col] = pd.to_numeric(df_rec_pump_1_rpm[data_col], errors="coerce")
        avg_rec_pump_1_rpm = df_rec_pump_1_rpm[data_col].mean()

    # Recycle Pump 2 RPM (arb_id = 549)
    df_rec_pump_2_rpm = df_msg[df_msg["arbitration_id"] == 549].copy()
    if not df_rec_pump_2_rpm.empty:
        df_rec_pump_2_rpm[data_col] = pd.to_numeric(df_rec_pump_2_rpm[data_col], errors="coerce")
        avg_rec_pump_2_rpm = df_rec_pump_2_rpm[data_col].mean()

    # First stage Compressor Pressure (arb_id = 1106)
    df_comp_first_stage_pressure = df_msg[df_msg["arbitration_id"] == 1106].copy()
    if not df_comp_first_stage_pressure.empty:
        df_comp_first_stage_pressure[data_col] = pd.to_numeric(df_comp_first_stage_pressure[data_col], errors="coerce")
        avg_comp_first_stage_pressure = df_comp_first_stage_pressure[data_col].mean()

    # Second stage Compressor Pressure (arb_id = 1108)
    df_comp_second_stage_pressure = df_msg[df_msg["arbitration_id"] == 1108].copy()
    if not df_comp_second_stage_pressure.empty:
        df_comp_second_stage_pressure[data_col] = pd.to_numeric(df_comp_second_stage_pressure[data_col], errors="coerce")
        avg_comp_second_stage_pressure = df_comp_second_stage_pressure[data_col].mean()

    # First stage Compressor Temperature (arb_id = 1083)
    df_comp_first_stage_temperature = df_msg[df_msg["arbitration_id"] == 1083].copy()
    if not df_comp_first_stage_temperature.empty:
        df_comp_first_stage_temperature[data_col] = pd.to_numeric(df_comp_first_stage_temperature[data_col], errors="coerce")
        avg_comp_first_stage_temperature = df_comp_first_stage_temperature[data_col].mean()

    # Second stage Compressor Temperature (arb_id = 1081)
    df_comp_second_stage_temperature = df_msg[df_msg["arbitration_id"] == 1081].copy()
    if not df_comp_second_stage_temperature.empty:
        df_comp_second_stage_temperature[data_col] = pd.to_numeric(df_comp_second_stage_temperature[data_col], errors="coerce")
        avg_comp_second_stage_temperature = df_comp_second_stage_temperature[data_col].mean()

    # Stack 1 Current (arb_id = 346)
    df_aec_stack_1_current = df_msg[df_msg["arbitration_id"] == 346].copy()
    if not df_aec_stack_1_current.empty:
        df_aec_stack_1_current[data_col] = pd.to_numeric(df_aec_stack_1_current[data_col], errors="coerce")
        avg_aec_stack_1_current = df_aec_stack_1_current[data_col].mean()

    # Stack 2 Current (arb_id = 348)
    df_aec_stack_2_current = df_msg[df_msg["arbitration_id"] == 348].copy()
    if not df_aec_stack_2_current.empty:
        df_aec_stack_2_current[data_col] = pd.to_numeric(df_aec_stack_2_current[data_col], errors="coerce")
        avg_aec_stack_2_current = df_aec_stack_2_current[data_col].mean()

    # Stack 3 Current (arb_id = 350)
    df_aec_stack_3_current = df_msg[df_msg["arbitration_id"] == 350].copy()
    if not df_aec_stack_3_current.empty:
        df_aec_stack_3_current[data_col] = pd.to_numeric(df_aec_stack_3_current[data_col], errors="coerce")
        avg_aec_stack_3_current = df_aec_stack_3_current[data_col].mean()

    # Stack 4 Current (arb_id = 352)
    df_aec_stack_4_current = df_msg[df_msg["arbitration_id"] == 352].copy()
    if not df_aec_stack_4_current.empty:
        df_aec_stack_4_current[data_col] = pd.to_numeric(df_aec_stack_4_current[data_col], errors="coerce")
        avg_aec_stack_4_current = df_aec_stack_4_current[data_col].mean()

    # Oxygen Flash Temperature (arb_id = 326)
    df_aec_oxygen_temp = df_msg[df_msg["arbitration_id"] == 326].copy()
    if not df_aec_oxygen_temp.empty:
        df_aec_oxygen_temp[data_col] = pd.to_numeric(df_aec_oxygen_temp[data_col], errors="coerce")
        avg_aec_oxygen_temp = df_aec_oxygen_temp[data_col].mean()
    
    # Hydrogen Flash Temperature (arb_id = 334)
    df_aec_hydrogen_temp = df_msg[df_msg["arbitration_id"] == 334].copy()
    if not df_aec_hydrogen_temp.empty:
        df_aec_hydrogen_temp[data_col] = pd.to_numeric(df_aec_hydrogen_temp[data_col], errors="coerce")
        avg_aec_hydrogen_temp = df_aec_hydrogen_temp[data_col].mean()

    # Oxygen Pressure (arb_id = 358)
    df_aec_oxygen_pressure = df_msg[df_msg["arbitration_id"] == 358].copy()
    if not df_aec_oxygen_pressure.empty:
        df_aec_oxygen_pressure[data_col] = pd.to_numeric(df_aec_oxygen_pressure[data_col], errors="coerce")
        avg_aec_oxygen_pressure = df_aec_oxygen_pressure[data_col].mean()

    # -------------------------
    # Compute Current Densities (DO NOT CHANGE VARIABLE NAMES)
    # -------------------------
    SURFACE_AREA_CM2 = 74.0

    if avg_aec_stack_1_current is not None:
        avg_aec_stack_1_current_density = avg_aec_stack_1_current / SURFACE_AREA_CM2

    if avg_aec_stack_2_current is not None:
        avg_aec_stack_2_current_density = avg_aec_stack_2_current / SURFACE_AREA_CM2

    if avg_aec_stack_3_current is not None:
        avg_aec_stack_3_current_density = avg_aec_stack_3_current / SURFACE_AREA_CM2

    if avg_aec_stack_4_current is not None:
        avg_aec_stack_4_current_density = avg_aec_stack_4_current / SURFACE_AREA_CM2

    # -------------------------
    # Fill missing values with 0 instead of skipping the file
    # -------------------------
    avg_reboiler_temp = avg_reboiler_temp if avg_reboiler_temp is not None else 0
    avg_sorbent_temp = avg_sorbent_temp if avg_sorbent_temp is not None else 0
    avg_big_fan_pwm = avg_big_fan_pwm if avg_big_fan_pwm is not None else 0
    avg_rec_pump_1_rpm = avg_rec_pump_1_rpm if avg_rec_pump_1_rpm is not None else 0
    avg_rec_pump_2_rpm = avg_rec_pump_2_rpm if avg_rec_pump_2_rpm is not None else 0
    avg_comp_first_stage_pressure = avg_comp_first_stage_pressure if avg_comp_first_stage_pressure is not None else 0
    avg_comp_second_stage_pressure = avg_comp_second_stage_pressure if avg_comp_second_stage_pressure is not None else 0
    avg_comp_first_stage_temperature = avg_comp_first_stage_temperature if avg_comp_first_stage_temperature is not None else 0
    avg_comp_second_stage_temperature = avg_comp_second_stage_temperature if avg_comp_second_stage_temperature is not None else 0
    avg_aec_stack_1_current = avg_aec_stack_1_current if avg_aec_stack_1_current is not None else 0
    avg_aec_stack_2_current = avg_aec_stack_2_current if avg_aec_stack_2_current is not None else 0
    avg_aec_stack_3_current = avg_aec_stack_3_current if avg_aec_stack_3_current is not None else 0
    avg_aec_stack_4_current = avg_aec_stack_4_current if avg_aec_stack_4_current is not None else 0
    avg_aec_stack_1_current_density = avg_aec_stack_1_current_density if avg_aec_stack_1_current_density is not None else 0
    avg_aec_stack_2_current_density = avg_aec_stack_2_current_density if avg_aec_stack_2_current_density is not None else 0
    avg_aec_stack_3_current_density = avg_aec_stack_3_current_density if avg_aec_stack_3_current_density is not None else 0
    avg_aec_stack_4_current_density = avg_aec_stack_4_current_density if avg_aec_stack_4_current_density is not None else 0
    avg_aec_oxygen_temp = avg_aec_oxygen_temp if avg_aec_oxygen_temp is not None else 0
    avg_aec_hydrogen_temp = avg_aec_hydrogen_temp if avg_aec_hydrogen_temp is not None else 0
    avg_aec_oxygen_pressure = avg_aec_oxygen_pressure if avg_aec_oxygen_pressure is not None else 0


    # -------------------------
    # Final Output Row
    # -------------------------
    return pd.DataFrame([{
        "db File": db_file,
        "Average Reboiler Temperature (°C)": avg_reboiler_temp,
        "Average Sorbent Top Temperature (°C)": avg_sorbent_temp,
        "Average Big Fan PWM": avg_big_fan_pwm,
        "Average Recycle Pump 1 RPM": avg_rec_pump_1_rpm,
        "Average Recycle Pump 2 RPM": avg_rec_pump_2_rpm,
        "Average Compressor First Stage Pressure (bar)": avg_comp_first_stage_pressure,
        "Average Compressor Second Stage Pressure (bar)": avg_comp_second_stage_pressure,
        "Average Compressor First Stage Temperature (°C)": avg_comp_first_stage_temperature,
        "Average Compressor Second Stage Temperature (°C)": avg_comp_second_stage_temperature,
        "Average AEC Stack 1 Current (A)": avg_aec_stack_1_current,
        "Average AEC Stack 2 Current (A)": avg_aec_stack_2_current,
        "Average AEC Stack 3 Current (A)": avg_aec_stack_3_current,
        "Average AEC Stack 4 Current (A)": avg_aec_stack_4_current,
        "Average AEC Stack 1 Current Density (A/cm²)": avg_aec_stack_1_current_density,
        "Average AEC Stack 2 Current Density (A/cm²)": avg_aec_stack_2_current_density,
        "Average AEC Stack 3 Current Density (A/cm²)": avg_aec_stack_3_current_density,
        "Average AEC Stack 4 Current Density (A/cm²)": avg_aec_stack_4_current_density,
        "Average AEC Oxygen Flash Temperature (°C)": avg_aec_oxygen_temp,
        "Average AEC Hydrogen Flash Temperature (°C)": avg_aec_hydrogen_temp,
        "Average AEC Oxygen Pressure (bar)": avg_aec_oxygen_pressure
    }])


# -------------------------
# db file duration
# -------------------------
def compute_db_file_duration(messages_file: Path) -> pd.DataFrame:
    """
    Compute start and end timestamps for a given messages parquet file.
    Returns a DataFrame with columns: db File, Start Time Stamp, End Time Stamp.
    """
    db_file = extract_db_file_name(messages_file)
    df_msg = read_messages_with_overlap(messages_file)

    if df_msg.empty or "timestamp" not in df_msg.columns:
        return pd.DataFrame([{
            "db File": db_file,
            "Start Time Stamp (s)": None,
            "End Time Stamp (s)": None
        }])

    # Normalize timestamps to numeric
    df_msg["timestamp"] = pd.to_numeric(df_msg["timestamp"], errors="coerce")
    df_msg = df_msg.dropna(subset=["timestamp"]).sort_values("timestamp")

    if df_msg.empty:
        return pd.DataFrame([{
            "db File": db_file,
            "Start Time Stamp (s)": None,
            "End Time Stamp (s)": None
        }])

    start_ts = df_msg["timestamp"].iloc[0]
    end_ts = df_msg["timestamp"].iloc[-1]

    return pd.DataFrame([{
        "db File": db_file,
        "Start Time Stamp (s)": start_ts,
        "End Time Stamp (s)": end_ts
    }])


# -------------------------
# Mean Residence Time
# -------------------------
def compute_mean_residence(commands_file: Path, messages_file: Path) -> pd.DataFrame:
    """
    Compute average Feed Pump PWM, RPM, Flowrate, and Residence Time for a db file.
    - PWM is taken from commands parquet (arb_id = 551)
    - RPM is taken from messages parquet (arb_id = 550)
    - Flowrate = 0.1044 * PWM - 2.1112
    - Residence Time = (Stripper Volume * 1e6) / Flowrate
    Stripper Volume = 0.00337 m^3
    """
    db_file = extract_db_file_name(commands_file)  # derive db File name from filename

    df_cmd = read_parquet_safe(commands_file)
    df_msg = read_messages_with_overlap(messages_file)

    if df_cmd.empty or "arbitration_id" not in df_cmd.columns:
        return pd.DataFrame()
    if df_msg.empty or "arbitration_id" not in df_msg.columns:
        return pd.DataFrame()

    data_col_cmd = "formatted_data" if "formatted_data" in df_cmd.columns else ("data" if "data" in df_cmd.columns else None)
    data_col_msg = "formatted_data" if "formatted_data" in df_msg.columns else ("data" if "data" in df_msg.columns else None)
    if data_col_cmd is None or data_col_msg is None:
        return pd.DataFrame()

    avg_feed_pwm, avg_feed_rpm, avg_feed_flowrate, avg_residence_time = None, None, None, None

    # --- PWM from commands_file (arb_id = 551) ---
    df_feed_pwm = df_cmd[df_cmd["arbitration_id"] == 551].copy()
    if not df_feed_pwm.empty:
        df_feed_pwm[data_col_cmd] = pd.to_numeric(df_feed_pwm[data_col_cmd], errors="coerce")
        avg_feed_pwm = df_feed_pwm[data_col_cmd].mean()

    # --- RPM from messages_file (arb_id = 550) ---
    df_feed_rpm = df_msg[df_msg["arbitration_id"] == 550].copy()
    if not df_feed_rpm.empty:
        df_feed_rpm[data_col_msg] = pd.to_numeric(df_feed_rpm[data_col_msg], errors="coerce")
        avg_feed_rpm = df_feed_rpm[data_col_msg].mean()

    # --- Flowrate Calculation from PWM ---
    if avg_feed_pwm is not None and not pd.isna(avg_feed_pwm):
        avg_feed_flowrate = 0.1044 * float(avg_feed_pwm) - 2.1112

    # --- Residence Time Calculation ---
    if avg_feed_flowrate is not None and not pd.isna(avg_feed_flowrate) and avg_feed_flowrate > 0:
        stripper_volume_m3 = 0.00337
        avg_residence_time = (stripper_volume_m3 * 1e6) / avg_feed_flowrate

    # --- Return DataFrame ---
    if None in [avg_feed_pwm, avg_feed_rpm, avg_feed_flowrate, avg_residence_time]:
        return pd.DataFrame()

    return pd.DataFrame([{
        "db File": db_file,
        "Average Feed Pump PWM": avg_feed_pwm,
        "Average Feed Pump RPM": avg_feed_rpm,
        "Average Feed Pump Flowrate (ml/s)": avg_feed_flowrate,
        "Average Residence Time (s)": avg_residence_time
    }])

# -------------------------
# Sidebar filter & export (overlay)
# -------------------------
def sidebar_filter_and_export_fn(long_df: pd.DataFrame):
    required = {"timestamp", "value", "signal"}
    if not required.issubset(long_df.columns):
        st.error(f"Expect columns: {sorted(list(required))}")
        return pd.DataFrame(), None, None

    df = long_df.dropna(subset=["timestamp", "value"]).copy()
    with st.sidebar:
        st.markdown("---")
        st.subheader("Filter & Export (overlay)")
        if df.empty:
            xmin_val, xmax_val = 0.0, 1.0
            ymin_val, ymax_val = 0.0, 1.0
        else:
            xmin_val = float(df["timestamp"].min())
            xmax_val = float(df["timestamp"].max())
            ymin_val = float(pd.to_numeric(df["value"], errors="coerce").min())
            ymax_val = float(pd.to_numeric(df["value"], errors="coerce").max())
            if xmin_val == xmax_val:
                xmax_val = xmin_val + 1e-6
            if ymin_val == ymax_val:
                ymax_val = ymin_val + 1e-6
        xmin, xmax = st.slider("X-range (seconds)", min_value=xmin_val, max_value=xmax_val,
                            value=(xmin_val, xmax_val), key="xrange_slider_overlay")
        ymin, ymax = st.slider("Y-range (data)", min_value=ymin_val, max_value=ymax_val,
                            value=(ymin_val, ymax_val), key="yrange_slider_overlay")
        st.write(f"{len(df):,} total points available")

    if df.empty:
        filtered_long = df.copy()
    else:
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
        filtered_long = df[
            (df["timestamp"] >= xmin) & (df["timestamp"] <= xmax) &
            (df["value_num"] >= ymin) & (df["value_num"] <= ymax)
        ].copy()
        if "value_num" in filtered_long.columns:
            filtered_long = filtered_long.drop(columns=["value_num"])

    # Export buttons in sidebar
    with st.sidebar:
        st.markdown("### Export filtered selection")
        st.write(f"{len(filtered_long):,} points in filtered selection")

        # Optional: include db File name in export filenames
        db_file = filtered_long["db File"].iloc[0] if "db File" in filtered_long.columns and not filtered_long.empty else "DATA"

        try:
            long_csv = filtered_long.to_csv(index=False).encode("utf-8")
            st.download_button("Download filtered (long) CSV", data=long_csv,
                            file_name=f"{db_file}_filtered_long.csv", mime="text/csv")
        except Exception:
            st.info("Could not create long CSV.")

        try:
            out_long = io.BytesIO()
            with pd.ExcelWriter(out_long, engine="openpyxl") as writer:
                filtered_long.to_excel(writer, index=False, sheet_name="long")
            st.download_button("Download filtered (long) Excel", data=out_long.getvalue(),
                            file_name=f"{db_file}_filtered_long.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.info("Install `openpyxl` to enable Excel (long) export.")

        if not filtered_long.empty:
            try:
                wide = filtered_long.pivot_table(index="timestamp", columns="signal", values="value", aggfunc="first").reset_index()
                wide_csv = wide.to_csv(index=False).encode("utf-8")
                st.download_button("Download filtered (wide) CSV", data=wide_csv,
                                file_name=f"{db_file}_filtered_wide.csv", mime="text/csv")
                out_wide = io.BytesIO()
                with pd.ExcelWriter(out_wide, engine="openpyxl") as writer:
                    wide.to_excel(writer, index=False, sheet_name="wide")
                st.download_button("Download filtered (wide) Excel", data=out_wide.getvalue(),
                                file_name=f"{db_file}_filtered_wide.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:
                st.info("Could not create wide exports (install openpyxl or check data).")

    return filtered_long, (xmin, xmax), (ymin, ymax)


# -------------------------
# Aggregate across db files (incremental)
# -------------------------

# Collect all commands and messages files in Parquet_Exports
command_files = find_parquet_files(PARENT_RUNS_FOLDER, "commands.parquet")
message_files = find_parquet_files(PARENT_RUNS_FOLDER, "messages.parquet")


# if not command_files or not message_files:
#     st.error(f"No commands_* or messages_* parquet files found in {PARENT_RUNS_FOLDER}")
#     st.stop()


# Load existing Parquet cache if present
existing_db_file_duration = load_cache("db_file_duration")
existing_runtime = load_cache("runtime")
existing_reboiler = load_cache("reboiler_temperature")
existing_bigfan_pwm = load_cache("big_fan_pwm")
existing_recycle_pump_1_rpm_df = load_cache("recycle_pump_1_rpm")
existing_recycle_pump_2_rpm_df = load_cache("recycle_pump_2_rpm")
existing_weather = load_cache("weather_classification")
existing_sorb = load_cache("sorbent_temperature")
existing_mean = load_cache("mean_values")
existing_mean_residence = load_cache("mean_residence_time")
existing_compressor_stage_1 = load_cache("compressor_first_stage_pressure")
existing_compressor_stage_2 = load_cache("compressor_second_stage_pressure")
existing_compressor_stage_1_temp = load_cache("compressor_first_stage_temperature")
existing_compressor_stage_2_temp = load_cache("compressor_second_stage_temperature")
existing_aec_stack_1_current = load_cache("aec_stack_1_current")
existing_aec_stack_2_current = load_cache("aec_stack_2_current")
existing_aec_stack_3_current = load_cache("aec_stack_3_current")
existing_aec_stack_4_current = load_cache("aec_stack_4_current")
existing_aec_stack_1_current_density = load_cache("aec_stack_1_current_density")
existing_aec_stack_2_current_density = load_cache("aec_stack_2_current_density")
existing_aec_stack_3_current_density = load_cache("aec_stack_3_current_density")
existing_aec_stack_4_current_density = load_cache("aec_stack_4_current_density")
existing_aec_oxygen_temp = load_cache("aec_oxygen_flash_temperature")
existing_aec_hydrogen_temp = load_cache("aec_hydrogen_flash_temperature")
existing_aec_oxygen_pressure = load_cache("aec_oxygen_pressure")
existing_active_state = load_cache("active_state")

# Existing db files already processed
existing_db_files = set(existing_runtime["db File"].dropna().astype(str).unique()) if not existing_runtime.empty else set()

# Pair commands.parquet and messages.parquet by folder name
paired_files = []
for cmd in command_files:
    folder = cmd.parent.name                         # ex: DATA2025-10-27-12-26-48
    msg = cmd.parent / "messages.parquet"
    if msg.exists():
        paired_files.append((cmd, msg))

# Filter out already processed db files
new_pairs = []
for cmd, msg in paired_files:
    db_file = extract_db_file_name(cmd)              # now returns folder name
    if db_file not in existing_db_files:
        new_pairs.append((cmd, msg))


# Prepare collectors
all_db_file_duration = []
all_runtime, all_reboiler, all_bigfan_pwm = [], [], []
all_sorb = []
all_recycle_pump_1_rpm, all_recycle_pump_2_rpm = [], []
all_weather, all_mean, all_mean_residence = [], [], []
all_compressor_stage_1, all_compressor_stage_2 = [], []
all_compressor_stage_1_temp, all_compressor_stage_2_temp = [], []
all_aec_stack_1_current, all_aec_stack_2_current = [], []
all_aec_stack_3_current, all_aec_stack_4_current = [], []
all_aec_stack_1_current_density, all_aec_stack_2_current_density = [], []
all_aec_stack_3_current_density, all_aec_stack_4_current_density = [], []
all_aec_oxygen_temp, all_aec_hydrogen_temp = [], []
all_aec_oxygen_pressure = []
all_active_state = []


if new_pairs:
    with st.spinner("Computing runtimes for missing db files..."):
        for commands_file, messages_file in new_pairs:
            db_file = extract_db_file_name(commands_file)
            st.markdown(
                f"""<div style="background-color:#F4F4F4;padding:12px 18px;border-left:6px solid #4A4A4A;border-radius:6px;margin-bottom:15px;box-shadow:0 2px 4px rgba(0,0,0,0.08);font-family:Segoe UI, Roboto, sans-serif;">
                <span style="color:#2F2B2C;font-weight:600;font-size:14px;letter-spacing:0.5px;">🔄 Processing <span style="color:#2F2B2C;">{db_file}</span> ...</span></div>""",
                unsafe_allow_html=True
            )


            df_db_file_duration = compute_db_file_duration(messages_file)
            df_run = compute_all_runtimes(messages_file, durations_df)
            df_reb = compute_reboiler_temperature_for_file(messages_file, durations_df)
            df_sorb = compute_sorbent_temperature_for_file(messages_file, durations_df)
            df_pwm = compute_big_fan_pwm_for_file(messages_file)
            df_rpm = compute_recycle_pump_1_rpm_for_file(messages_file, durations_df)
            df_rpm2 = compute_recycle_pump_2_rpm_for_file(messages_file, durations_df)
            df_weather = compute_weather_classification_for_file(messages_file)
            df_mean = compute_mean(messages_file)
            df_mean_residence = compute_mean_residence(commands_file, messages_file)
            df_comp1 = compute_compressor_first_stage_pressure_for_file(messages_file, durations_df)
            df_comp2 = compute_compressor_second_stage_pressure_for_file(messages_file, durations_df)
            df_comp1_temp = compute_compressor_first_stage_temperature_for_file(messages_file, durations_df)
            df_comp2_temp = compute_compressor_second_stage_temperature_for_file(messages_file, durations_df)
            df_aec1 = compute_aec_stack_1_current_for_file(messages_file, durations_df)
            df_aec2 = compute_aec_stack_2_current_for_file(messages_file, durations_df)
            df_aec3 = compute_aec_stack_3_current_for_file(messages_file, durations_df)
            df_aec4 = compute_aec_stack_4_current_for_file(messages_file, durations_df)
            df_aec1_den = compute_aec_stack_1_current_density_for_file(messages_file, durations_df)
            df_aec2_den = compute_aec_stack_2_current_density_for_file(messages_file, durations_df)
            df_aec3_den = compute_aec_stack_3_current_density_for_file(messages_file, durations_df)
            df_aec4_den = compute_aec_stack_4_current_density_for_file(messages_file, durations_df)
            df_aec_oxy_temp = compute_aec_oxygen_temperature_for_file(messages_file, durations_df)
            df_aec_hyd_temp = compute_aec_hydrogen_temperature_for_file(messages_file, durations_df)
            df_aec_oxy_pres = compute_aec_oxygen_pressure_for_file(messages_file, durations_df)
            df_active = compute_active_state_for_file(messages_file, durations_df)


            # 1. Update durations_df with raw bounds for this file
            df_duration_raw = compute_db_file_duration_raw(messages_file)
            durations_df = pd.concat([durations_df, df_duration_raw], ignore_index=True)
            durations_df = durations_df.drop_duplicates(subset=["db File"]).reset_index(drop=True)

            # After all compute_* calls for this file, add this:
            df_msg_raw = read_parquet_safe(messages_file)
            if df_msg_raw.empty or "timestamp" not in df_msg_raw.columns:
                start_ts, end_ts = None, None
            else:
                df_msg_raw["timestamp"] = pd.to_numeric(df_msg_raw["timestamp"], errors="coerce")
                df_msg_raw = df_msg_raw.dropna(subset=["timestamp"]).sort_values("timestamp")
                start_ts = df_msg_raw["timestamp"].iloc[0] if not df_msg_raw.empty else None
                end_ts = df_msg_raw["timestamp"].iloc[-1] if not df_msg_raw.empty else None

            # Append to collectors
            if not df_db_file_duration.empty:
                all_db_file_duration.append(df_db_file_duration)
            if not df_run.empty:
                all_runtime.append(df_run)
            if not df_reb.empty:
                all_reboiler.append(df_reb)
            if not df_sorb.empty:
                all_sorb.append(df_sorb)
            if not df_pwm.empty:
                all_bigfan_pwm.append(df_pwm)
            if not df_rpm.empty:
                all_recycle_pump_1_rpm.append(df_rpm)
            if not df_rpm2.empty:
                all_recycle_pump_2_rpm.append(df_rpm2)
            if not df_weather.empty:
                all_weather.append(df_weather)
            if not df_mean.empty:
                all_mean.append(df_mean)
            if not df_mean_residence.empty:
                all_mean_residence.append(df_mean_residence)
            if not df_comp1.empty:
                all_compressor_stage_1.append(df_comp1)
            if not df_comp2.empty:
                all_compressor_stage_2.append(df_comp2)
            if not df_comp1_temp.empty:
                all_compressor_stage_1_temp.append(df_comp1_temp)
            if not df_comp2_temp.empty:
                all_compressor_stage_2_temp.append(df_comp2_temp)
            if not df_aec1.empty:
                all_aec_stack_1_current.append(df_aec1)
            if not df_aec2.empty:
                all_aec_stack_2_current.append(df_aec2)
            if not df_aec3.empty:
                all_aec_stack_3_current.append(df_aec3)
            if not df_aec4.empty:
                all_aec_stack_4_current.append(df_aec4)
            if not df_aec1_den.empty:
                all_aec_stack_1_current_density.append(df_aec1_den)
            if not df_aec2_den.empty:
                all_aec_stack_2_current_density.append(df_aec2_den)
            if not df_aec3_den.empty:
                all_aec_stack_3_current_density.append(df_aec3_den)
            if not df_aec4_den.empty:
                all_aec_stack_4_current_density.append(df_aec4_den)
            if not df_aec_oxy_temp.empty:
                all_aec_oxygen_temp.append(df_aec_oxy_temp)
            if not df_aec_hyd_temp.empty:
                all_aec_hydrogen_temp.append(df_aec_hyd_temp)
            if not df_aec_oxy_pres.empty:
                all_aec_oxygen_pressure.append(df_aec_oxy_pres)
            if not df_active.empty:
                all_active_state.append(df_active)

            # --- PATCH: Save to Excel after each db file ---
            # Concatenate new results so far
            new_db_file_duration_df = pd.concat(all_db_file_duration, ignore_index=True) if all_db_file_duration else pd.DataFrame()
            new_runtime_df = pd.concat(all_runtime, ignore_index=True) if all_runtime else pd.DataFrame()
            new_reboiler_df = pd.concat(all_reboiler, ignore_index=True) if all_reboiler else pd.DataFrame()
            new_sorb_df = pd.concat(all_sorb, ignore_index=True) if all_sorb else pd.DataFrame()
            new_bigfan_pwm_df = pd.concat(all_bigfan_pwm, ignore_index=True) if all_bigfan_pwm else pd.DataFrame()
            new_recycle_pump_1_rpm_df = pd.concat(all_recycle_pump_1_rpm, ignore_index=True) if all_recycle_pump_1_rpm else pd.DataFrame()
            new_recycle_pump_2_rpm_df = pd.concat(all_recycle_pump_2_rpm, ignore_index=True) if all_recycle_pump_2_rpm else pd.DataFrame()
            new_weather_df = pd.concat(all_weather, ignore_index=True) if all_weather else pd.DataFrame()
            new_mean_df = pd.concat(all_mean, ignore_index=True) if all_mean else pd.DataFrame()
            new_mean_residence_df = pd.concat(all_mean_residence, ignore_index=True) if all_mean_residence else pd.DataFrame()
            new_compressor_stage_1_df = pd.concat(all_compressor_stage_1, ignore_index=True) if all_compressor_stage_1 else pd.DataFrame()
            new_compressor_stage_2_df = pd.concat(all_compressor_stage_2, ignore_index=True) if all_compressor_stage_2 else pd.DataFrame()
            new_compressor_stage_1_temp_df = pd.concat(all_compressor_stage_1_temp, ignore_index=True) if all_compressor_stage_1_temp else pd.DataFrame()
            new_compressor_stage_2_temp_df = pd.concat(all_compressor_stage_2_temp, ignore_index=True) if all_compressor_stage_2_temp else pd.DataFrame()
            new_aec_stack_1_current_df = pd.concat(all_aec_stack_1_current, ignore_index=True) if all_aec_stack_1_current else pd.DataFrame()
            new_aec_stack_2_current_df = pd.concat(all_aec_stack_2_current, ignore_index=True) if all_aec_stack_2_current else pd.DataFrame()
            new_aec_stack_3_current_df = pd.concat(all_aec_stack_3_current, ignore_index=True) if all_aec_stack_3_current else pd.DataFrame()
            new_aec_stack_4_current_df = pd.concat(all_aec_stack_4_current, ignore_index=True) if all_aec_stack_4_current else pd.DataFrame()
            new_aec_stack_1_current_density_df = pd.concat(all_aec_stack_1_current_density, ignore_index=True) if all_aec_stack_1_current_density else pd.DataFrame()
            new_aec_stack_2_current_density_df = pd.concat(all_aec_stack_2_current_density, ignore_index=True) if all_aec_stack_2_current_density else pd.DataFrame()
            new_aec_stack_3_current_density_df = pd.concat(all_aec_stack_3_current_density, ignore_index=True) if all_aec_stack_3_current_density else pd.DataFrame()
            new_aec_stack_4_current_density_df = pd.concat(all_aec_stack_4_current_density, ignore_index=True) if all_aec_stack_4_current_density else pd.DataFrame()
            new_aec_oxygen_temp_df = pd.concat(all_aec_oxygen_temp, ignore_index=True) if all_aec_oxygen_temp else pd.DataFrame()
            new_aec_hydrogen_temp_df = pd.concat(all_aec_hydrogen_temp, ignore_index=True) if all_aec_hydrogen_temp else pd.DataFrame()
            new_aec_oxygen_pressure_df = pd.concat(all_aec_oxygen_pressure, ignore_index=True) if all_aec_oxygen_pressure else pd.DataFrame()
            new_active_state_df = pd.concat(all_active_state, ignore_index=True) if all_active_state else pd.DataFrame()

            # Merge with existing
            db_file_duration_df = pd.concat([existing_db_file_duration, new_db_file_duration_df], ignore_index=True) if not existing_db_file_duration.empty or not new_db_file_duration_df.empty else pd.DataFrame()
            runtime_df = pd.concat([existing_runtime, new_runtime_df], ignore_index=True) if not existing_runtime.empty or not new_runtime_df.empty else pd.DataFrame()
            reboiler_df = pd.concat([existing_reboiler, new_reboiler_df], ignore_index=True) if not existing_reboiler.empty or not new_reboiler_df.empty else pd.DataFrame()
            sorb_df = pd.concat([existing_sorb, new_sorb_df], ignore_index=True) if not existing_sorb.empty or not new_sorb_df.empty else pd.DataFrame()
            bigfan_pwm_df = pd.concat([existing_bigfan_pwm, new_bigfan_pwm_df], ignore_index=True) if not existing_bigfan_pwm.empty or not new_bigfan_pwm_df.empty else pd.DataFrame()
            recycle_pump_1_rpm_df = pd.concat([existing_recycle_pump_1_rpm_df, new_recycle_pump_1_rpm_df], ignore_index=True) if not existing_recycle_pump_1_rpm_df.empty or not new_recycle_pump_1_rpm_df.empty else pd.DataFrame()
            recycle_pump_2_rpm_df = pd.concat([existing_recycle_pump_2_rpm_df, new_recycle_pump_2_rpm_df], ignore_index=True) if not existing_recycle_pump_2_rpm_df.empty or not new_recycle_pump_2_rpm_df.empty else pd.DataFrame()
            weather_df = pd.concat([existing_weather, new_weather_df], ignore_index=True) if not existing_weather.empty or not new_weather_df.empty else pd.DataFrame()
            mean_df = pd.concat([existing_mean, new_mean_df], ignore_index=True) if not existing_mean.empty or not new_mean_df.empty else pd.DataFrame()
            mean_residence_df = pd.concat([existing_mean_residence, new_mean_residence_df], ignore_index=True) if not existing_mean_residence.empty or not new_mean_residence_df.empty else pd.DataFrame()
            compressor_stage_1_df = pd.concat([existing_compressor_stage_1, new_compressor_stage_1_df], ignore_index=True) if not existing_compressor_stage_1.empty or not new_compressor_stage_1_df.empty else pd.DataFrame()
            compressor_stage_2_df = pd.concat([existing_compressor_stage_2, new_compressor_stage_2_df], ignore_index=True) if not existing_compressor_stage_2.empty or not new_compressor_stage_2_df.empty else pd.DataFrame()
            compressor_stage_1_temp_df = pd.concat([existing_compressor_stage_1_temp, new_compressor_stage_1_temp_df], ignore_index=True) if not existing_compressor_stage_1_temp.empty or not new_compressor_stage_1_temp_df.empty else pd.DataFrame()
            compressor_stage_2_temp_df = pd.concat([existing_compressor_stage_2_temp, new_compressor_stage_2_temp_df], ignore_index=True) if not existing_compressor_stage_2_temp.empty or not new_compressor_stage_2_temp_df.empty else pd.DataFrame()
            aec_stack_1_df = pd.concat([existing_aec_stack_1_current, new_aec_stack_1_current_df], ignore_index=True) if not existing_aec_stack_1_current.empty or not new_aec_stack_1_current_df.empty else pd.DataFrame()
            aec_stack_2_df = pd.concat([existing_aec_stack_2_current, new_aec_stack_2_current_df], ignore_index=True) if not existing_aec_stack_2_current.empty or not new_aec_stack_2_current_df.empty else pd.DataFrame()
            aec_stack_3_df = pd.concat([existing_aec_stack_3_current, new_aec_stack_3_current_df], ignore_index=True) if not existing_aec_stack_3_current.empty or not new_aec_stack_3_current_df.empty else pd.DataFrame()
            aec_stack_4_df = pd.concat([existing_aec_stack_4_current, new_aec_stack_4_current_df], ignore_index=True) if not existing_aec_stack_4_current.empty or not new_aec_stack_4_current_df.empty else pd.DataFrame()
            aec_stack_1_density_df = pd.concat([existing_aec_stack_1_current_density, new_aec_stack_1_current_density_df], ignore_index=True) if not existing_aec_stack_1_current_density.empty or not new_aec_stack_1_current_density_df.empty else pd.DataFrame()
            aec_stack_2_density_df = pd.concat([existing_aec_stack_2_current_density, new_aec_stack_2_current_density_df], ignore_index=True) if not existing_aec_stack_2_current_density.empty or not new_aec_stack_2_current_density_df.empty else pd.DataFrame()
            aec_stack_3_density_df = pd.concat([existing_aec_stack_3_current_density, new_aec_stack_3_current_density_df], ignore_index=True) if not existing_aec_stack_3_current_density.empty or not new_aec_stack_3_current_density_df.empty else pd.DataFrame()
            aec_stack_4_density_df = pd.concat([existing_aec_stack_4_current_density, new_aec_stack_4_current_density_df], ignore_index=True) if not existing_aec_stack_4_current_density.empty or not new_aec_stack_4_current_density_df.empty else pd.DataFrame()
            aec_oxygen_temp_df = pd.concat([existing_aec_oxygen_temp, new_aec_oxygen_temp_df], ignore_index=True) if not existing_aec_oxygen_temp.empty or not new_aec_oxygen_temp_df.empty else pd.DataFrame()
            aec_hydrogen_temp_df = pd.concat([existing_aec_hydrogen_temp, new_aec_hydrogen_temp_df], ignore_index=True) if not existing_aec_hydrogen_temp.empty or not new_aec_hydrogen_temp_df.empty else pd.DataFrame()
            aec_oxygen_pressure_df = pd.concat([existing_aec_oxygen_pressure, new_aec_oxygen_pressure_df], ignore_index=True) if not existing_aec_oxygen_pressure.empty or not new_aec_oxygen_pressure_df.empty else pd.DataFrame()
            active_state_df = pd.concat([existing_active_state, new_active_state_df], ignore_index=True) if not existing_active_state.empty or not new_active_state_df.empty else pd.DataFrame()

            # Deduplicate
            if not db_file_duration_df.empty:
                db_file_duration_df = db_file_duration_df.drop_duplicates().reset_index(drop=True)
            if not runtime_df.empty:
                runtime_df = runtime_df.drop_duplicates().reset_index(drop=True)
            if not reboiler_df.empty:
                reboiler_df = reboiler_df.drop_duplicates().reset_index(drop=True)
            if not sorb_df.empty:
                sorb_df = sorb_df.drop_duplicates().reset_index(drop=True)
            if not bigfan_pwm_df.empty:
                bigfan_pwm_df = bigfan_pwm_df.drop_duplicates().reset_index(drop=True)
            if not recycle_pump_1_rpm_df.empty:
                recycle_pump_1_rpm_df = recycle_pump_1_rpm_df.drop_duplicates().reset_index(drop=True)
            if not recycle_pump_2_rpm_df.empty:
                recycle_pump_2_rpm_df = recycle_pump_2_rpm_df.drop_duplicates().reset_index(drop=True)
            if not weather_df.empty:
                weather_df = weather_df.drop_duplicates().reset_index(drop=True)
            if not mean_df.empty:
                mean_df = mean_df.drop_duplicates().reset_index(drop=True)
            if not mean_residence_df.empty:
                mean_residence_df = mean_residence_df.drop_duplicates().reset_index(drop=True)
            if not compressor_stage_1_df.empty:
                compressor_stage_1_df = compressor_stage_1_df.drop_duplicates().reset_index(drop=True)
            if not compressor_stage_2_df.empty:
                compressor_stage_2_df = compressor_stage_2_df.drop_duplicates().reset_index(drop=True)
            if not compressor_stage_1_temp_df.empty:
                compressor_stage_1_temp_df = compressor_stage_1_temp_df.drop_duplicates().reset_index(drop=True)
            if not compressor_stage_2_temp_df.empty:
                compressor_stage_2_temp_df = compressor_stage_2_temp_df.drop_duplicates().reset_index(drop=True)
            if not aec_stack_1_df.empty:
                aec_stack_1_df = aec_stack_1_df.drop_duplicates().reset_index(drop=True)
            if not aec_stack_2_df.empty:
                aec_stack_2_df = aec_stack_2_df.drop_duplicates().reset_index(drop=True)
            if not aec_stack_3_df.empty:
                aec_stack_3_df = aec_stack_3_df.drop_duplicates().reset_index(drop=True)
            if not aec_stack_4_df.empty:
                aec_stack_4_df = aec_stack_4_df.drop_duplicates().reset_index(drop=True)
            if not aec_stack_1_density_df.empty:
                aec_stack_1_density_df = aec_stack_1_density_df.drop_duplicates().reset_index(drop=True)
            if not aec_stack_2_density_df.empty:
                aec_stack_2_density_df = aec_stack_2_density_df.drop_duplicates().reset_index(drop=True)
            if not aec_stack_3_density_df.empty:
                aec_stack_3_density_df = aec_stack_3_density_df.drop_duplicates().reset_index(drop=True)
            if not aec_stack_4_density_df.empty:
                aec_stack_4_density_df = aec_stack_4_density_df.drop_duplicates().reset_index(drop=True)
            if not aec_oxygen_temp_df.empty:
                aec_oxygen_temp_df = aec_oxygen_temp_df.drop_duplicates().reset_index(drop=True)
            if not aec_hydrogen_temp_df.empty:
                aec_hydrogen_temp_df = aec_hydrogen_temp_df.drop_duplicates().reset_index(drop=True)
            if not aec_oxygen_pressure_df.empty:
                aec_oxygen_pressure_df = aec_oxygen_pressure_df.drop_duplicates().reset_index(drop=True)
            if not active_state_df.empty:
                active_state_df = active_state_df.drop_duplicates().reset_index(drop=True)

            # Save to Parquet cache after each db file
            save_cache("db_file_duration", db_file_duration_df)
            save_cache("runtime", runtime_df)
            save_cache("reboiler_temperature", reboiler_df)
            save_cache("sorbent_temperature", sorb_df)
            save_cache("big_fan_pwm", bigfan_pwm_df)
            save_cache("recycle_pump_1_rpm", recycle_pump_1_rpm_df)
            save_cache("recycle_pump_2_rpm", recycle_pump_2_rpm_df)
            save_cache("weather_classification", weather_df)
            save_cache("mean_values", mean_df)
            save_cache("mean_residence_time", mean_residence_df)
            save_cache("compressor_first_stage_pressure", compressor_stage_1_df)
            save_cache("compressor_second_stage_pressure", compressor_stage_2_df)
            save_cache("compressor_first_stage_temperature", compressor_stage_1_temp_df)
            save_cache("compressor_second_stage_temperature", compressor_stage_2_temp_df)
            save_cache("aec_stack_1_current", aec_stack_1_df)
            save_cache("aec_stack_2_current", aec_stack_2_df)
            save_cache("aec_stack_3_current", aec_stack_3_df)
            save_cache("aec_stack_4_current", aec_stack_4_df)
            save_cache("aec_stack_1_current_density", aec_stack_1_density_df)
            save_cache("aec_stack_2_current_density", aec_stack_2_density_df)
            save_cache("aec_stack_3_current_density", aec_stack_3_density_df)
            save_cache("aec_stack_4_current_density", aec_stack_4_density_df)
            save_cache("aec_oxygen_flash_temperature", aec_oxygen_temp_df)
            save_cache("aec_hydrogen_flash_temperature", aec_hydrogen_temp_df)
            save_cache("aec_oxygen_pressure", aec_oxygen_pressure_df)
            save_cache("active_state", active_state_df)
            st.success(f"Processed new run: {db_file} — cache updated!")

    # Calculate number of new runs processed
    num_new_runs = len(all_db_file_duration)
    st.success(f"Processed {num_new_runs} new run(s) — cache updated!")
else:
    st.info("No new db files to process. Using existing Parquet cache contents.")

# Ensure all summary DataFrames are defined, even if no new db files were processed
if 'runtime_df' not in locals():
    runtime_df = load_cache("runtime")
if 'reboiler_df' not in locals():
    reboiler_df = load_cache("reboiler_temperature")
if 'sorb_df' not in locals():
    sorb_df = load_cache("sorbent_temperature")
if 'recycle_pump_1_rpm_df' not in locals():
    recycle_pump_1_rpm_df = load_cache("recycle_pump_1_rpm")
if 'recycle_pump_2_rpm_df' not in locals():
    recycle_pump_2_rpm_df = load_cache("recycle_pump_2_rpm")
if 'compressor_stage_1_df' not in locals():
    compressor_stage_1_df = load_cache("compressor_first_stage_pressure")
if 'compressor_stage_2_df' not in locals():
    compressor_stage_2_df = load_cache("compressor_second_stage_pressure")
if 'compressor_stage_1_temp_df' not in locals():
    compressor_stage_1_temp_df = load_cache("compressor_first_stage_temperature")
if 'compressor_stage_2_temp_df' not in locals():
    compressor_stage_2_temp_df = load_cache("compressor_second_stage_temperature")
if 'aec_stack_1_df' not in locals():
    aec_stack_1_df = load_cache("aec_stack_1_current")
if 'aec_stack_2_df' not in locals():
    aec_stack_2_df = load_cache("aec_stack_2_current")
if 'aec_stack_3_df' not in locals():
    aec_stack_3_df = load_cache("aec_stack_3_current")
if 'aec_stack_4_df' not in locals():
    aec_stack_4_df = load_cache("aec_stack_4_current")
if 'aec_stack_1_density_df' not in locals():
    aec_stack_1_density_df = load_cache("aec_stack_1_current_density")
if 'aec_stack_2_density_df' not in locals():
    aec_stack_2_density_df = load_cache("aec_stack_2_current_density")
if 'aec_stack_3_density_df' not in locals():
    aec_stack_3_density_df = load_cache("aec_stack_3_current_density")
if 'aec_stack_4_density_df' not in locals():
    aec_stack_4_density_df = load_cache("aec_stack_4_current_density")
if 'aec_oxygen_temp_df' not in locals():
    aec_oxygen_temp_df = load_cache("aec_oxygen_flash_temperature")
if 'aec_hydrogen_temp_df' not in locals():
    aec_hydrogen_temp_df = load_cache("aec_hydrogen_flash_temperature")
if 'aec_oxygen_pressure_df' not in locals():
    aec_oxygen_pressure_df = load_cache("aec_oxygen_pressure")
if 'active_state_df' not in locals():
    active_state_df = load_cache("active_state")




# -------------------------
# Importing the grand mean values from Parquet cache
# -------------------------
mean_values_df = load_cache("mean_values")
if not mean_values_df.empty:
    required_cols = [
        "Average Reboiler Temperature (°C)",
        "Average Sorbent Top Temperature (°C)",
        "Average Big Fan PWM",
        "Average Recycle Pump 1 RPM",
        "Average Recycle Pump 2 RPM",
        "Average Compressor First Stage Pressure (bar)",
        "Average Compressor Second Stage Pressure (bar)",
        "Average Compressor First Stage Temperature (°C)",
        "Average Compressor Second Stage Temperature (°C)",
        "Average AEC Stack 1 Current (A)",
        "Average AEC Stack 2 Current (A)",
        "Average AEC Stack 3 Current (A)",
        "Average AEC Stack 4 Current (A)",
        "Average AEC Stack 1 Current Density (A/cm²)",
        "Average AEC Stack 2 Current Density (A/cm²)",
        "Average AEC Stack 3 Current Density (A/cm²)",
        "Average AEC Stack 4 Current Density (A/cm²)",
        "Average AEC Oxygen Flash Temperature (°C)",
        "Average AEC Hydrogen Flash Temperature (°C)",
        "Average AEC Oxygen Pressure (bar)"
    ]
    if all(col in mean_values_df.columns for col in required_cols):
        grand_mean_reboiler_temp = mean_values_df["Average Reboiler Temperature (°C)"].mean()
        grand_mean_sorbent_temp = mean_values_df["Average Sorbent Top Temperature (°C)"].mean()
        grand_mean_big_fan_pwm = mean_values_df["Average Big Fan PWM"].mean()
        grand_mean_recycle_pump_1_rpm = mean_values_df["Average Recycle Pump 1 RPM"].mean()
        grand_mean_recycle_pump_2_rpm = mean_values_df["Average Recycle Pump 2 RPM"].mean()
        grand_mean_compressor_stage_1 = mean_values_df["Average Compressor First Stage Pressure (bar)"].mean()
        grand_mean_compressor_stage_2 = mean_values_df["Average Compressor Second Stage Pressure (bar)"].mean()
        grand_mean_compressor_stage_1_temp = mean_values_df["Average Compressor First Stage Temperature (°C)"].mean()
        grand_mean_compressor_stage_2_temp = mean_values_df["Average Compressor Second Stage Temperature (°C)"].mean()
        grand_mean_aec_stack_1_current = mean_values_df["Average AEC Stack 1 Current (A)"].mean()
        grand_mean_aec_stack_2_current = mean_values_df["Average AEC Stack 2 Current (A)"].mean()
        grand_mean_aec_stack_3_current = mean_values_df["Average AEC Stack 3 Current (A)"].mean()
        grand_mean_aec_stack_4_current = mean_values_df["Average AEC Stack 4 Current (A)"].mean()
        grand_mean_aec_stack_1_current_density = mean_values_df["Average AEC Stack 1 Current Density (A/cm²)"].mean()
        grand_mean_aec_stack_2_current_density = mean_values_df["Average AEC Stack 2 Current Density (A/cm²)"].mean()
        grand_mean_aec_stack_3_current_density = mean_values_df["Average AEC Stack 3 Current Density (A/cm²)"].mean()
        grand_mean_aec_stack_4_current_density = mean_values_df["Average AEC Stack 4 Current Density (A/cm²)"].mean()
        grand_mean_aec_oxygen_temp = mean_values_df["Average AEC Oxygen Flash Temperature (°C)"].mean()
        grand_mean_aec_hydrogen_temp = mean_values_df["Average AEC Hydrogen Flash Temperature (°C)"].mean()
        grand_mean_aec_oxygen_pressure = mean_values_df["Average AEC Oxygen Pressure (bar)"].mean()
    else:
        st.warning("Required columns not found in mean_values cache.")
else:
    st.info("No mean_values data available yet.")

# Load active state data once for dashboard
active_state_df = load_cache("active_state")
active_state_df.columns = active_state_df.columns.str.strip()

# -------------------------
# Subsystem Selection (Home Page)
# -------------------------
st.markdown("## Select a Subsystem to Explore")

# ---------- TOP ROW ----------
col1, col2, col3 = st.columns(3)

with col1:
    dac_clicked = st.button("🌀 Direct Air Capture", use_container_width=True)

with col2:
    fm_clicked = st.button("⚙️ Fluid Machinery", use_container_width=True)

with col3:
    ae_clicked = st.button("⚡ Alkaline Electrolyser", use_container_width=True)


# ---------- BOTTOM ROW ----------
col4, col5 = st.columns(2)

with col4:
    msr_clicked = st.button("🛢️ Methanol Synthesis Reactor", use_container_width=True)

with col5:
    mdist_clicked = st.button("🏭 Methanol Distillation Column", use_container_width=True)

# Store selection in session state
if dac_clicked:
    st.session_state["subsystem"] = "Direct Air Capture"
elif fm_clicked:
    st.session_state["subsystem"] = "Fluid Machinery"
elif ae_clicked:
    st.session_state["subsystem"] = "Alkaline Electrolyser"
elif msr_clicked:
    st.session_state["subsystem"] = "Methanol Synthesis Reactor"
elif mdist_clicked:
    st.session_state["subsystem"] = "Methanol Distillation Column"

selected_subsystem = st.session_state.get("subsystem", None)


if selected_subsystem == "Direct Air Capture":
    st.markdown("### 🌀 Direct Air Capture Dashboard")

    # -------------------------
    # Visualization: honest ordering
    # -------------------------
    st.markdown("## ⚙️ Equipment Runtime Summary")
    if not runtime_df.empty:
        summary = (
            runtime_df.groupby("Component", as_index=False)["Hours of operation (hr)"]
            .sum()
            .sort_values("Hours of operation (hr)", ascending=False)
        )

        # Desired custom order (without Total GUI Time or Active State for pie chart)
        custom_order = [
            "Big Fan", "Recycle Pump 1", "Recycle Pump 2", "Feed Pump",
            "Heater 1", "Heater 2", "Heater 3", "Heater 4", "Heater 5", "Heater 6"
        ]

        # ...existing code...

        # Build ordered dict of components present
        comp_hours = summary.set_index("Component")["Hours of operation (hr)"].to_dict()

        # Apply offsets once before visualization
        offsets = {
            "Big Fan": 306.60,
            "Recycle Pump 1": 335.43,
            "Feed Pump": 222.87
        }
        for comp, offset in offsets.items():
            if comp in comp_hours:
                comp_hours[comp] += offset

        # Prepare card values (add Total GUI Time and Active State at the end)
        ordered = {k: comp_hours[k] for k in custom_order if k in comp_hours}
        ordered["Total GUI Time"] = total_gui_time

        # --- Add Active State (DAC) ---
        total_active_time_dac = 0.0
        if not active_state_df.empty and "Active Time DAC (hr)" in active_state_df.columns:
            total_active_time_dac = pd.to_numeric(active_state_df["Active Time DAC (hr)"], errors="coerce").sum()
        ordered["Active State (DAC)"] = total_active_time_dac

        # Dynamic metric grid (5 per row)
        components = list(ordered.keys())
        values = list(ordered.values())
        cols_per_row = 5
        for i in range(0, len(components), cols_per_row):
            row_comps = components[i:i + cols_per_row]
            row_vals = values[i:i + cols_per_row]
            cols = st.columns(len(row_comps))
            for j, comp in enumerate(row_comps):
                with cols[j]:
                    st.markdown(
                    f"""
                    <div style="
                        background: #f8f9fa;
                        border-radius: 12px;
                        padding: 15px 18px;
                        text-align: center;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                        border: 1px solid #e6e6e6;
                    ">
                        <div style="font-size: 16px; font-weight: 600;">{comp}</div>
                        <div style="font-size: 26px; margin-top: 4px; color:#2a2a2a;">
                            {row_vals[j]:.2f} hr
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Pie chart (do NOT include Total GUI Time or Active State)
        pie_components = [k for k in custom_order if k in comp_hours]
        pie_values = [comp_hours[k] for k in pie_components]
        df_plot = pd.DataFrame({"Component": pie_components, "Hours": pie_values})
        df_plot["Component"] = pd.Categorical(df_plot["Component"], categories=custom_order, ordered=True)
        df_plot = df_plot.sort_values("Component")
        fig_pie = px.pie(df_plot, names="Component", values="Hours", hole=0.5)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}: %{value:.2f} hr")
        fig_pie.update_layout(
            height=500,
            margin=dict(l=40, r=40, t=40, b=100),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    

    else:
        st.info("No runtime data available to visualize.")



    # -------------------------------
    # Reboiler and Sorbent Temperature Histograms
    # -------------------------------

    col1, col2 = st.columns(2)

    with col1:
        plot_categorical_histogram(
            reboiler_df,
            category_col="Temperature range",
            value_col="Hours of operation (hr)",
            custom_order=["<=100°C","100°C<T<=110°C","110°C<T<=120°C","120°C<T<=130°C",
                        "130°C<T<=135°C","135°C<T<=140°C","140°C<T<=145°C",
                        "145°C<T<=150°C","150°C<T<=155°C",">155°C"],
            title="🌡️ Reboiler Temperature Duration",
            mean_value=locals().get("grand_mean_reboiler_temp"),
            mean_label="<b>Average: {val:.2f} °C</b>",
            filename="reboiler_histogram.csv",
            xaxis_title="Temperature range",
            yaxis_title="Hours of operation (hr)"
        )

    with col2:
        plot_categorical_histogram(
            sorb_df,
            category_col="Temperature range",
            value_col="Hours of operation (hr)",
            custom_order=["<=60°C","60°C<T<=70°C","70°C<T<=80°C","80°C<T<=85°C",
                        "85°C<T<=90°C","90°C<T<=95°C","95°C<T<=100°C",
                        "100°C<T<=105°C",">105°C"],
            title="🌡️ Sorbent Temperature Duration",
            mean_value=locals().get("grand_mean_sorbent_temp"),
            mean_label="<b>Average: {val:.2f} °C</b>",
            filename="sorbent_histogram.csv",
            xaxis_title="Temperature range",
            yaxis_title="Hours of operation (hr)"
        )

    # -------------------------------
    # Histograms for Recycle Pump RPMs
    # -------------------------------

    col1, col2 = st.columns(2)

    rpm_order = ["<=10","10<RPM<=20","20<RPM<=30","30<RPM<=40",
                "40<RPM<=50","50<RPM<=60","60<RPM<=70","70<RPM<=80",
                "80<RPM<=90",">90"]

    with col1:
        plot_categorical_histogram(
            recycle_pump_1_rpm_df,
            category_col="RPM Category",
            value_col="Hours of operation (hr)",
            custom_order=rpm_order,
            title="🕑 Recycle Pump 1 RPM Duration",
            mean_value=locals().get("grand_mean_recycle_pump_1_rpm"),
            mean_label="<b>Average RPM: {val:.2f}</b>",
            filename="pump1_rpm_histogram.csv",
            xaxis_title="RPM Category",
            yaxis_title="Hours of operation (hr)"
        )

    with col2:
        plot_categorical_histogram(
            recycle_pump_2_rpm_df,
            category_col="RPM Category",
            value_col="Hours of operation (hr)",
            custom_order=rpm_order,
            title="🕑 Recycle Pump 2 RPM Duration",
            mean_value=locals().get("grand_mean_recycle_pump_2_rpm"),
            mean_label="<b>Average RPM: {val:.2f}</b>",
            filename="pump2_rpm_histogram.csv",
            xaxis_title="RPM Category",
            yaxis_title="Hours of operation (hr)"
        )


    # -------------------------
    # Read weather_classification sheet from runtime_summary.xlsx
    # -------------------------
    def load_weather_classification_from_cache() -> pd.DataFrame:
        df = load_cache("weather_classification")
        # Ensure consistency: rename Run -> db File if needed
        if "Run" in df.columns:
            df = df.rename(columns={"Run": "db File"})
        return df

    # -------------------------
    # Weather classification of previous runs
    # -------------------------
    additional_runs_data = [
        {"db File": "Run_1", "Average Temperature (°C)": 31.7, "Average Relative Humidity (%)": 37.70},
        {"db File": "Run_2", "Average Temperature (°C)": 22.73, "Average Relative Humidity (%)": 66.90},
        {"db File": "Run_3", "Average Temperature (°C)": 26.50, "Average Relative Humidity (%)": 51.45},
        {"db File": "Run_4", "Average Temperature (°C)": 23.05, "Average Relative Humidity (%)": 58.53},
        {"db File": "Run_6", "Average Temperature (°C)": 30.76, "Average Relative Humidity (%)": 46.30},
        {"db File": "Run_7", "Average Temperature (°C)": 26.54, "Average Relative Humidity (%)": 64.50},
        {"db File": "Run_8", "Average Temperature (°C)": 24.34, "Average Relative Humidity (%)": 69.62},
        {"db File": "Run_9", "Average Temperature (°C)": 26.93, "Average Relative Humidity (%)": 52.80},
        {"db File": "Run_10", "Average Temperature (°C)": 22.68, "Average Relative Humidity (%)": 48.44},
        {"db File": "Run_11", "Average Temperature (°C)": 25.20, "Average Relative Humidity (%)": 61.28},
        {"db File": "Run_12", "Average Temperature (°C)": 23.98, "Average Relative Humidity (%)": 51.85},
        {"db File": "Run_13", "Average Temperature (°C)": 30.05, "Average Relative Humidity (%)": 47.53},
        {"db File": "Run_18", "Average Temperature (°C)": 20.80, "Average Relative Humidity (%)": 76.92},
        {"db File": "Run_19", "Average Temperature (°C)": 15.66, "Average Relative Humidity (%)": 60.22},
        {"db File": "Run_20", "Average Temperature (°C)": 12.94, "Average Relative Humidity (%)": 81.60},
        {"db File": "Run_21", "Average Temperature (°C)": 17.18, "Average Relative Humidity (%)": 53.68},
        {"db File": "Run_22", "Average Temperature (°C)": 21.20, "Average Relative Humidity (%)": 59.86},
        {"db File": "Run_23", "Average Temperature (°C)": 18.60, "Average Relative Humidity (%)": 85.36},
        {"db File": "Run_24", "Average Temperature (°C)": 19.74, "Average Relative Humidity (%)": 78.10},
        {"db File": "Run_25", "Average Temperature (°C)": 21.78, "Average Relative Humidity (%)": 71.96},
        {"db File": "Run_26", "Average Temperature (°C)": 18.34, "Average Relative Humidity (%)": 79.88},
        {"db File": "Run_27", "Average Temperature (°C)": 18.46, "Average Relative Humidity (%)": 83.12},
        {"db File": "Run_29", "Average Temperature (°C)": 17.27, "Average Relative Humidity (%)": 75.40},
        {"db File": "Run_30", "Average Temperature (°C)": 17.77, "Average Relative Humidity (%)": 86.30},
        {"db File": "Run_31", "Average Temperature (°C)": 12.45, "Average Relative Humidity (%)": 99.90},
        {"db File": "Run_32", "Average Temperature (°C)": 13.15, "Average Relative Humidity (%)": 99.90},
        {"db File": "Run_33", "Average Temperature (°C)": 14.70, "Average Relative Humidity (%)": 98.70},
        {"db File": "Run_34", "Average Temperature (°C)": 14.10, "Average Relative Humidity (%)": 78.00},
        {"db File": "Run_39", "Average Temperature (°C)": 14.10, "Average Relative Humidity (%)": 78.00},
        {"db File": "Run_40", "Average Temperature (°C)": 20.10, "Average Relative Humidity (%)": 78.60},
        {"db File": "Run_41", "Average Temperature (°C)": 19.50, "Average Relative Humidity (%)": 76.20},
        {"db File": "Run_42", "Average Temperature (°C)": 16.30, "Average Relative Humidity (%)": 82.90},
        {"db File": "Run_43", "Average Temperature (°C)": 17.30, "Average Relative Humidity (%)": 61.90},
        {"db File": "Run_44", "Average Temperature (°C)": 15.60, "Average Relative Humidity (%)": 64.80},
        {"db File": "Run_45", "Average Temperature (°C)": 18.60, "Average Relative Humidity (%)": 77.20},
        {"db File": "Run_46", "Average Temperature (°C)": 16.20, "Average Relative Humidity (%)": 84.90},
        {"db File": "Run_47", "Average Temperature (°C)": 15.00, "Average Relative Humidity (%)": 99.90},
        {"db File": "Run_48", "Average Temperature (°C)": 16.00, "Average Relative Humidity (%)": 86.00},
        {"db File": "Run_49", "Average Temperature (°C)": 16.60, "Average Relative Humidity (%)": 75.00},
        {"db File": "Run_50", "Average Temperature (°C)": 12.20, "Average Relative Humidity (%)": 99.90},
        {"db File": "Run_53", "Average Temperature (°C)": 11.60, "Average Relative Humidity (%)": 90.30},
        {"db File": "Run_54", "Average Temperature (°C)": 6.80, "Average Relative Humidity (%)": 94.70},
        {"db File": "Run_55", "Average Temperature (°C)": 6.80, "Average Relative Humidity (%)": 82.60},
        {"db File": "Run_56", "Average Temperature (°C)": 11.20, "Average Relative Humidity (%)": 67.80},
        {"db File": "Run_57", "Average Temperature (°C)": 13.60, "Average Relative Humidity (%)": 73.60},
        {"db File": "Run_58", "Average Temperature (°C)": 12.70, "Average Relative Humidity (%)": 68.10},
        {"db File": "Run_59", "Average Temperature (°C)": 12.10, "Average Relative Humidity (%)": 84.20},
        {"db File": "Run_60", "Average Temperature (°C)": 9.60, "Average Relative Humidity (%)": 61.50},
        {"db File": "Run_61", "Average Temperature (°C)": 6.00, "Average Relative Humidity (%)": 89.60}
    ]

    additional_df = pd.DataFrame(additional_runs_data)

    def classify_weather(temp, rh):
        if temp >= 24 and rh >= 60:
            return "Hot and humid"
        elif temp >= 24 and rh <= 40:
            return "Hot and dry"
        elif temp <= 18 and rh >= 60:
            return "Cold and humid"
        elif temp <= 18 and rh <= 40:
            return "Cold and dry"
        elif temp <= 18 and (rh > 40 and rh < 60):
            return "Cold and Moderate humidity"
        elif temp >= 24 and (rh > 40 and rh < 60):
            return "Hot and Moderate humidity"
        elif (temp > 18 and temp < 24) and rh <= 40:
            return "Dry and Moderate temperature"
        elif (temp > 18 and temp < 24) and rh >= 60:
            return "Humid and Moderate temperature"
        else:
            return "Moderate"

    additional_df["Weather Classification"] = additional_df.apply(
        lambda row: classify_weather(row["Average Temperature (°C)"], row["Average Relative Humidity (%)"]),
        axis=1
    )

    weather_df = load_weather_classification_from_cache()
    combined_df = pd.concat([weather_df, additional_df], ignore_index=True)


    # -------------------------
    # Weather classification quadrant plot (extended categories + refinements)
    # -------------------------
    def plot_weather_quadrant(weather_df: pd.DataFrame):
        if weather_df.empty:
            st.info("No weather classification data available for plotting.")
            return

        st.markdown("## 🌤️ Weather Classification")

        # Rename for convenience
        dfw = weather_df.rename(columns={
            "Average Temperature (°C)": "avg_temp",
            "Average Relative Humidity (%)": "avg_rh",
            "Weather Classification": "classification",
            "db File": "db_file"   # ✅ use db File instead of Run
        })

        # Thresholds
        T_COLD, T_HOT = 18.0, 24.0
        RH_DRY, RH_HUMID = 40.0, 60.0

        fig = go.Figure()

        # --- Shaded regions with distinct colors ---
        fig.add_shape(type="rect", x0=0, x1=T_COLD, y0=0, y1=RH_DRY,
                    fillcolor="rgba(173,216,230,0.35)", line=dict(width=0))  # Cold & Dry
        fig.add_shape(type="rect", x0=0, x1=T_COLD, y0=RH_HUMID, y1=100,
                    fillcolor="rgba(0,0,139,0.35)", line=dict(width=0))      # Cold & Humid
        fig.add_shape(type="rect", x0=0, x1=T_COLD, y0=RH_DRY, y1=RH_HUMID,
                    fillcolor="rgba(70,130,180,0.35)", line=dict(width=0))   # Cold & Moderate humidity

        fig.add_shape(type="rect", x0=T_HOT, x1=50, y0=0, y1=RH_DRY,
                    fillcolor="rgba(255,160,122,0.35)", line=dict(width=0))  # Hot & Dry
        fig.add_shape(type="rect", x0=T_HOT, x1=50, y0=RH_HUMID, y1=100,
                    fillcolor="rgba(220,20,60,0.35)", line=dict(width=0))    # Hot & Humid
        fig.add_shape(type="rect", x0=T_HOT, x1=50, y0=RH_DRY, y1=RH_HUMID,
                    fillcolor="rgba(178,34,34,0.35)", line=dict(width=0))    # Hot & Moderate humidity

        fig.add_shape(type="rect", x0=T_COLD, x1=T_HOT, y0=0, y1=RH_DRY,
                    fillcolor="rgba(255,255,0,0.25)", line=dict(width=0))    # Dry & Moderate temp
        fig.add_shape(type="rect", x0=T_COLD, x1=T_HOT, y0=RH_HUMID, y1=100,
                    fillcolor="rgba(0,255,255,0.25)", line=dict(width=0))    # Humid & Moderate temp
        fig.add_shape(type="rect", x0=T_COLD, x1=T_HOT, y0=RH_DRY, y1=RH_HUMID,
                    fillcolor="rgba(0,128,0,0.25)", line=dict(width=0))      # Moderate

        # --- Guideline lines ---
        fig.add_vline(x=T_COLD, line_dash="dot", line_color="gray")
        fig.add_vline(x=T_HOT, line_dash="dot", line_color="gray")
        fig.add_hline(y=RH_DRY, line_dash="dot", line_color="gray")
        fig.add_hline(y=RH_HUMID, line_dash="dot", line_color="gray")

        # --- Zone labels ---
        fig.add_annotation(x=T_COLD/2, y=RH_DRY/2, text="Cold & Dry", showarrow=False, font=dict(color="blue", size=14))
        fig.add_annotation(x=T_COLD/2, y=(RH_HUMID+100)/2, text="Cold & Humid", showarrow=False, font=dict(color="darkblue", size=14))
        fig.add_annotation(x=T_COLD/2, y=(RH_DRY+RH_HUMID)/2, text="Cold & Moderate humidity", showarrow=False, font=dict(color="steelblue", size=14))

        fig.add_annotation(x=(T_HOT+50)/2, y=RH_DRY/2, text="Hot & Dry", showarrow=False, font=dict(color="firebrick", size=14))
        fig.add_annotation(x=(T_HOT+50)/2, y=(RH_HUMID+100)/2, text="Hot & Humid", showarrow=False, font=dict(color="crimson", size=14))
        fig.add_annotation(x=(T_HOT+50)/2, y=(RH_DRY+RH_HUMID)/2, text="Hot & Moderate humidity", showarrow=False, font=dict(color="darkred", size=14))

        fig.add_annotation(x=(T_COLD+T_HOT)/2, y=RH_DRY/2, text="Dry<br>&<br>Moderate<br>temperature", showarrow=False, font=dict(color="goldenrod", size=14))
        fig.add_annotation(x=(T_COLD+T_HOT)/2, y=(RH_HUMID+100)/2, text="Humid<br>&<br>Moderate<br>temperature", showarrow=False, font=dict(color="teal", size=14))
        fig.add_annotation(x=(T_COLD+T_HOT)/2, y=(RH_DRY+RH_HUMID)/2, text="Moderate", showarrow=False, font=dict(color="green", size=14))

        # --- Scatter points (dB Files) ---
        color_map = {
            "Cold and dry": "blue",
            "Cold and humid": "darkblue",
            "Cold and Moderate humidity": "steelblue",
            "Hot and dry": "orange",
            "Hot and humid": "crimson",
            "Hot and Moderate humidity": "darkred",
            "Dry and Moderate temperature": "gold",
            "Humid and Moderate temperature": "cyan",
            "Moderate": "green",
        }
        dfw["color"] = dfw["classification"].map(lambda c: color_map.get(c, "gray"))

        fig.add_trace(go.Scatter(
            x=dfw["avg_temp"],
            y=dfw["avg_rh"],
            mode="markers",
            text=dfw["db_file"],   # ✅ show dB File instead of Run
            textposition="top center",
            marker=dict(size=10, color=dfw["color"], line=dict(width=1, color="white")),
            hovertemplate="db File: %{text}<br>Temp: %{x:.1f} °C<br>RH: %{y:.1f}%<br>Class: %{customdata}",
            customdata=dfw["classification"]
        ))

        # --- Reference points ---
        reference_points = pd.DataFrame([
            {"Location": "Oman coast", "T": 35, "RH": 70},
            {"Location": "Oman interior", "T": 40, "RH": 30},
            {"Location": "Portugal coast", "T": 25, "RH": 70},
            {"Location": "Portugal interior", "T": 32.5, "RH": 40},
        ])
        fig.add_trace(go.Scatter(
            x=reference_points["T"],
            y=reference_points["RH"],
            mode="markers+text",
            text=reference_points["Location"],
            textposition="bottom center",
            marker=dict(size=8, color="black", symbol="diamond"),
            hovertemplate="Location: %{text}<br>Temp: %{x} °C<br>RH: %{y} %"
        ))

        fig.update_layout(
            xaxis_title="Average Temperature (°C)",
            yaxis_title="Average Relative Humidity (%)",
            height=650,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=False)


    # -------------------------
    # Plotting weather classification quadrant
    # -------------------------
    weather_df = load_weather_classification_from_cache()
    plot_weather_quadrant(combined_df)

    # -------------------------
    # Overlay Explorer per db File
    # -------------------------
    st.markdown("---")
    st.header("Interactive Parquet Overlay Explorer")

    messages_files = sorted(PARENT_RUNS_FOLDER.glob("messages.parquet"))
    names_files = sorted(PARENT_RUNS_FOLDER.glob("names.parquet"))
    commands_files = sorted(PARENT_RUNS_FOLDER.glob("commands.parquet"))

    if not messages_files:
        st.error(f"No messages parquet found in {PARENT_RUNS_FOLDER}")
        st.stop()

    # Build db File identifiers from messages files
    db_files = [extract_db_file_name(p) for p in messages_files]
    selected_db_file = st.selectbox("Select a db File", options=sorted(db_files))
    if not selected_db_file:
        st.stop()

    # Match files by suffix (timestamp part)
    def get_suffix(path: Path) -> str:
        # e.g. messages_2025_11_13_09_47_21 → "2025-11-13-09-47-21"
        parts = path.stem.split("_")[1:]
        return "-".join(parts)

    # Extract suffix from selected db File (DATA2025-11-13-09-47-21 → "2025-11-13-09-47-21")
    suffix = selected_db_file.replace("DATA", "")

    messages_path = next((p for p in messages_files if get_suffix(p) == suffix), None)
    names_path = next((p for p in names_files if get_suffix(p) == suffix), None)
    commands_path = next((p for p in commands_files if get_suffix(p) == suffix), None)

    if not messages_path:
        st.error(f"No messages parquet found for {selected_db_file}")
        st.stop()

    # Read messages parquet
    try:
        df_msg = read_messages_parquet(messages_path)
    except Exception as e:
        st.error(f"Failed to read selected messages parquet ({messages_path}): {e}")
        st.stop()

    # Load mapping Excel
    mapping_path = messages_path.parent / "msgID_list.xlsx"
    if not mapping_path.exists():
        st.error(f"No msgID_list.xlsx found in {messages_path.parent}")
        st.stop()

    try:
        mapping = pd.read_excel(mapping_path)
    except Exception as e:
        st.error(f"Failed to read mapping Excel ({mapping_path}): {e}")
        st.stop()

    needed_cols = {"Subsystem", "msgID", "GUI Name"}
    if not needed_cols.issubset(mapping.columns):
        st.error(f"Excel mapping must include columns: {needed_cols} — found: {mapping.columns.tolist()}")
        st.stop()

    # Subsystem selection
    subsystems = sorted(pd.Series(mapping["Subsystem"]).dropna().unique().tolist())
    if not subsystems:
        st.error("No subsystems found in mapping.")
        st.stop()

    col_sel = st.columns(2)
    with col_sel[0]:
        subsystem = st.selectbox("Subsystem", options=subsystems)
        msg_tbl = mapping.loc[mapping["Subsystem"] == subsystem, ["msgID", "GUI Name"]].dropna()
        if msg_tbl.empty:
            st.error(f"No msgIDs/GUI Names for subsystem '{subsystem}'.")
            st.stop()
        msg_tbl["display"] = msg_tbl.apply(lambda r: f"{r['msgID']} - {r['GUI Name']}", axis=1)

    with col_sel[1]:
        msg_displays = st.multiselect(
            "Select one or more msgIDs for overlay",
            options=msg_tbl["display"].tolist(),
            default=msg_tbl["display"].tolist()[:1]
        )
        if not msg_displays:
            st.info("Select at least one signal (msgID) to plot.")
            st.stop()

    # Parse arbitration IDs
    arb_ids, label_map = [], {}
    for disp in msg_displays:
        try:
            arb = int(str(disp).split("-")[0].strip())
            arb_ids.append(arb)
            label_map[arb] = disp
        except Exception:
            st.error(f"Could not parse arbitration_id from selection: '{disp}'")
            st.stop()

    # Validate required columns
    req_cols = {"timestamp", "formatted_data", "arbitration_id"}
    missing = [c for c in req_cols if c not in df_msg.columns]
    if missing:
        st.error(f"Missing required columns in selected messages file: {missing}")
        st.stop()

    # Filter and prepare long-format data
    df_msg = df_msg[["timestamp", "formatted_data", "arbitration_id"]]
    df_msg = df_msg[df_msg["arbitration_id"].isin(arb_ids)]
    df_msg = df_msg.dropna(subset=["timestamp", "formatted_data"]).sort_values(["arbitration_id", "timestamp"])

    if df_msg.empty:
        st.warning("No rows for selected arbitration_id(s) after cleaning.")
        st.stop()

    t0 = float(df_msg["timestamp"].min())
    df_msg["timestamp"] = pd.to_numeric(df_msg["timestamp"], errors="coerce") - t0

    long_full = df_msg.rename(columns={"formatted_data": "value"})
    long_full["signal"] = long_full["arbitration_id"].map(label_map).astype(str)
    long_full = long_full[["timestamp", "value", "signal"]]

    # Apply sidebar filters
    filtered_long, x_range, y_range = sidebar_filter_and_export_fn(long_full)

    with st.sidebar:
        st.markdown("---")
        st.subheader("Rendering")
        max_points = st.slider(
            "Max points to render (total across signals)",
            min_value=1000, max_value=400_000, value=120_000, step=1000,
            key="max_points_overlay"
        )

    long_plot = per_series_downsample_long(filtered_long, max_points=max_points)

    if long_plot.empty:
        st.info("No data to plot after filtering.")
    else:
        fig = px.line(
            long_plot, x="timestamp", y="value", color="signal",
            labels={"timestamp": "Time (s)", "value": "Value", "signal": "Signal"},
            title=f"Overlay Explorer — {selected_db_file}"
        )
        fig.update_layout(height=700, margin=dict(l=40, r=40, t=60, b=40))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        st.plotly_chart(fig, use_container_width=False)

    with st.expander("Quick stats per signal"):
        if not filtered_long.empty:
            stats = filtered_long.groupby("signal").agg(
                points=("value", "size"),
                t_min=("timestamp", "min"),
                t_max=("timestamp", "max"),
                v_min=("value", lambda x: pd.to_numeric(x, errors="coerce").min()),
                v_max=("value", lambda x: pd.to_numeric(x, errors="coerce").max()),
            ).reset_index()
            st.dataframe(stats)
        else:
            st.write("No points in the filtered selection.")




elif selected_subsystem == "Fluid Machinery":
    st.markdown("## ⚙️ Fluid Machinery Runtime Summary")

    TEMPERATURE_LABELS = [label for (label, _) in TEMPERATURE_RANGES]

    if not runtime_df.empty:
        fm_df = runtime_df[runtime_df["Subsystem"] == "FM"]

        if fm_df.empty:
            st.info("No Fluid Machinery runtime data available.")
            st.stop()

        summary = (
            fm_df.groupby("Component", as_index=False)["Hours of operation (hr)"]
                .sum()
                .sort_values("Hours of operation (hr)", ascending=False)
        )

        # Ordering for nice display
        fm_order = [
            "Fan Drying System",
            "Condenser Stage 1",
            "Freezer Stage 2A",
            "Freezer Stage 2B",
            "Compressor First Stage",
            "Compressor Second Stage"
        ]

        comp_hours = summary.set_index("Component")["Hours of operation (hr)"].to_dict()
        ordered = {k: comp_hours.get(k, 0) for k in fm_order}

        # --- Add Active State (FM) ---
        total_active_time_fm = 0.0
        if not active_state_df.empty and "Active Time FM (hr)" in active_state_df.columns:
            total_active_time_fm = pd.to_numeric(active_state_df["Active Time FM (hr)"], errors="coerce").sum()
        ordered["Active State (FM)"] = total_active_time_fm

        # Display metrics as cards
        comps = list(ordered.keys())
        vals = list(ordered.values())

        cols_per_row = 3
        for i in range(0, len(comps), cols_per_row):
            row_comps = comps[i:i+cols_per_row]
            row_vals = vals[i:i+cols_per_row]
            cols = st.columns(len(row_comps))
            for j, comp in enumerate(row_comps):
                with cols[j]:
                    st.markdown(
                    f"""
                    <div style="
                        background: #f8f9fa;
                        border-radius: 12px;
                        padding: 15px 18px;
                        text-align: center;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                        border: 1px solid #e6e6e6;
                    ">
                        <div style="font-size: 16px; font-weight: 600;">{comp}</div>
                        <div style="font-size: 26px; margin-top: 4px; color:#2a2a2a;">
                            {row_vals[j]:.2f} hr
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                    
        pie_comps = [k for k in fm_order if k in comp_hours]
        pie_vals = [comp_hours[k] for k in pie_comps]
        df_plot = pd.DataFrame({"Component": pie_comps, "Hours": pie_vals})
        fig = px.pie(df_plot, names="Component", values="Hours", hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Fluid Machinery: Compressor Pressure Histograms
    # -------------------------------

    # --- Row 1: Compressor stage 1 pressure | Compressor stage 2 pressure
    
    col1, col2 = st.columns(2)

    with col1:
        if not compressor_stage_1_df.empty:

            # Plot First Stage Compressor Pressure Histogram
            plot_categorical_histogram(
                compressor_stage_1_df,
                category_col="Pressure Range",
                value_col="Hours of operation (hr)",
                custom_order=[ "<1 bar","1–10 bar","10–20 bar","20–30 bar","30–40 bar","40–50 bar",">50 bar" ],
                title="📊 Compressor First Stage Pressure Duration",
                mean_value=grand_mean_compressor_stage_1,
                mean_label="<b>Average Pressure: {val:.2f} bar</b>",
                filename="compressor_stage_1_histogram.csv",
                xaxis_title="Pressure Range",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stage 1 pressure histogram data available.")

    with col2:
        if not compressor_stage_2_df.empty:

        # Plot Second Stage Compressor Pressure Histogram
            plot_categorical_histogram(
                compressor_stage_2_df,
                category_col="Pressure Range",
                value_col="Hours of operation (hr)",
                custom_order=[ "<1 bar","1–10 bar","10–20 bar","20–30 bar","30–40 bar","40–50 bar",">50 bar" ],
                title="📊 Compressor Second Stage Pressure Duration",
                mean_value=grand_mean_compressor_stage_2,
                mean_label="<b>Average Pressure: {val:.2f} bar</b>",
                filename="compressor_stage_2_histogram.csv",
                xaxis_title="Pressure Range",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stage 2 pressure histogram data available.")


    # --- Row 2: Compressor stage 1 temperature | Compressor stage 2 temperature

    col1, col2 = st.columns(2)

    with col1:
        if not compressor_stage_1_temp_df.empty:

            # Plot First Stage Compressor Temperature Histogram
            plot_categorical_histogram(
                compressor_stage_1_temp_df,
                category_col="Temperature Range",
                value_col="Hours of operation (hr)",
                custom_order=TEMPERATURE_LABELS,
                title="📊 Compressor First Stage Temperature Duration",
                mean_value=grand_mean_compressor_stage_1_temp,
                mean_label="<b>Average Temperature: {val:.2f} °C</b>",
                filename="compressor_stage_1_temp_histogram.csv",
                xaxis_title="Temperature Range",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stage 1 temperature histogram data available.")
        
    with col2:
        if not compressor_stage_2_temp_df.empty:

            # Plot Second Stage Compressor Temperature Histogram
            plot_categorical_histogram(
                compressor_stage_2_temp_df,
                category_col="Temperature Range",
                value_col="Hours of operation (hr)",
                custom_order=TEMPERATURE_LABELS,
                title="📊 Compressor Second Stage Temperature Duration",
                mean_value=grand_mean_compressor_stage_2_temp,
                mean_label="<b>Average Temperature: {val:.2f} °C</b>",
                filename="compressor_stage_2_temp_histogram.csv",
                xaxis_title="Temperature Range",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stage 2 temperature histogram data available.")


elif selected_subsystem == "Alkaline Electrolyser":
    st.markdown("## ⚙️ Alkaline Electolyser Runtime Summary")

    # ---------------------------------------------------------------------
    # Prepare label orders used by plot_categorical_histogram (reuse your ranges)
    # ---------------------------------------------------------------------
    CURRENT_LABELS = [label for (label, _) in CURRENT_RANGES]
    CURRENT_DENSITY_LABELS = [label for (label, _) in CURRENT_DENSITY_RANGES]
    TEMPERATURE_LABELS = [label for (label, _) in TEMPERATURE_RANGES]
    PRESSURE_LABELS = [label for (label, _) in PRESSURE_RANGES]

    # ---------------------------------------------------------------------
    # 1) Equipment runtime summary for Stack 1..4 (using runtime_df)
    # ---------------------------------------------------------------------

    if not runtime_df.empty:
        aec_df = runtime_df[runtime_df["Subsystem"] == "AEC"]

        if aec_df.empty:
            st.info("No Alkaline Electrolyser runtime data available.")
            st.stop()

        summary = (
            aec_df.groupby("Component", as_index=False)["Hours of operation (hr)"]
                .sum()
                .sort_values("Hours of operation (hr)", ascending=False)
        )

        # Ordering for nice display
        aec_order = [
            "Stack 1",
            "Stack 2",
            "Stack 3",
            "Stack 4"
        ]

        comp_hours = summary.set_index("Component")["Hours of operation (hr)"].to_dict()
        ordered = {k: comp_hours.get(k, 0) for k in aec_order}

        # --- Add Total GUI Time and Active Time (AEC) ---
        # Compute Total Active Time AEC from active_state_df
        total_active_time_aec = 0.0
        if not active_state_df.empty and "Active Time AEC (hr)" in active_state_df.columns:
            total_active_time_aec = pd.to_numeric(active_state_df["Active Time AEC (hr)"], errors="coerce").sum()
        ordered["Total Active Time (AEC)"] = total_active_time_aec

        # Add to cards (not to pie chart)
        ordered["Total GUI Time"] = total_gui_time
        ordered["Total Active Time (AEC)"] = total_active_time_aec

        # Display metrics as cards
        comps = list(ordered.keys())
        vals = list(ordered.values())

        cols_per_row = 2
        for i in range(0, len(comps), cols_per_row):
            row_comps = comps[i:i+cols_per_row]
            row_vals = vals[i:i+cols_per_row]
            cols = st.columns(len(row_comps))
            for j, comp in enumerate(row_comps):
                with cols[j]:
                    st.markdown(
                    f"""
                    <div style="
                        background: #f8f9fa;
                        border-radius: 12px;
                        padding: 15px 18px;
                        text-align: center;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                        border: 1px solid #e6e6e6;
                    ">
                        <div style="font-size: 16px; font-weight: 600;">{comp}</div>
                        <div style="font-size: 26px; margin-top: 4px; color:#2a2a2a;">
                            {row_vals[j]:.2f} hr
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Pie chart (do NOT include Total GUI Time or Total Active Time)
        pie_comps = [k for k in aec_order if k in comp_hours]
        pie_vals = [comp_hours[k] for k in pie_comps]
        df_plot = pd.DataFrame({"Component": pie_comps, "Hours": pie_vals})
        fig = px.pie(df_plot, names="Component", values="Hours", hole=0.5)
        st.plotly_chart(fig, use_container_width=True)


    # --- Row 1: Current Stack 1 (col1) | Current Stack 2 (col2)

    
    col1, col2 = st.columns(2)

    with col1:
        if not aec_stack_1_df.empty:
            plot_categorical_histogram(
                aec_stack_1_df,
                category_col="Current Range" if "Current Range" in aec_stack_1_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=CURRENT_LABELS,
                title="Current - Stack 1",
                mean_value=grand_mean_aec_stack_1_current,
                mean_label="<b>Average Current: {val:.2f} A</b>",
                filename="aec_stack_1_current_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stack 1 current histogram data available.")

    with col2:
        if not aec_stack_2_df.empty:
            plot_categorical_histogram(
                aec_stack_2_df,
                category_col="Current Range" if "Current Range" in aec_stack_2_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=CURRENT_LABELS,
                title="Current - Stack 2",
                mean_value=grand_mean_aec_stack_2_current,
                mean_label="<b>Average Current: {val:.2f} A</b>",
                filename="aec_stack_2_current_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stack 2 current histogram data available.")

    st.markdown("---")

    # --- Row 2: Current Stack 3 | Current Stack 4
    col1, col2 = st.columns(2)

    with col1:
        if not aec_stack_3_df.empty:
            plot_categorical_histogram(
                aec_stack_3_df,
                category_col="Current Range" if "Current Range" in aec_stack_3_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=CURRENT_LABELS,
                title="Current - Stack 3",
                mean_value=grand_mean_aec_stack_3_current,
                mean_label="<b>Average Current: {val:.2f} A</b>",
                filename="aec_stack_3_current_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stack 3 current histogram data available.")

    with col2:
        if not aec_stack_4_df.empty:
            plot_categorical_histogram(
                aec_stack_4_df,
                category_col="Current Range" if "Current Range" in aec_stack_4_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=CURRENT_LABELS,
                title="Current - Stack 4",
                mean_value=grand_mean_aec_stack_4_current,
                mean_label="<b>Average Current: {val:.2f} A</b>",
                filename="aec_stack_4_current_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stack 4 current histogram data available.")

    st.markdown("---")

    # --- Row 3: Current Density Stack1 | Current Density Stack2
    col1, col2 = st.columns(2)

    with col1:
        if not aec_stack_1_density_df.empty:
            plot_categorical_histogram(
                aec_stack_1_density_df,
                category_col="Current Density Range" if "Current Density Range" in aec_stack_1_density_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=CURRENT_DENSITY_LABELS,
                title="Current Density - Stack 1",
                mean_value=grand_mean_aec_stack_1_current_density,
                mean_label="<b>Average Density: {val:.4f} A/cm²</b>",
                filename="aec_stack_1_current_density_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stack 1 current-density histogram data available.")

    with col2:
        if not aec_stack_2_density_df.empty:
            plot_categorical_histogram(
                aec_stack_2_density_df,
                category_col="Current Density Range" if "Current Density Range" in aec_stack_2_density_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=CURRENT_DENSITY_LABELS,
                title="Current Density - Stack 2",
                mean_value=grand_mean_aec_stack_2_current_density,
                mean_label="<b>Average Density: {val:.4f} A/cm²</b>",
                filename="aec_stack_2_current_density_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stack 2 current-density histogram data available.")

    st.markdown("---")

    # --- Row 4: Current Density Stack3 | Current Density Stack4
    col1, col2 = st.columns(2)

    with col1:
        if not aec_stack_3_density_df.empty:
            plot_categorical_histogram(
                aec_stack_3_density_df,
                category_col="Current Density Range" if "Current Density Range" in aec_stack_3_density_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=CURRENT_DENSITY_LABELS,
                title="Current Density - Stack 3",
                mean_value=grand_mean_aec_stack_3_current_density,
                mean_label="<b>Average Density: {val:.4f} A/cm²</b>",
                filename="aec_stack_3_current_density_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stack 3 current-density histogram data available.")

    with col2:
        if not aec_stack_4_density_df.empty:
            plot_categorical_histogram(
                aec_stack_4_density_df,
                category_col="Current Density Range" if "Current Density Range" in aec_stack_4_density_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=CURRENT_DENSITY_LABELS,
                title="Current Density - Stack 4",
                mean_value=grand_mean_aec_stack_4_current_density,
                mean_label="<b>Average Density: {val:.4f} A/cm²</b>",
                filename="aec_stack_4_current_density_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No Stack 4 current-density histogram data available.")

    st.markdown("---")

    # --- Row 5: Oxygen Flash Temperature | Hydrogen Flash Temperature
    col1, col2 = st.columns(2)

    with col1:
        if not aec_oxygen_temp_df.empty:
            plot_categorical_histogram(
                aec_oxygen_temp_df,
                category_col="Temperature Range" if "Temperature Range" in aec_oxygen_temp_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=TEMPERATURE_LABELS,
                title="Oxygen Flash Temperature",
                mean_value=grand_mean_aec_oxygen_temp,
                mean_label="<b>Average O₂ Temp: {val:.2f} °C</b>",
                filename="aec_oxygen_flash_temperature_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No oxygen flash temperature histogram data available.")

    with col2:
        if not aec_hydrogen_temp_df.empty:
            plot_categorical_histogram(
                aec_hydrogen_temp_df,
                category_col="Temperature Range" if "Temperature Range" in aec_hydrogen_temp_df.columns else "category",
                value_col="Hours of operation (hr)",
                custom_order=TEMPERATURE_LABELS,
                title="Hydrogen Flash Temperature",
                mean_value=grand_mean_aec_hydrogen_temp,
                mean_label="<b>Average H₂ Temp: {val:.2f} °C</b>",
                filename="aec_hydrogen_flash_temperature_histogram.csv",
                yaxis_title="Hours of operation (hr)"
            )
        else:
            st.info("No hydrogen flash temperature histogram data available.")

    st.markdown("---")

    # --- Row 6: Oxygen Pressure
    if not aec_oxygen_pressure_df.empty:
        plot_categorical_histogram(
            aec_oxygen_pressure_df,
            category_col="Pressure Range" if "Pressure Range" in aec_oxygen_pressure_df.columns else "category",
            value_col="Hours of operation (hr)",
            custom_order=PRESSURE_LABELS,
            title="Oxygen Pressure",
            mean_value=grand_mean_aec_oxygen_pressure,
            mean_label="<b>Average O₂ Pressure: {val:.2f} bar</b>",
            filename="aec_oxygen_pressure_histogram.csv",
            yaxis_title="Hours of operation (hr)"
        )
    else:
        st.info("No oxygen pressure histogram data available.")


elif selected_subsystem == "Methanol Synthesis Reactor":
    st.markdown("### 🛢️ Methanol Synthesis Reactor Dashboard")
    st.info("This section is under construction. Stay tuned!")

elif selected_subsystem == "Methanol Distillation Column":
    st.markdown("### 🏭 Methanol Distillation Column")

    # ---------------------------------------------------------------------
    # 1) Equipment runtime summary for heaters
    # ---------------------------------------------------------------------

    if not runtime_df.empty:
        ds_df = runtime_df[runtime_df["Subsystem"] == "DS"]

        if ds_df.empty:
            st.info("No Methanol Distillation runtime data available.")
            st.stop()

        summary = (
            ds_df.groupby("Component", as_index=False)["Hours of operation (hr)"]
                .sum()
                .sort_values("Hours of operation (hr)", ascending=False)
        )

        # Ordering for nice display
        ds_order = [
            "Heater 1",
            "Heater 2",
            "Heater 3",
            "Heater 4",
            "Heater 5",
            "Heater 6",
            "Distillation Column",
            "Feed Pump",
        ]

        comp_hours = summary.set_index("Component")["Hours of operation (hr)"].to_dict()
        ordered = {k: comp_hours.get(k, 0) for k in ds_order}

        total_active_time_ms = 0.0
        if not active_state_df.empty and "Active Time MS (hr)" in active_state_df.columns:
            total_active_time_ms = pd.to_numeric(active_state_df["Active Time MS (hr)"], errors="coerce").sum()
        ordered["Active State (MS)"] = total_active_time_ms

        # Display metrics
        comps = list(ordered.keys())
        vals = list(ordered.values())

        cols_per_row = 4
        for i in range(0, len(comps), cols_per_row):
            row_comps = comps[i:i+cols_per_row]
            row_vals = vals[i:i+cols_per_row]
            cols = st.columns(len(row_comps))
            for j, comp in enumerate(row_comps):
                with cols[j]:
                    st.markdown(
                    f"""
                    <div style="
                        background: #f8f9fa;
                        border-radius: 12px;
                        padding: 15px 18px;
                        text-align: center;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                        border: 1px solid #e6e6e6;
                    ">
                        <div style="font-size: 16px; font-weight: 600;">{comp}</div>
                        <div style="font-size: 26px; margin-top: 4px; color:#2a2a2a;">
                            {row_vals[j]:.2f} hr
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # Pie chart
        df_plot = pd.DataFrame({"Component": comps, "Hours": vals})
        fig = px.pie(df_plot, names="Component", values="Hours", hole=0.5)
        st.plotly_chart(fig, use_container_width=True)



    st.markdown("---")

else:
    st.warning("Please select a subsystem above to begin.")
