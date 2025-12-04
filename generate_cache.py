import os
from pathlib import Path
import pandas as pd
import numpy as np


# Import all processing functions from Hours_V31.py
from Hours_V31 import (
    find_parquet_files, extract_db_file_name, compute_db_file_duration, compute_db_file_duration_raw,
    compute_all_runtimes, compute_reboiler_temperature_for_file, compute_sorbent_temperature_for_file,
    compute_big_fan_pwm_for_file, compute_recycle_pump_1_rpm_for_file, compute_recycle_pump_2_rpm_for_file,
    compute_weather_classification_for_file, compute_mean, compute_mean_residence,
    compute_compressor_first_stage_pressure_for_file, compute_compressor_second_stage_pressure_for_file,
    compute_compressor_first_stage_temperature_for_file, compute_compressor_second_stage_temperature_for_file,
    compute_aec_stack_1_current_for_file, compute_aec_stack_2_current_for_file, compute_aec_stack_3_current_for_file,
    compute_aec_stack_4_current_for_file, compute_aec_stack_1_current_density_for_file, compute_aec_stack_2_current_density_for_file,
    compute_aec_stack_3_current_density_for_file, compute_aec_stack_4_current_density_for_file,
    compute_aec_oxygen_temperature_for_file, compute_aec_hydrogen_temperature_for_file,
    compute_aec_oxygen_pressure_for_file, compute_active_state_for_file
)


# UNIVERSAL PATH LOGIC — works everywhere
REMOTE_PATH = "ZEF_Container_Dashboard:Parquet_Exports"

env_val = os.getenv("PARENT_RUNS_FOLDER")
if env_val:
    # If it's a remote path (contains ':'), use local Parquet_Exports for file ops
    if ":" in env_val:
        print(f"Remote data folder detected: {env_val}. Using local Parquet_Exports for cache generation.")
        PARENT_RUNS_FOLDER = Path("Parquet_Exports")
        CACHE_DIR = PARENT_RUNS_FOLDER / "cache"
    else:
        PARENT_RUNS_FOLDER = Path(env_val)
        CACHE_DIR = PARENT_RUNS_FOLDER / "cache"
else:
    PARENT_RUNS_FOLDER = Path("Parquet_Exports")
    CACHE_DIR = PARENT_RUNS_FOLDER / "cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data folder: {PARENT_RUNS_FOLDER}")
print(f"Cache folder: {CACHE_DIR}")


# Merge new results with existing cache and save
def save_cache(name: str, df: pd.DataFrame):
    cache_path = CACHE_DIR / f"{name}.parquet"
    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        # Merge, avoiding duplicates (by 'db File' if present)
        if not df.empty:
            if 'db File' in df.columns and 'db File' in existing.columns:
                merged = pd.concat([existing, df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["db File", "Component"], keep="last") if "Component" in merged.columns else merged.drop_duplicates(subset=["db File"], keep="last")
                merged.to_parquet(cache_path, index=False)
            else:
                merged = pd.concat([existing, df], ignore_index=True).drop_duplicates(keep="last")
                merged.to_parquet(cache_path, index=False)
        else:
            existing.to_parquet(cache_path, index=False)
    else:
        df.to_parquet(cache_path, index=False)


# Load existing cache to determine already processed db files
def load_cache(name: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

existing_runtime = load_cache("runtime")
existing_db_files = set(existing_runtime["db File"].dropna().astype(str).unique()) if not existing_runtime.empty else set()

# Find all command/message pairs

# Print all data folders and cache status
command_files = find_parquet_files(PARENT_RUNS_FOLDER, "commands.parquet")
all_db_files = [extract_db_file_name(cmd) for cmd in command_files]
to_process = [cmd for cmd in command_files if extract_db_file_name(cmd) not in existing_db_files]
print(f"\nFound {len(command_files)} data folders.")
print(f"Already cached: {len(existing_db_files)}")
print(f"To process: {len(to_process)}\n")
if len(to_process) > 0:
    print("Files to process:")
    for i, cmd in enumerate(to_process, 1):
        print(f"  {i}. {extract_db_file_name(cmd)}")
else:
    print("All files are already cached.")

paired_files = []
for cmd in to_process:
    msg = cmd.parent / "messages.parquet"
    db_file = extract_db_file_name(cmd)
    if not msg.exists():
        print(f"[ERROR] Missing messages.parquet for {db_file}. Skipping.")
        continue
    paired_files.append((cmd, msg))

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


# For headless operation, pass empty DataFrames for durations_df where required


total_files = len(paired_files)
for idx, (commands_file, messages_file) in enumerate(paired_files, 1):
    db_file = extract_db_file_name(commands_file)
    print(f"\n[{idx}/{total_files}] Processing db file: {db_file}")
    try:
        df_db_file_duration = compute_db_file_duration(messages_file)
        df_run = compute_all_runtimes(messages_file, pd.DataFrame())
        df_reb = compute_reboiler_temperature_for_file(messages_file, pd.DataFrame())
        df_sorb = compute_sorbent_temperature_for_file(messages_file, pd.DataFrame())
        df_pwm = compute_big_fan_pwm_for_file(messages_file)
        df_rpm = compute_recycle_pump_1_rpm_for_file(messages_file, pd.DataFrame())
        df_rpm2 = compute_recycle_pump_2_rpm_for_file(messages_file, pd.DataFrame())
        df_weather = compute_weather_classification_for_file(messages_file)
        df_mean = compute_mean(messages_file)
        df_mean_residence = compute_mean_residence(commands_file, messages_file)
        df_comp1 = compute_compressor_first_stage_pressure_for_file(messages_file, pd.DataFrame())
        df_comp2 = compute_compressor_second_stage_pressure_for_file(messages_file, pd.DataFrame())
        df_comp1_temp = compute_compressor_first_stage_temperature_for_file(messages_file, pd.DataFrame())
        df_comp2_temp = compute_compressor_second_stage_temperature_for_file(messages_file, pd.DataFrame())
        df_aec1 = compute_aec_stack_1_current_for_file(messages_file, pd.DataFrame())
        df_aec2 = compute_aec_stack_2_current_for_file(messages_file, pd.DataFrame())
        df_aec3 = compute_aec_stack_3_current_for_file(messages_file, pd.DataFrame())
        df_aec4 = compute_aec_stack_4_current_for_file(messages_file, pd.DataFrame())
        df_aec1_den = compute_aec_stack_1_current_density_for_file(messages_file, pd.DataFrame())
        df_aec2_den = compute_aec_stack_2_current_density_for_file(messages_file, pd.DataFrame())
        df_aec3_den = compute_aec_stack_3_current_density_for_file(messages_file, pd.DataFrame())
        df_aec4_den = compute_aec_stack_4_current_density_for_file(messages_file, pd.DataFrame())
        df_aec_oxy_temp = compute_aec_oxygen_temperature_for_file(messages_file, pd.DataFrame())
        df_aec_hyd_temp = compute_aec_hydrogen_temperature_for_file(messages_file, pd.DataFrame())
        df_aec_oxy_pres = compute_aec_oxygen_pressure_for_file(messages_file, pd.DataFrame())
        df_active = compute_active_state_for_file(messages_file, pd.DataFrame())

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
    except Exception as e:
        print(f"[ERROR] Failed to process {db_file}: {e}")
        continue

# Concatenate and save all cache files

def concat_and_save(name, dfs):
    df = pd.concat([df for df in dfs if not df.empty and not df.isna().all().all()], ignore_index=True) if dfs else pd.DataFrame()
    print(f"Updating cache file: {name}.parquet, rows to add: {len(df)}")
    save_cache(name, df)

concat_and_save("db_file_duration", all_db_file_duration)
concat_and_save("runtime", all_runtime)
concat_and_save("reboiler_temperature", all_reboiler)
concat_and_save("sorbent_temperature", all_sorb)
concat_and_save("big_fan_pwm", all_bigfan_pwm)
concat_and_save("recycle_pump_1_rpm", all_recycle_pump_1_rpm)
concat_and_save("recycle_pump_2_rpm", all_recycle_pump_2_rpm)
concat_and_save("weather_classification", all_weather)
concat_and_save("mean_values", all_mean)
concat_and_save("mean_residence_time", all_mean_residence)
concat_and_save("compressor_first_stage_pressure", all_compressor_stage_1)
concat_and_save("compressor_second_stage_pressure", all_compressor_stage_2)
concat_and_save("compressor_first_stage_temperature", all_compressor_stage_1_temp)
concat_and_save("compressor_second_stage_temperature", all_compressor_stage_2_temp)
concat_and_save("aec_stack_1_current", all_aec_stack_1_current)
concat_and_save("aec_stack_2_current", all_aec_stack_2_current)
concat_and_save("aec_stack_3_current", all_aec_stack_3_current)
concat_and_save("aec_stack_4_current", all_aec_stack_4_current)
concat_and_save("aec_stack_1_current_density", all_aec_stack_1_current_density)
concat_and_save("aec_stack_2_current_density", all_aec_stack_2_current_density)
concat_and_save("aec_stack_3_current_density", all_aec_stack_3_current_density)
concat_and_save("aec_stack_4_current_density", all_aec_stack_4_current_density)
concat_and_save("aec_oxygen_flash_temperature", all_aec_oxygen_temp)
concat_and_save("aec_hydrogen_flash_temperature", all_aec_hydrogen_temp)
concat_and_save("aec_oxygen_pressure", all_aec_oxygen_pressure)
concat_and_save("active_state", all_active_state)

print("Cache generation complete.")