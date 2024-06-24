# %%
import os

import pandas as pd
from _stepresponse_analysis_original import analyze_yaw_maneuvers

os.chdir('..') # Temporal fix to go back one folder for the interactive cell run

# INFO: Import raw data
def import_data_from_asc(file_path, sensor_name):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    data = [x.split(b';') for x in raw_data.split(b'\r\n')]
    data = pd.DataFrame([[x.decode(), y.decode()] for x, y in data[2:-1]], columns=['time', sensor_name])
    data[sensor_name] = data[sensor_name].astype(float)
    data = data.groupby('time').mean().reset_index()
    data.set_index(pd.to_datetime(data['time'], format='%d.%m.%y %H:%M:%S'), inplace=True)
    data.drop(columns=['time'], inplace=True)
    data[sensor_name] = data[sensor_name].astype(float)
    data_resampled = data.resample('1s').ffill()
    print(f"Data for {sensor_name} loaded and resampled. Shape: {data_resampled.shape}")
    return data_resampled

def filter_and_fill_data(df, power_range=(400, 4900), pitch_range=(0, 2), operation_state_value=6, controller_state_value=1,
                         apply_power_filter=1, apply_pitch_filter=1, apply_operation_state_filter=1, apply_controller_state_filter=1):
    # Initialize the filter with all True values (no filtering)
    filter_mask = pd.Series(True, index=df.index)
    
    # Conditionally apply each filter
    if apply_power_filter:
        power_filter = df.filter(like='power_').apply(lambda x: x.between(*power_range), axis=0).all(axis=1)
        filter_mask &= power_filter
    if apply_pitch_filter:
        pitch_filter = df.filter(like='pitch_').apply(lambda x: x.between(*pitch_range), axis=0).all(axis=1)
        filter_mask &= pitch_filter
    if apply_operation_state_filter:
        operation_state_filter = (df.filter(like='operation_state_') == operation_state_value).all(axis=1)
        filter_mask &= operation_state_filter
    if apply_controller_state_filter:
        controller_state_filter = (df.filter(like='controller_state_') == controller_state_value).all(axis=1)
        filter_mask &= controller_state_filter
    
    # Apply the combined filter mask to the DataFrame
    filtered_df = df[filter_mask]
    
    # Ensure the original index is preserved
    filled_df = filtered_df.reindex(df.index)
    
    return filled_df

def load_data(data_folder):
    yaw_file_names = ["VA_YawPositionModulus_N3_4.ASC"]
    wind_file_names = ["VA_WindSpeed_Avg10s_N3_4.ASC"]
    wind_dir_file_names = ["VA_WindDirectionModulus_Avg10s_N3_4.ASC"]
    power_file_names = ["VA_WindTurbineActivePowerOutput_Avg1s_N3_4.ASC"]
    pitch_file_names = ["VA_PitchMinimumPositionOfAllBlades_N3_4.ASC"]
    operation_state_file_names = ["VA_OperationState_N3_4.ASC"]
    controller_state_file_names = ["VA_YawWindTrailingFunctionActive_N3_4.ASC"]

    yaw_dfs, wind_dfs, wind_dir_dfs, power_dfs, pitch_dfs, operation_state_dfs, controller_state_dfs = [], [], [], [], [], [], []
    
    for yaw_file, wind_file, wind_dir_file, power_file, pitch_file, operation_state_file in zip(yaw_file_names, wind_file_names, wind_dir_file_names, power_file_names, pitch_file_names, operation_state_file_names):
        yaw_path = os.path.join(data_folder, yaw_file)
        wind_path = os.path.join(data_folder, wind_file)
        wind_dir_path = os.path.join(data_folder, wind_dir_file)
        power_path = os.path.join(data_folder, power_file)
        pitch_path = os.path.join(data_folder, pitch_file)
        operation_state_path = os.path.join(data_folder, operation_state_file)
        
        if all(os.path.exists(path) for path in [yaw_path, wind_path, wind_dir_path, power_path, pitch_path, operation_state_path]):
            yaw_df = import_data_from_asc(yaw_path, 'yaw_' + yaw_file.split("_")[-1][:-4]).resample('1s').ffill()
            wind_df = import_data_from_asc(wind_path, 'windspeed_' + wind_file.split("_")[-1][:-4]).resample('1s').ffill()
            wind_dir_df = import_data_from_asc(wind_dir_path, 'winddir_' + wind_dir_file.split("_")[-1][:-4]).resample('1s').ffill()
            power_df = import_data_from_asc(power_path, 'power_' + power_file.split("_")[-1][:-4]).resample('1s').ffill()
            pitch_df = import_data_from_asc(pitch_path, 'pitch_' + pitch_file.split("_")[-1][:-4]).resample('1s').ffill()
            operation_state_df = import_data_from_asc(operation_state_path, 'operation_state_' + operation_state_file.split("_")[-1][:-4]).resample('1s').ffill()
            yaw_dfs.append(yaw_df)
            wind_dfs.append(wind_df)
            wind_dir_dfs.append(wind_dir_df)
            power_dfs.append(power_df)
            pitch_dfs.append(pitch_df)
            operation_state_dfs.append(operation_state_df)
            print(f"Data from {yaw_file}, {wind_file}, {wind_dir_file}, {power_file}, {pitch_file}, and {operation_state_file} added to DataFrame list.")
        else:
            print(f"Some files not found in folder: {data_folder}")
        
    for controller_state_file in controller_state_file_names:
        controller_state_path = os.path.join(data_folder, controller_state_file)
        if os.path.exists(controller_state_path):
            controller_state_df = import_data_from_asc(controller_state_path, 'controller_state_' + controller_state_file.split("_")[-1][:-4]).resample('1s').ffill()
            controller_state_dfs.append(controller_state_df)
        else:
            print(f"Controller state file not found: {controller_state_path}")

    if yaw_dfs and wind_dfs and wind_dir_dfs and power_dfs and pitch_dfs and operation_state_dfs and controller_state_dfs:
        combined_yaw_df = pd.concat(yaw_dfs, axis=1)
        combined_wind_df = pd.concat(wind_dfs, axis=1)
        combined_wind_dir_df = pd.concat(wind_dir_dfs, axis=1)
        combined_power_df = pd.concat(power_dfs, axis=1)
        combined_pitch_df = pd.concat(pitch_dfs, axis=1)
        combined_operation_state_df = pd.concat(operation_state_dfs, axis=1)
        combined_controller_state_df = pd.concat(controller_state_dfs, axis=1)
        combined_df = pd.concat([combined_yaw_df, combined_wind_df, combined_wind_dir_df, combined_power_df, combined_pitch_df, combined_operation_state_df, combined_controller_state_df], axis=1)
        combined_df = combined_df.ffill().asfreq("s")
        print(f"Combined DataFrame created for folder {data_folder}. Shape:", combined_df.shape)
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        return combined_df
    else:
        print(f"No data files were loaded from folder: {data_folder}")
        return None

    yaw_dfs, wind_dfs, wind_dir_dfs, power_dfs, pitch_dfs, operation_state_dfs, controller_state_dfs = [], [], [], [], [], [], []
    for yaw_file, wind_file, wind_dir_file, power_file, pitch_file, operation_state_file in zip(yaw_file_names, wind_file_names, wind_dir_file_names, power_file_names, pitch_file_names, operation_state_file_names):
        yaw_path = os.path.join(data_folder, yaw_file)
        wind_path = os.path.join(data_folder, wind_file)
        wind_dir_path = os.path.join(data_folder, wind_dir_file)
        power_path = os.path.join(data_folder, power_file)
        pitch_path = os.path.join(data_folder, pitch_file)
        operation_state_path = os.path.join(data_folder, operation_state_file)
        
        yaw_df = import_data_from_asc(yaw_path, 'yaw_' + yaw_file.split("_")[-1][:-4]).resample('1s').ffill()
        wind_df = import_data_from_asc(wind_path, 'windspeed_' + wind_file.split("_")[-1][:-4]).resample('1s').ffill()
        wind_dir_df = import_data_from_asc(wind_dir_path, 'winddir_' + wind_dir_file.split("_")[-1][:-4]).resample('1s').ffill()
        power_df = import_data_from_asc(power_path, 'power_' + power_file.split("_")[-1][:-4]).resample('1s').ffill()
        pitch_df = import_data_from_asc(pitch_path, 'pitch_' + pitch_file.split("_")[-1][:-4]).resample('1s').ffill()
        operation_state_df = import_data_from_asc(operation_state_path, 'operation_state_' + operation_state_file.split("_")[-1][:-4]).resample('1s').ffill()
        yaw_dfs.append(yaw_df)
        wind_dfs.append(wind_df)
        wind_dir_dfs.append(wind_dir_df)
        power_dfs.append(power_df)
        pitch_dfs.append(pitch_df)
        operation_state_dfs.append(operation_state_df)
        print(f"Data from {yaw_file}, {wind_file}, {wind_dir_file}, {power_file}, {pitch_file}, and {operation_state_file} added to DataFrame list.")
        
    for controller_state_file in controller_state_file_names:
        controller_state_path = os.path.join(data_folder, controller_state_file)
        if os.path.exists(controller_state_path):
            controller_state_df = import_data_from_asc(controller_state_path, 'controller_state_' + controller_state_file.split("_")[-1][:-4]).resample('1s').ffill()
            controller_state_dfs.append(controller_state_df)
        else:
            print(f"Controller state file not found: {controller_state_path}")

    if yaw_dfs and wind_dfs and wind_dir_dfs and power_dfs and pitch_dfs and operation_state_dfs and controller_state_dfs:
        combined_yaw_df = pd.concat(yaw_dfs, axis=1)
        combined_wind_df = pd.concat(wind_dfs, axis=1)
        combined_wind_dir_df = pd.concat(wind_dir_dfs, axis=1)
        combined_power_df = pd.concat(power_dfs, axis=1)
        combined_pitch_df = pd.concat(pitch_dfs, axis=1)
        combined_operation_state_df = pd.concat(operation_state_dfs, axis=1)
        combined_controller_state_df = pd.concat(controller_state_dfs, axis=1)
        combined_df = pd.concat([combined_yaw_df, combined_wind_df, combined_wind_dir_df, combined_power_df, combined_pitch_df, combined_operation_state_df, combined_controller_state_df], axis=1)
        combined_df = combined_df.ffill().asfreq("s")
        print("Combined DataFrame created. Shape:", combined_df.shape)
        return combined_df
    else:
        print("No data files were loaded.")
        return None

# Load the data
data_folders = [
    "Data\\raw\\2023-06-01_2023-07-31",
    "Data\\raw\\2023-09-01_2023-11-19",
    "Data\\raw\\2023-11-20_2024-01-31"
]
combined_data = pd.DataFrame()
for folder in data_folders:
    folder_data = load_data(folder)
    if folder_data is not None:
        if combined_data.empty:
            combined_data = folder_data
        else:
            # Ensure no overlap by only appending data after the last timestamp in combined_data
            last_timestamp = combined_data.index.max()
            new_data = folder_data[folder_data.index > last_timestamp]
            combined_data = pd.concat([combined_data, new_data])

combined_data = combined_data.sort_index().ffill()
print(f"Final combined data shape: {combined_data.shape}")
print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
# combined_data.to_csv('combined_data.csv', index=True)

# Correct windspeed for all timestamps after 2023-12-01
correction_date = pd.Timestamp('2023-12-01')
windspeed_columns = [col for col in combined_data.columns if 'windspeed_' in col]
mask = combined_data.index >= correction_date
for col in windspeed_columns:
    combined_data.loc[mask, col] = combined_data.loc[mask, col] * ((0.8274/0.9006) + 0.5645)
print("Windspeed correction applied for data after 2023-12-01")

filtered_data = filter_and_fill_data(combined_data, power_range=(400, 4900), pitch_range=(0, 2), operation_state_value=6, controller_state_value=1)
# filtered_data.to_csv('filtered_data.csv', index=True)

print("head: ", filtered_data.head())
print("tail: ", filtered_data.tail())
print("columns: ", filtered_data.columns)
print("shape: ", filtered_data.shape)
print("sample: ", filtered_data.sample(10))
print("----------------------------------")
# Print a sample of 10 random non-NaN power values from the filtered data
power_columns = [col for col in filtered_data.columns if 'power_' in col]
non_nan_power_data = filtered_data[power_columns].dropna()
sampled_power_data = non_nan_power_data.sample(10)
print("Sample of 10 random non-NaN power values:")
print(sampled_power_data)

print("="*20)

if filtered_data is not None:
    path2dir_fig_base = 'Figures/identified_yaw_maneuvers'
    date_range_total_str = '2023-06-01_2024-01-31'
    resample_str = '1s'

    results = analyze_yaw_maneuvers(
        filtered_data[['yaw_4']],
        filtered_data[['windspeed_4']],
        path2dir_fig_base,
        date_range_total_str,
        resample_str
    )
    results_mean = results['Wind Speed Offset'].mean()
    results_sem = results['Wind Speed Offset'].sem()
    print("Mean: ", results_mean)
    print("SEM: ", results_sem)
# %%