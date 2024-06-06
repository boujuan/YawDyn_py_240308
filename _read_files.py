import os
import pandas as pd
from _stepresponse_analysis import plot_identified_yaw_maneuvers

# INFO: Import raw data
def import_data_from_asc(file_path, sensor_name):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    data = [x.split(b';') for x in raw_data.split(b'\r\n')]
    data = pd.DataFrame([[x.decode(), y.decode()] for x, y in data[2:-1]], columns=['time', sensor_name])
    data = data.loc[~data['time'].duplicated()]  # Remove duplicate timestamps
    data.set_index(pd.to_datetime(data['time'], format='%d.%m.%y %H:%M:%S'), inplace=True)
    data.drop(columns=['time'], inplace=True)
    data[sensor_name] = data[sensor_name].str.strip().astype('float')
    data_resampled = data.resample('1s').mean()
    print(f"Data for {sensor_name} loaded and resampled. Shape: {data_resampled.shape}")
    return data_resampled

def load_yaw_data(data_folder):
    file_names = [
        "VA_YawPositionModulus_N3_3.ASC",
        "VA_YawPositionModulus_N3_4.ASC",
        "VA_YawPositionModulus_N3_5.ASC",
        "VA_YawPositionModulus_N3_6.ASC",
    ]

    dfs = []
    for file_name in file_names:
        file_path = os.path.join(data_folder, file_name)
        try:
            df = import_data_from_asc(file_path, file_name.split("_")[-1][:-4])
            dfs.append(df)
            print(f"Data from {file_name} added to DataFrame list.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")

    if dfs:
        combined_df = pd.concat(dfs, axis=1)
        combined_df = combined_df.ffill().asfreq("s")
        print("Combined DataFrame created. Shape:", combined_df.shape)
        return combined_df
    else:
        print("No data files were loaded.")
        return None

def load_wind_speed_data(data_folder):
    file_names = [
        "VA_WindSpeed_Avg10s_N3_3.ASC",
        "VA_WindSpeed_Avg10s_N3_4.ASC",
        "VA_WindSpeed_Avg10s_N3_5.ASC",
        "VA_WindSpeed_Avg10s_N3_6.ASC",
    ]

    dfs = []
    for file_name in file_names:
        file_path = os.path.join(data_folder, file_name)
        try:
            df = import_data_from_asc(file_path, file_name.split("_")[-2] + "_" + file_name.split("_")[-1][:-4])
            dfs.append(df)
            print(f"Data from {file_name} added to DataFrame list.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")

    if dfs:
        combined_df = pd.concat(dfs, axis=1)
        combined_df = combined_df.ffill().asfreq("s")
        print("Combined DataFrame created. Shape:", combined_df.shape)
        return combined_df
    else:
        print("No data files were loaded.")
        return None

# Load the yaw data
data_folder = "Data/raw/2023-06-01_2023-07-31"
yaw_data = load_yaw_data(data_folder)
# Load the wind speed data
wind_speed_data = load_wind_speed_data(data_folder)

print(yaw_data.tail())
print(wind_speed_data.tail())

if yaw_data is not None and wind_speed_data is not None:
    path2dir_fig_base = 'Figures/identified_yaw_maneuvers'
    date_range_total_str = '2023-06-01_2023-07-31'
    resample_str = '1s'

    plot_identified_yaw_maneuvers(
        yaw_data,
        wind_speed_data,
        path2dir_fig_base,
        date_range_total_str,
        resample_str
    )