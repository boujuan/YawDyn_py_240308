import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plot_tools import make_dir_if_not_exists
#import datetime
import numpy as np
#import os

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# INFO: TASK 3:

def plot_identified_yaw_maneuvers(yaw_data, wind_speed_data, path2dir_fig_base, date_range_total_str, resample_str):
    path2dir_yaw_maneuvers = f"{path2dir_fig_base}/identified_yaw_maneuvers/{date_range_total_str}_{resample_str}"
    make_dir_if_not_exists(path2dir_yaw_maneuvers)

    for turb_key in yaw_data.keys():
        print(turb_key)
        yaw_filt = yaw_data[turb_key]
        wind_speed_filt = wind_speed_data[turb_key]
        
        # Correctly unpack all returned values from find_yaw_maneuver
        yaw_man, yaw_length, yaw_duration, cw_pairs, ccw_pairs = find_yaw_maneuver(yaw_filt)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=yaw_filt.index, y=yaw_filt, mode='lines', name='Yaw Angle'), row=1, col=1)
        fig.add_trace(go.Scatter(x=wind_speed_filt.index, y=wind_speed_filt, mode='lines', name='Wind Speed'), row=2, col=1)

        # Add markers for the start and stop of each maneuver
        starts = yaw_man[yaw_man == 'cw_start'].index.tolist() + yaw_man[yaw_man == 'ccw_start'].index.tolist()
        stops = yaw_man[yaw_man == 'cw_stop'].index.tolist() + yaw_man[yaw_man == 'ccw_stop'].index.tolist()

        fig.add_trace(go.Scatter(x=starts, y=[yaw_filt.loc[idx] for idx in starts], mode='markers', marker=dict(color='green', size=10), name='Start'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stops, y=[yaw_filt.loc[idx] for idx in stops], mode='markers', marker=dict(color='red', size=10), name='Stop'), row=1, col=1)

        fig.update_layout(title=f"Yaw Maneuvers and Wind Speed for Turbine {turb_key}", xaxis_title='Time', yaxis1_title='Yaw Angle (degrees)', yaxis2_title='Wind Speed (m/s)')
        fig.write_html(f"{path2dir_yaw_maneuvers}/{turb_key}_yaw_maneuvers_interactive.html")
        
# INFO: Algorithm from Andreas
def find_yaw_maneuver(yaw_filt):
    yaw_diff = yaw_filt.diff()

    yaw_diff_sum_big = yaw_diff.rolling("20s", center=True).sum()
    yaw_diff_sum_small = yaw_diff.rolling("3s", center=False).sum()

    yaw_man_cw = (yaw_diff_sum_big >= 3) & (yaw_diff_sum_small >= 1)
    yaw_man_ccw = (yaw_diff_sum_big <= -3) & (yaw_diff_sum_small <= -1)

    cw_start = yaw_man_cw.index[yaw_man_cw & ~yaw_man_cw.shift(1).ffill().astype(bool)]
    cw_stop = yaw_man_cw.index[~yaw_man_cw & yaw_man_cw.shift(1).ffill().astype(bool)]
    ccw_start = yaw_man_ccw.index[yaw_man_ccw & ~yaw_man_ccw.shift(1).ffill().astype(bool)]
    ccw_stop = yaw_man_ccw.index[~yaw_man_ccw & yaw_man_ccw.shift(1).ffill().astype(bool)]

    # Ensure starts and stops are sorted (if not already)
    cw_start = np.sort(cw_start)
    cw_stop = np.sort(cw_stop)
    ccw_start = np.sort(ccw_start)
    ccw_stop = np.sort(ccw_stop)

    # Initialize the series
    yaw_man = pd.Series(np.nan, index=yaw_filt.index, dtype=object)
    yaw_length = pd.Series(np.nan, index=yaw_filt.index)
    yaw_duration = pd.Series(np.nan, index=yaw_filt.index)

    def find_next_stop(starts, stops):
        pairs = {}
        used_stops = set()
        stop_iter = iter(stops)
        try:
            next_stop = next(stop_iter)
            for start in starts:
                while next_stop <= start:
                    next_stop = next(stop_iter)
                if next_stop > start and yaw_filt[next_stop] != yaw_filt[start]:
                    pairs[start] = next_stop
                    used_stops.add(next_stop)
                    next_stop = next(stop_iter)
        except StopIteration:
            pass
        return pairs

    cw_pairs = find_next_stop(cw_start, cw_stop)
    ccw_pairs = find_next_stop(ccw_start, ccw_stop)

    for start, stop in cw_pairs.items():
        yaw_man[start] = 'cw_start'
        yaw_man[stop] = 'cw_stop'
        yaw_length[start] = yaw_filt.loc[stop] - yaw_filt.loc[start]
        yaw_duration[start] = pd.Timedelta(stop - start).total_seconds()

    for start, stop in ccw_pairs.items():
        yaw_man[start] = 'ccw_start'
        yaw_man[stop] = 'ccw_stop'
        yaw_length[start] = yaw_filt.loc[stop] - yaw_filt.loc[start]
        yaw_duration[start] = pd.Timedelta(stop - start).total_seconds()

    return yaw_man, yaw_length, yaw_duration, cw_pairs, ccw_pairs

def print_maneuver_table(yaw_data):
    for turb_key in yaw_data.keys():
        print(f'Maneuvers for Turbine {turb_key}:')
        data = yaw_data[turb_key]
        yaw_man, yaw_length, yaw_duration, cw_pairs, ccw_pairs = find_yaw_maneuver(data)

        maneuvers = []

        for start, stop in cw_pairs.items():
            yaw_change = data.loc[stop] - data.loc[start]
            if yaw_change != 0:  # Check if there is an actual yaw angle change
                new_row = pd.DataFrame({
                    'Maneuver Type': ['CW'],
                    'Start Time': [start],
                    'End Time': [stop],
                    'Duration (s)': [pd.Timedelta(stop - start).total_seconds()],
                    'Yaw Change (degrees)': [yaw_change]
                })
                if not new_row.dropna().empty:  # Check if the new row is not entirely NA and not empty
                    maneuvers.append(new_row)

        for start, stop in ccw_pairs.items():
            yaw_change = data.loc[stop] - data.loc[start]
            if yaw_change != 0:  # Check if there is an actual yaw angle change
                new_row = pd.DataFrame({
                    'Maneuver Type': ['CCW'],
                    'Start Time': [start],
                    'End Time': [stop],
                    'Duration (s)': [pd.Timedelta(stop - start).total_seconds()],
                    'Yaw Change (degrees)': [yaw_change]
                })
                if not new_row.dropna().empty:  # Check if the new row is not entirely NA and not empty
                    maneuvers.append(new_row)

        if maneuvers:
            maneuvers_df = pd.concat(maneuvers, ignore_index=True)
            maneuvers_df.dropna(inplace=True)
            print(maneuvers_df)
        else:
            print("No valid maneuvers found.")

# def load_yaw_data():
#     data_folders = [
#         "Data/raw/2023-06-01_2023-07-31",
#         "Data/raw/2023-09-01_2023-11-19",
#         "Data/raw/2023-11-20_2024-01-31",
#     ]
#     file_names = [
#         "VA_YawPositionModulus_N3_3.ASC",
#         "VA_YawPositionModulus_N3_4.ASC",
#         "VA_YawPositionModulus_N3_5.ASC",
#         "VA_YawPositionModulus_N3_6.ASC",
#     ]

#     dfs = []
#     for folder in data_folders:
#         for file_name in file_names:
#             file_path = os.path.join(folder, file_name)
#             try:
#                 df = pd.read_csv(
#                     file_path,
#                     sep=";",
#                     header=None,
#                     names=["timestamp", file_name.split("_")[-1][:-4]],
#                     parse_dates=["timestamp"],
#                     date_parser=lambda x: pd.datetime.strptime(x, "%d.%m.%y %H:%M:%S"),
#                 )
#                 dfs.append(df)
#             except FileNotFoundError:
#                 print(f"File not found: {file_path}")

#     if dfs:
#         combined_df = pd.concat(dfs, axis=1)
#         combined_df = combined_df.ffill().asfreq("S")
#         return combined_df
#     else:
#         return None

# def main():
#     yaw_data = load_yaw_data()
#     turb_keys_to_process = ['N3_3', 'N3_4', 'N3_5', 'N3_6']
#     path2dir_fig_base = 'Figures/identified_yaw_maneuvers'
#     date_range_total_str = '2023-06-01_2024-01-31'
#     resample_str = '1s'

#     for ctrl_key in df_filt_yaw_dict.keys():
#         for turb_key in turb_keys_to_process:
#             if turb_key in yaw_data.columns:
#                 yaw_filt = yaw_data[turb_key]
#                 yaw_man, yaw_length, yaw_duration = find_yaw_maneuver(yaw_filt)

#                 plot_identified_yaw_maneuvers(
#                     {ctrl_key: {turb_key: yaw_filt}},
#                     [turb_key],
#                     path2dir_fig_base,
#                     date_range_total_str,
#                     resample_str
#                 )

# if __name__ == "__main__":
#     main()




