import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plot_tools import make_dir_if_not_exists
import datetime
import numpy as np
import os

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


# INFO: TASK 3:

def plot_identified_yaw_maneuvers(yaw_data, wind_speed_data, path2dir_fig_base, date_range_total_str, resample_str):
    path2dir_yaw_maneuvers = f"{path2dir_fig_base}/identified_yaw_maneuvers/{date_range_total_str}_{resample_str}"
    make_dir_if_not_exists(path2dir_yaw_maneuvers)

    for turb_key in yaw_data.columns:
        print(turb_key)
        yaw_filt = yaw_data[turb_key]
        wind_speed_filt = wind_speed_data[f'windspeed_{turb_key.split("_")[-1]}']
        yaw_man, yaw_length, yaw_duration = find_yaw_maneuver(yaw_filt)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=yaw_filt.index, y=yaw_filt, mode='lines', name='Yaw Angle'), row=1, col=1)
        fig.add_trace(go.Scatter(x=wind_speed_filt.index, y=wind_speed_filt, mode='lines', name='Wind Speed'), row=2, col=1)

        # Add markers for the start and stop of each maneuver
        starts = yaw_man[yaw_man == 'cw_start'].index.union(yaw_man[yaw_man == 'ccw_start'].index)
        stops = yaw_man[yaw_man == 'cw_stop'].index.union(yaw_man[yaw_man == 'ccw_stop'].index)

        fig.add_trace(go.Scatter(x=starts, y=yaw_filt.loc[starts], mode='markers', marker=dict(color='green', size=10), name='Start'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stops, y=yaw_filt.loc[stops], mode='markers', marker=dict(color='red', size=10), name='Stop'), row=1, col=1)

        fig.update_layout(title=f"Yaw Maneuvers and Wind Speed for Turbine {turb_key}", xaxis_title='Time', yaxis1_title='Yaw Angle (degrees)', yaxis2_title='Wind Speed (m/s)')
        fig.write_html(f"{path2dir_yaw_maneuvers}/{turb_key}_yaw_maneuvers_interactive.html")
        
def analyze_yaw_maneuvers(yaw_data, wind_speed_data, path2dir_fig_base, date_range_total_str, resample_str):
    path2dir_yaw_maneuvers = f"{path2dir_fig_base}/analyzed_yaw_maneuvers/{date_range_total_str}_{resample_str}"
    make_dir_if_not_exists(path2dir_yaw_maneuvers)

    results = []

    for turb_key in yaw_data.columns:
        yaw_filt = yaw_data[turb_key]
        wind_speed_filt = wind_speed_data[f'windspeed_{turb_key.split("_")[-1]}']
        yaw_man, yaw_length, yaw_duration = find_yaw_maneuver(yaw_filt)

        for start, stop in zip(yaw_man[yaw_man == 'cw_start'].index, yaw_man[yaw_man == 'cw_stop'].index):
            if start and stop:
                start_period = start - pd.Timedelta(seconds=60)
                stop_period = stop + pd.Timedelta(seconds=60)
                yaw_chunk = yaw_filt[start_period:stop_period]
                wind_speed_chunk = wind_speed_filt[start_period:stop_period]

                wind_speed_offset = wind_speed_chunk.iloc[-1] - wind_speed_chunk.iloc[0]
                relative_change = (wind_speed_chunk.pct_change().mean()) * 100

                results.append({
                    'Turbine': turb_key,
                    'Maneuver Start': start,
                    'Maneuver Stop': stop,
                    'Yaw Length (degrees)': yaw_length[start],
                    'Yaw Duration (seconds)': yaw_duration[start],
                    'Wind Speed Offset': wind_speed_offset,
                    'Relative Change of Wind Speed (%)': relative_change
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{path2dir_yaw_maneuvers}/maneuver_analysis_results.csv", index=False)

    # Plotting bins
    plt.figure(figsize=(10, 6))
    for length_bin in pd.cut(results_df['Yaw Length (degrees)'], bins=10).unique():
        bin_data = results_df[pd.cut(results_df['Yaw Length (degrees)'], bins=10) == length_bin]
        plt.bar(str(length_bin), bin_data['Relative Change of Wind Speed (%)'].mean(), yerr=bin_data['Relative Change of Wind Speed (%)'].std())

    plt.xlabel('Yaw Length (degrees) Bins')
    plt.ylabel('Average Relative Change of Wind Speed (%)')
    plt.title('Relative Change of Wind Speed by Yaw Length Bins')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{path2dir_yaw_maneuvers}/Relative_Change_by_Yaw_Length.png")
    plt.show()

    return results_df

def plot_yaw_misalignment_histogram(yaw_data, wind_direction_data, path2dir_fig_base, date_range_total_str, resample_str):
    path2dir_yaw_misalignment = f"{path2dir_fig_base}/yaw_misalignment/{date_range_total_str}_{resample_str}"
    make_dir_if_not_exists(path2dir_yaw_misalignment)

    for turb_key in yaw_data.columns:
        yaw_filt = yaw_data[turb_key]
        wind_dir_filt = wind_direction_data[f'winddir_{turb_key.split("_")[-1]}']
        
        # Calculate yaw misalignment
        yaw_misalignment = (yaw_filt - wind_dir_filt).apply(lambda x: (x + 180) % 360 - 180)
        
        # Filter data around the plateau (300 to 320 degrees)
        plateau_mask = (wind_dir_filt >= 300) & (wind_dir_filt <= 320)
        yaw_misalignment_plateau = yaw_misalignment[plateau_mask]
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(yaw_misalignment_plateau, bins=30, edgecolor='black')
        plt.xlabel('Yaw Misalignment (degrees)')
        plt.ylabel('Frequency')
        plt.title(f'Yaw Misalignment Histogram for Turbine {turb_key} (Wind Direction 300-320 degrees)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path2dir_yaw_misalignment}/{turb_key}_yaw_misalignment_histogram.png")
        plt.show()
        
def plot_yaw_scatter_around_plateau(yaw_data, wind_direction_data, path2dir_fig_base, date_range_total_str, resample_str):
    path2dir_yaw_scatter = f"{path2dir_fig_base}/yaw_scatter/{date_range_total_str}_{resample_str}"
    make_dir_if_not_exists(path2dir_yaw_scatter)

    for turb_key in yaw_data.columns:
        yaw_filt = yaw_data[turb_key]
        wind_dir_filt = wind_direction_data[f'winddir_{turb_key.split("_")[-1]}']
        
        # Filter data around the plateau (300 to 320 degrees)
        plateau_mask = (wind_dir_filt >= 300) & (wind_dir_filt <= 320)
        yaw_plateau = yaw_filt[plateau_mask]
        
        # Resample to 60-second means
        yaw_plateau_resampled = yaw_plateau.resample('60s').mean()
        
        # Plot scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(yaw_plateau_resampled.index, yaw_plateau_resampled, alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Yaw Angle (degrees)')
        plt.title(f'60-Second Resampling Mean of Yaw Angles for Turbine {turb_key} (Wind Direction 300-320 degrees)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{path2dir_yaw_scatter}/{turb_key}_yaw_scatter.png")
        plt.show()

# INFO: Algorithm from Andreas
def find_yaw_maneuver(yaw_filt):
    
    ### DEBUG
    print("yaw_filt:")
    print(yaw_filt)

    yaw_temp = yaw_filt.copy()
    yaw_temp.index = yaw_temp.index + datetime.timedelta(seconds=1)
    # Align both series
    yaw_filt_aligned, yaw_temp_aligned = yaw_filt.align(yaw_temp, join='inner', fill_value=np.nan)
    
    # DEBUG
    print("yaw_temp_aligned:")
    print(yaw_temp_aligned)
    
    yaw_diff = (yaw_filt_aligned - yaw_temp_aligned).apply(lambda x: np.mod(x + 180, 360) - 180)
    
    # DEBUG
    print("yaw_diff:")
    print(yaw_diff)
    
    yaw_diff_sum_big = yaw_diff.rolling("20s", center=True).sum() # 20s Pandas rolling window (sum of yaw angle differences over 20s)
    yaw_diff_sum_small = yaw_diff.rolling("3s", center=False).sum() # 3s Pandas rolling window (sum of yaw angle differences over 3s)
    
    # DEBUG
    print("yaw_diff_sum_big:")
    print(yaw_diff_sum_big)
    print("yaw_diff_sum_small:")
    print(yaw_diff_sum_small)
    
    yaw_man_cw = (yaw_diff_sum_big >= 3) & (yaw_diff_sum_small >=1) #  True for timestamps where yaw_diff_sum_big is >= to 3 & yaw_diff_sum_small >= 1
    yaw_man_ccw = (yaw_diff_sum_big <= -3) & (yaw_diff_sum_small<=-1) #  True for timestamps where yaw_diff_sum_big is <= to -3 & yaw_diff_sum_small <= -1
    cw_start = (yaw_man_cw[yaw_man_cw.apply(lambda x: int(x)).diff() == 1].index - datetime.timedelta(seconds=2))
    ccw_start = (yaw_man_ccw[yaw_man_ccw.apply(lambda x: int(x)).diff() == 1].index - datetime.timedelta(seconds=2))
    cw_stop = (yaw_man_cw[yaw_man_cw.apply(lambda x: int(x)).diff() == -1].index - datetime.timedelta(seconds=2))
    ccw_stop = (yaw_man_ccw[yaw_man_ccw.apply(lambda x: int(x)).diff() == -1].index - datetime.timedelta(seconds=2))
    
    # Initialize yaw_man before using it
    yaw_man = pd.Series(data=np.full(len(yaw_filt), np.nan, dtype=object), index=yaw_filt.index)
        
    # DEBUG
    # Check if indices exist in yaw_man before setting values
    valid_cw_start = cw_start[cw_start.isin(yaw_man.index)]
    yaw_man[valid_cw_start] = 'cw_start'
    print("Valid cw_start indices:", valid_cw_start)
    print("cw_start:", cw_start)
    print("cw_stop:", cw_stop)
    print("ccw_start:", ccw_start)
    print("ccw_stop:", ccw_stop)
    
    if (len(cw_start)>0)&(len(cw_stop)>0): # Handle possible overlaps in the yaw maneuvers
        if cw_start[0]>cw_stop[0]:
            cw_stop = cw_stop[1:]
        if cw_start[-1]>cw_stop[-1]:
            cw_start = cw_start[:-1]
    if (len(ccw_start)>0)&(len(ccw_stop)>0):
        if ccw_start[0]>ccw_stop[0]:
            ccw_stop = ccw_stop[1:]
        if ccw_start[-1]>ccw_stop[-1]:
            ccw_start = ccw_start[:-1]
    yaw_man = pd.Series(data=np.full(len(yaw_filt),np.nan,dtype=object),index=yaw_filt.index) # Stores strings indicating yaw maneuver events (cw_start, cw_stop, ccw_start, ccw_stop)
    yaw_man[cw_start]='cw_start'
    yaw_man[ccw_start]='ccw_start'
    yaw_man[cw_stop]='cw_stop'
    yaw_man[ccw_stop]='ccw_stop'
    yaw_length = pd.Series(data=np.full(len(yaw_filt),np.nan),index=yaw_filt.index) # Stores the length of the yaw maneuver
    yaw_length[cw_start] = np.mod(yaw_filt.loc[cw_stop].values - yaw_filt.loc[cw_start].values+180,360)-180
    yaw_length[ccw_start] = np.mod(yaw_filt.loc[ccw_stop].values - yaw_filt.loc[ccw_start].values+180,360)-180
    yaw_duration = pd.Series(data=np.full(len(yaw_filt),np.nan),index=yaw_filt.index) # Stores the duration of the yaw maneuver
    yaw_duration[cw_start] =  (yaw_filt.loc[cw_stop].index-yaw_filt.loc[cw_start].index).total_seconds()
    yaw_duration[ccw_start] = (yaw_filt.loc[ccw_stop].index-yaw_filt.loc[ccw_start].index).total_seconds()
    
    # DEBUG
    print("yaw_duration[cw_start]:")
    print(yaw_duration[cw_start])
    
    id_bad = (yaw_duration[cw_start] <=10).values
    yaw_man[cw_start][id_bad]= np.nan
    yaw_man[cw_stop][id_bad]= np.nan
    yaw_length[cw_start][id_bad] = np.nan
    yaw_length[cw_stop][id_bad] = np.nan
    yaw_duration[cw_start][id_bad] = np.nan
    yaw_duration[cw_start][id_bad] = np.nan


    return yaw_man, yaw_length, yaw_duration

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


