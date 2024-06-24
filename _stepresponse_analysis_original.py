import pandas as pd
from plot_tools import make_dir_if_not_exists
import datetime
import numpy as np
import os

#import plotly.graph_objs as go
#from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
        
def analyze_yaw_maneuvers(yaw_data, wind_speed_data, path2dir_fig_base, date_range_total_str, resample_str):
    path2dir_yaw_maneuvers = f"{path2dir_fig_base}/analyzed_yaw_maneuvers/{date_range_total_str}_{resample_str}"
    make_dir_if_not_exists(path2dir_yaw_maneuvers)

    results = []

    for turb_key in yaw_data.columns:
        yaw_filt = yaw_data[turb_key]
        wind_speed_filt = wind_speed_data[f'windspeed_{turb_key.split("_")[-1]}']
        yaw_man, yaw_lengths, yaw_durations = find_yaw_maneuver(yaw_filt)
        
        # Process both cw and ccw maneuvers
        for maneuver_type in ['cw', 'ccw']:
            start_indices = yaw_man[yaw_man == f'{maneuver_type}_start'].index
            stop_indices = yaw_man[yaw_man == f'{maneuver_type}_stop'].index
            
            for start, stop in zip(start_indices, stop_indices):
                if start and stop:
                    yaw_length = np.mod(yaw_filt.loc[stop] - yaw_filt.loc[start] + 180, 360) - 180
                    start_period = start - pd.Timedelta(seconds=60)
                    stop_period = stop + pd.Timedelta(seconds=60)
                    
                    wind_speed_chunk_before = wind_speed_filt[start_period:start].mean()                
                    wind_speed_chunk_after = wind_speed_filt[stop:stop_period].mean()

                    wind_speed_offset = wind_speed_chunk_after - wind_speed_chunk_before
                    relative_change = wind_speed_chunk_after / wind_speed_chunk_before

                    results.append({
                        'Turbine': turb_key,
                        'Maneuver Start': start,
                        'Maneuver Stop': stop,
                        'Yaw Length': yaw_length,
                        'Yaw Duration': (stop - start).total_seconds(),
                        'Wind Speed Offset': wind_speed_offset,
                        'Relative Wind Speed Change': relative_change,
                        'Maneuver Type': maneuver_type
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{path2dir_yaw_maneuvers}/maneuver_analysis_results.csv", index=False)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    for i, maneuver_type in enumerate(['cw', 'ccw', 'combined']):
        if maneuver_type != 'combined':
            subset = results_df[results_df['Maneuver Type'] == maneuver_type]
        else:
            subset = results_df
        
        if len(subset) > 1:  # Check if there's enough data to plot
            ax2 = axs[i].twinx()  # Create a twin Axes sharing the xaxis
            
            # Remove NaN values for fitting
            valid_data = subset.dropna(subset=['Yaw Length', 'Wind Speed Offset', 'Relative Wind Speed Change'])
            
            axs[i].scatter(valid_data['Yaw Length'], valid_data['Wind Speed Offset'], label=f'{maneuver_type} maneuvers', color='blue')
            ax2.scatter(valid_data['Yaw Length'], valid_data['Relative Wind Speed Change'], label=f'{maneuver_type} maneuvers (Relative Change)', color='green', marker='x')
            
            mean_value = valid_data['Wind Speed Offset'].mean()
            axs[i].axhline(mean_value, color='r', linestyle='--', label=f'Mean Offset: {mean_value:.2f}')
            
            # Fit and plot trend lines for Wind Speed Offset and Relative Wind Speed Change
            if len(valid_data) > 1:  # Ensure we have at least 2 points for fitting
                # Wind Speed Offset (blue)
                z_offset = np.polyfit(valid_data['Yaw Length'], valid_data['Wind Speed Offset'], 1)
                p_offset = np.poly1d(z_offset)
                x_range = np.linspace(valid_data['Yaw Length'].min(), valid_data['Yaw Length'].max(), 100)
                axs[i].plot(x_range, p_offset(x_range), "b--", 
                            label=f'Trend Offset: {z_offset[0]:.2f}x + {z_offset[1]:.2f}')
                
                # Relative Wind Speed Change (green)
                z_relative = np.polyfit(valid_data['Yaw Length'], valid_data['Relative Wind Speed Change'], 1)
                p_relative = np.poly1d(z_relative)
                ax2.plot(x_range, p_relative(x_range), "g--", 
                         label=f'Trend Relative: {z_relative[0]:.2f}x + {z_relative[1]:.2f}')
            else:
                print(f"Not enough valid data points to fit trend lines for {maneuver_type} maneuvers")
            
            # Labels and titles
            axs[i].set_xlabel('Yaw Length')
            axs[i].set_ylabel('Wind Speed Offset')
            ax2.set_ylabel('Relative Wind Speed Change', color='green')
            axs[i].set_title(f'{maneuver_type.capitalize()} Maneuvers: Yaw Length vs Wind Speed Offset and Relative Change')
            
            # Legends
            axs[i].legend(loc='upper left')
            ax2.legend(loc='upper right')
        else:
            axs[i].text(0.5, 0.5, f'No data for {maneuver_type} maneuvers', 
                        horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)

    plt.tight_layout()
    plt.savefig(f"{path2dir_yaw_maneuvers}/Yaw_Maneuvers_Analysis.png")
    plt.show()

    return results_df

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
    print("yaw_duration[ccw_start]:")
    print(yaw_duration[ccw_start])    
    
    id_bad = (yaw_duration[cw_start] <=4).values
    id_bad_ccw = (yaw_duration[ccw_start] <=4).values
    yaw_man[cw_start][id_bad]= np.nan
    yaw_man[cw_stop][id_bad]= np.nan
    yaw_length[cw_start][id_bad] = np.nan
    yaw_length[cw_stop][id_bad] = np.nan
    yaw_duration[cw_start][id_bad] = np.nan
    yaw_duration[cw_start][id_bad] = np.nan
    yaw_man[ccw_start][id_bad_ccw]= np.nan
    yaw_man[ccw_stop][id_bad_ccw]= np.nan
    yaw_length[ccw_start][id_bad_ccw] = np.nan
    yaw_length[ccw_stop][id_bad_ccw] = np.nan
    yaw_duration[ccw_start][id_bad_ccw] = np.nan
    yaw_duration[ccw_start][id_bad_ccw] = np.nan


    return yaw_man, yaw_length, yaw_duration