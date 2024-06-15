import os
import pandas as pd
import numpy as np
from _stepresponse_analysis import plot_identified_yaw_maneuvers, print_maneuver_table

# Load configuration
from config import turb_keys, var_keys

# Load processed data
def load_processed_data(path2dir_in_base, start_date_list, end_date_list):
    print('-- start loading processed data from disk')

    df_dict = {}

    for turb_key in turb_keys:
        print(f'- start loading processed data for {turb_key}')

        df_list = []

        for date_int_n, (start_date, end_date) in enumerate(zip(start_date_list, end_date_list)):
            date_list = pd.date_range(start=start_date, end=end_date)
            date_list_str = date_list.astype(str)
            path2dir_in = os.path.join(path2dir_in_base, '1s')

            for date_n, date_str in enumerate(date_list_str):
                fname = f'{date_str}_1s_{turb_key}.h5'
                df_ = pd.read_hdf(os.path.join(path2dir_in, fname))
                df_list.append(df_)
                print(f'loaded {date_str} for {turb_key}')

        df_dict[turb_key] = pd.concat(df_list, keys=None, levels=None)

    print('-- finish loading processed data from disk')

    return df_dict

# Filter data
def filter_data(df_dict):
    print('-- start filtering data')

    df_filt_dict = {}

    for turb_key in turb_keys:
        df = df_dict[turb_key]

        # Filter based on power, pitch, and error code
        power_mask = (df['Power'] >= 500) & (df['Power'] <= 4900)
        pitch_mask = (df['Pitch'] >= 0) & (df['Pitch'] <= 2)
        errorcode_mask = df['Errorcode'] == 6

        df_filt = df[power_mask & pitch_mask & errorcode_mask]
        df_filt_dict[turb_key] = df_filt

        print(f'Filtered data for {turb_key}: {df_filt.shape[0]} rows')

    print('-- finish filtering data')

    return df_filt_dict

# Prepare data for yaw maneuver analysis
def prepare_data_for_yaw_maneuver_analysis(df_filt_dict):
    yaw_data = {}
    wind_speed_data = {}

    for turb_key in turb_keys:
        yaw_data[turb_key] = df_filt_dict[turb_key]['Yaw'].squeeze()
        wind_speed_data[turb_key] = df_filt_dict[turb_key]['WSpeed'].squeeze()

    return yaw_data, wind_speed_data

# Main function
def main():
    # Set paths and date ranges
    path2dir_in_base = 'Data/processed'
    start_date_list = ['2023-06-01', '2023-09-01']
    end_date_list = ['2023-07-31', '2024-01-31']

    # Load processed data
    df_dict = load_processed_data(path2dir_in_base, start_date_list, end_date_list)

    # Filter data
    df_filt_dict = filter_data(df_dict)

    # Prepare data for yaw maneuver analysis
    yaw_data, wind_speed_data = prepare_data_for_yaw_maneuver_analysis(df_filt_dict)

    # Plot identified yaw maneuvers
    path2dir_fig_base = 'Figures'
    date_range_total_str = '2023-06-01_2024-01-31'
    resample_str = '1s'

    plot_identified_yaw_maneuvers(yaw_data, wind_speed_data, path2dir_fig_base, date_range_total_str, resample_str)

    # Print maneuver table
    print_maneuver_table(yaw_data)

if __name__ == "__main__":
    main()