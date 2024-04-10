"""
Processes raw data for YawDyn from BBO 1 from Oceanbreeze and stores HDF files for each turbine and each day
Need to install 'tables' package

----- IMPORTANT NOTE -----
The raw data files from Oceanbreeze contain the filename and the unit in lines 1 and 2.
For Wind directions, the unit is given as <circle> sign, which leads to problems, when the file
is read with pandas (even with skiprows). Thus, the first and second line of the raw files have to be
removed, e.g. using the script delete_first_lines.sh, before working on them

"""

##
import os
import pandas as pd

import analysis as ana
from plot_tools import make_dir_if_not_exists

from config import filenames_dict

print('--- script started:', os.path.basename(__file__), '---')

## SETTINGS
path2dir_in_base = '/home/jj/Projects/YawDyn/Data/raw'
# path2dir_in = '/home/jj/Projects/YawDyn/Data/raw/YawDyn_Daten_230601_230801'
path2dir_out_base = r'/home/jj/Projects/YawDyn/Data/processed'
make_dir_if_not_exists(path2dir_out_base)
# path2dir_in = '/home/jj/Data/Yawdyn/current'
# path2dir_out = '/home/jj/Data/Yawdyn/current/processed'

## SELECT turb ids and quantities to plot,
turb_ids_to_process = [
    3, 4, 5, 6
]

# variables to process. need to be strings from quant_keys
var_keys_to_process = [
    'Power',
    'WSpeed',
    'WDir',
    'Yaw',
    'Pitch',
    'Errorcode',
    'PowerRef',
    'ControlSwitch'
]

turb_keys_to_process = ['T' + str(turb_id) for turb_id in turb_ids_to_process]

## load data
# number of quantities to be loaded
n_selected = len(var_keys_to_process)

# incl. start date
# incl. end date
# start_date = '2023-06-01'
# end_date = '2023-07-31'
# start_date = '2023-09-01'
# end_date = '2023-11-19'
start_date = '2023-11-20'
end_date = '2024-01-31'

# id_str = '230901_231120'
id_str = start_date + '_' + end_date

path2dir_in = path2dir_in_base + os.sep + id_str

date_list = pd.date_range(start=start_date, end=end_date)
date_list_str = date_list.astype(str)

# number of data rows to be loaded. 'None' to load all rows.
# n_rows_to_load = 10000
# n_rows_to_load = len(date_list) * 24 * 60 * 60 * 4
n_rows_to_load = None

# resample time interval for high frequency data (power etc.)
# can be '1s', '10s', '60s', ..., or None (for irregular data)
# resample_str_list = [1, 10, 60, 600, 3600]
# resample_str_list = [60, 600, 3600]
resample_str_list = [10]
# resample_str_list = [600]
resample_str_list = [str(t_) + 's' for t_ in resample_str_list]
# resample_str_list = [None]

for resample_str in resample_str_list:

    print('-- start processing for ' + resample_str + ' --')

    if resample_str is None:
        path2dir_out = path2dir_out_base + os.sep + 'irreg'
    else:
        path2dir_out = path2dir_out_base + os.sep + resample_str

    make_dir_if_not_exists(path2dir_out)

    for turb_n, turb_key in enumerate(turb_keys_to_process):

        df_list = []

        for quant_n, var_key in enumerate(var_keys_to_process):
            if var_key == 'ControlSwitch' and turb_key not in ['T4', 'T6']:
                pass
            # if not (var_key == 'ControlSwitch' and turb_key in ['T3', 'T5']):
            else:
                df = pd.read_csv(path2dir_in + os.sep + filenames_dict[turb_key][var_key],
                                 delimiter=';', names=['Datetime', var_key],
                                 nrows=n_rows_to_load,
                                 parse_dates=[0], date_format='%d.%m.%y %H:%M:%S')
                df.set_index('Datetime', inplace=True)

                if resample_str is not None:
                    # remove duplicate timesteps
                    # the smallest time step in the raw data is one second
                    # however, it occurs that there are several values for the same time step
                    # for uniform processing: if a Datetime index has several values
                    # use only the last value and discard the others.
                    # Power has higher writing frequency (3.16 Hz), but those are
                    # moving average values over 1 second
                    df = df.groupby(df.index).last()

                    # after duplicate removal the data is at a sampling interval of 1 sec or larger
                    # for some quantities it is also irregular
                    # i.e. for a resampling to 1 sec time step, it will change nothing for power,
                    # and it will be an upsampling for all other quantities
                    # for larger sampling intervals, it will be a downsampling,
                    # need to choose the right averaging method for angle averages

                    if resample_str == '1s':
                        if var_key == 'Power':
                            # Power has higher writing frequency (3.16 Hz), but those are
                            # moving average values over 1 second
                            df = df.resample('1s').ffill()

                    if resample_str != '1s':
                        # forward fill every variable to 1 sec first for proper downsampling
                        df = df.resample('1s').ffill()

                        # downsampling by averaging
                        df = ana.downsample_dataframe_properly(df, var_key, resample_str)

                    df_list.append(df)
                    print('processed', turb_key, var_key)

                else:
                    # case where resample_str is None
                    for date_n, date_str in enumerate(date_list_str):

                        fname = var_key + '_' + date_str + '_' + turb_key

                        df.loc[date_str:date_str].to_hdf(path2dir_out + os.sep + fname + '.h5',
                                                         'scada', mode='w')

        if resample_str is not None:
            df = pd.concat(df_list, axis=1)

            # optional: fill NaN with last valid value from the past
            if resample_str != '1s':
                df = df.ffill()
            # optional: after ffill, fill NaN from beginning of time series
            # with the next valid future value
            # df = df.bfill()

            for date_n, date_str in enumerate(date_list_str):

                fname = date_str + '_' + resample_str + '_' + turb_key

                df.loc[date_str:date_str].to_hdf(path2dir_out + os.sep + fname + '.h5',
                                                 'scada', mode='w')
            print('--> stored data for', turb_key)

    print('-- finish processing for ' + resample_str + ' --')

##
print('--- script finished:', os.path.basename(__file__), '---')
