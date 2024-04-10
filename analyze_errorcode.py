## IMPORTS

import os

import matplotlib as mpl
# MPL_BACKEND = 'TkAgg'
MPL_BACKEND = 'QtAgg'
mpl.use(MPL_BACKEND)
import ipython_tools
ipython_tools.set_mpl_magic(MPL_BACKEND)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import config
import processing as proc
from plot_tools import cm2inch, make_dir_if_not_exists
from my_rc_params import rc_params_dict

mpl.rcParams.update(mpl.rcParamsDefault)  # reset to default settings
mpl.rcParams.update(rc_params_dict)  # set to custom settings

plt.close('all')
print('--- script started:', os.path.basename(__file__), '---')

## CUSTOM FUNCTIONS


def my_err_formatter(x):
    if x >= 1e6:
        s = f'{x:1.1e}'
    else:
        s = f'{x:.0f}'
    return s.rjust(8)


## CONTROLS

# filtering --> for testing purposes
bool_filter_data = 0

## SETTINGS
path2dir_code = '/home/jj/Projects/YawDyn/Code/YawDyn_py'
path2dir_data_base = '/home/jj/Projects/YawDyn/Data'
path2dir_fig_base = '/home/jj/Projects/YawDyn/figs'
path2dir_in_base = path2dir_data_base + os.sep + 'processed'

# --- dates to include
# incl. end_date, must be time ordered
start_date_list = ['2023-06-01', '2023-09-01']
end_date_list = ['2023-07-31', '2024-01-31']
# start_date_list = ['2023-07-01']
# end_date_list = ['2023-07-05']
# start_date_list = ['2023-09-01']
# end_date_list = ['2023-12-04']

date_range_total_str = start_date_list[0] + '_' + end_date_list[-1]

# --- resample interval of processed data to be loaded
# resample_interval_s = 1
resample_interval_s = 10
# resample_interval_s = 60

resample_str = str(resample_interval_s) + 's'

turb_keys_to_process = [
    'T3',
    'T4',
    'T5',
    'T6'
]

n_turbs = len(turb_keys_to_process)

turb_keys_split_by_pair = [
    ['T3', 'T4'],
    ['T5', 'T6'],
]

n_turb_pairs = len(turb_keys_split_by_pair)

# T4 and T6 are upstream
idx_upstream = 1
idx_downstream = 0

turb_keys_up = [tk[idx_upstream] for tk in turb_keys_split_by_pair]
turb_keys_down = [tk[idx_downstream] for tk in turb_keys_split_by_pair]

ctrl_keys = ['on', 'off']

## FILTER SETTINGS

# for filtering (testing purposes)
power_min = 100
power_max = 6000

errorcode_min = -0.5
errorcode_max = 0.5
# errorcode_min = 5.5
# errorcode_max = 6.5

## PATHS

path2dir_errcode = (
        path2dir_data_base + os.sep + 'error_code' + os.sep + date_range_total_str
        + os.sep + resample_str
)
make_dir_if_not_exists(path2dir_errcode)

path2dir_fig_errcode = (
        path2dir_fig_base + os.sep + 'error_code' + os.sep + resample_str
        + os.sep + date_range_total_str)
make_dir_if_not_exists(path2dir_fig_errcode)

## PLOT SETTINGS

# FIG_WINDOW_START_POS = np.array((0, 0))
FIG_WINDOW_START_POS = np.array((-1920, -230))
fig_window_pos = np.copy(FIG_WINDOW_START_POS)

color_dict_by_turb = config.color_dict_by_turb
marker_dict_by_turb = config.marker_dict_by_turb

## ANALYZE ERRORCODE

# --- load processed data
df_dict = proc.load_processed_data(
    turb_keys_to_process,
    start_date_list,
    end_date_list,
    path2dir_in_base,
    resample_interval_s
)

# --- count errorcode values
path2file_err_val_count = path2dir_errcode + os.sep + 'err_val_cnt.txt'

err_val_cnt = {}
for turb_n, turb_key in enumerate(turb_keys_to_process):
    df_x = df_dict[turb_key]
    err_val_cnt[turb_key] = df_x['Errorcode'].value_counts()

    if turb_n == 0:
        err_val_cnt[turb_key].to_csv(
            path2file_err_val_count,
            sep=',',
            mode='w'
        )

    else:
        err_val_cnt[turb_key].to_csv(
            path2file_err_val_count,
            sep=',',
            mode='a'
        )

# --- bar plot of err code value count

figsize_cm = (5 * n_turbs, 8)

fig, axes = plt.subplots(nrows=1, ncols=n_turbs, sharey=True, figsize=cm2inch(figsize_cm))

for turb_n, turb_key in enumerate(turb_keys_to_process):

    ax = axes[turb_n]

    err_cnt_x = err_val_cnt[turb_key]
    x_pos = np.arange(err_cnt_x.values.shape[0])
    x_ticks = err_cnt_x.index.values
    x_ticks = map(my_err_formatter, x_ticks)

    ax.bar(
        x_pos, err_cnt_x.values,
        color=color_dict_by_turb[turb_key],
        width=0.8,
        label=turb_key
    )

    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(my_err_formatter)
    ax.set_xticks(x_pos, x_ticks)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlabel('Errorcode')
    ax.grid(visible=True, axis='y', zorder=0)
    ax.legend(loc=1)

axes[0].set_ylabel('Counts')

fig.tight_layout()
figname = 'err_val_cnt.png'
fig.savefig(path2dir_fig_errcode + os.sep + figname)

# filter data

if bool_filter_data:

    df_filt_dict = {}

    for turb_key in turb_keys_to_process:

        df_x = df_dict[turb_key]
        mask_err = df_x['Errorcode'].between(errorcode_min, errorcode_max, inclusive='both')
        mask_pow = df_x['Power'].between(power_min, power_max, inclusive='both')

        mask = mask_err & mask_pow

        df_filt_dict[turb_key] = df_x[mask]

    # --- plot err count for filtered data

    err_val_cnt_filt = {}

    for turb_n, turb_key in enumerate(turb_keys_to_process):

        df_x = df_filt_dict[turb_key]
        err_val_cnt_filt[turb_key] = df_x['Errorcode'].value_counts()

    figsize_cm = (5 * n_turbs, 8)

    fig, axes = plt.subplots(nrows=1, ncols=n_turbs, sharey=True, figsize=cm2inch(figsize_cm))

    for turb_n, turb_key in enumerate(turb_keys_to_process):

        ax = axes[turb_n]
        err_cnt_x = err_val_cnt_filt[turb_key]
        x_pos = np.arange(err_cnt_x.values.shape[0])
        x_ticks = err_cnt_x.index.values
        x_ticks = map(my_err_formatter, x_ticks)

        ax.bar(
            x_pos, err_cnt_x.values,
            color=color_dict_by_turb[turb_key],
            width=0.8,
            label=turb_key
        )

        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(my_err_formatter)
        ax.set_xticks(x_pos, x_ticks)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('Errorcode')
        ax.grid(visible=True, axis='y', zorder=0)
        ax.legend(loc=1)

    axes[0].set_ylabel('Counts')

    fig.tight_layout()
    figname = 'err_val_cnt_filt.png'
    fig.savefig(path2dir_fig_errcode + os.sep + figname)

print('--- script finished:', os.path.basename(__file__), '---')
