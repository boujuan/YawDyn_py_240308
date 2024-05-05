"""
load processed SCADA data (.hdf5 or .h5 format)

IN WORK: disregard bins with too few entries

NOTES:
get group keys etc for pandas grouping objects and binned multindex dataframe
group_keys = list(df_binned_2D_grouper_dict['on']['T3'].groups.keys())
--> careful: group_keys might contain empty bins. this can lead to errors, a fix how to clean
the group_keys is used in the code
wspd intervals:
wspd_index = df_binned_2D_mean_dict['on']['T3'].index.get_level_values(level='WSpeed').unique()
wdir intervals:
wdir_index = df_binned_2D_std_dict['on']['T3'].index.get_level_values(level='WDir').unique()
"""

## IMPORTS

import os

import matplotlib as mpl
MPL_BACKEND = 'TkAgg'
# MPL_BACKEND = 'QtAgg'
mpl.use(MPL_BACKEND)
import ipython_tools
ipython_tools.set_mpl_magic(MPL_BACKEND)
import matplotlib.pyplot as plt

import numpy as np

import warnings
# careful with removing warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
# careful with removing warnings
pd.options.mode.chained_assignment = None  # default='warn'

import config
import processing as proc
import plotting as plo
from plot_tools import cm2inch, move_figure_window, make_dir_if_not_exists, marker_list, \
    delete_all_files_from_dir
from my_rc_params import rc_params_dict
mpl.rcParams.update(mpl.rcParamsDefault) # reset to default settings
mpl.rcParams.update(rc_params_dict) # set to custom settings

plt.close('all')
print('--- script started:', os.path.basename(__file__), '---')

## CONTROLS

# INFO: CHOOSE WHETHER TO USE 1D AND/OR 2D BINNING
# --- general usage switch
bool_use_1D_binned_filtering = 1
bool_use_2D_binned_filtering = 1

# INFO: CHOOSE WHETHER TO REFILTER DATA (IMPORTANT) THIS WILL GENERATE NEW FILTERED DATA
# --- filtering: filtering has only to be done once. Data will then be saved to hard disk.
# -> Only need to refilter, if settings have changed.
bool_refilter_data_for_limits = 1
bool_refilter_1D_binned_for_time_interv = 1
bool_refilter_2D_binned_for_time_interv = 1

# to divide standard deviation by sqrt(N), where N is the number of datapoints in each bin
bool_use_std_of_mean = 1

# INFO: CHOOSE WHETHER TO PLOT
# --- plotting
# save figs to hard drive
bool_save_fig = 1

# activate ALL possible plots or NO plots or select specific plots below
bool_plot_all = 0
bool_plot_nothing = 0

# select specific plots
bool_plot_yaw_table = 1

bool_plot_errorcode_counts = 1

bool_count_intervals_1D_and_plot = 0
bool_count_intervals_2D_and_plot = 1

# careful, when plotting full dataset with high time resolution (e.g. 1 second)
# this can take long time or lead to memory error
bool_plot_unfiltered_data_overview = 0

bool_plot_limit_filtered_data_overview = 1
bool_plot_limit_filtered_data_binned = 1

bool_plot_1D_binned_time_int_filtered_data = 0
bool_plot_2D_binned_time_int_filtered_data = 1

bool_plot_1D_binned_valid_intervals = 0
bool_plot_2D_binned_valid_intervals = 0

# select what to include in overview plots (for processed and filtered data)
bool_plot_overview = 1
bool_plot_overview_split = 1
bool_plot_wdir_selected_range = 1
bool_plot_wdir_vs_yaw = 1
bool_plot_wdir_vs_yaw_selected_range = 1

# --- derived controls
if bool_refilter_data_for_limits:
    bool_save_filtered_data = 1
else:
    bool_save_filtered_data = 0

# switch on / off all plotting commands
if bool_plot_all:
    bool_plot_switch = 1
elif bool_plot_nothing:
    bool_plot_switch = 0

if bool_plot_all or bool_plot_nothing:

    bool_save_fig = bool_plot_switch

    bool_count_intervals_1D_and_plot = bool_plot_switch
    bool_count_intervals_2D_and_plot = bool_plot_switch

    bool_plot_yaw_table = bool_plot_switch
    bool_plot_errorcode_counts = bool_plot_switch

    bool_plot_unfiltered_data_overview = bool_plot_switch
    bool_plot_limit_filtered_data_overview = bool_plot_switch
    bool_plot_limit_filtered_data_binned = bool_plot_switch
    bool_plot_1D_binned_time_int_filtered_data = bool_plot_switch
    bool_plot_2D_binned_time_int_filtered_data = bool_plot_switch
    bool_plot_1D_binned_valid_intervals = bool_plot_switch
    bool_plot_2D_binned_valid_intervals = bool_plot_switch

if not bool_use_1D_binned_filtering:
    bool_refilter_1D_binned_for_time_interv = 0
    bool_count_intervals_1D_and_plot = 0
    bool_plot_1D_binned_time_int_filtered_data = 0
    bool_plot_1D_binned_valid_intervals = 0

if not bool_use_2D_binned_filtering:
    bool_refilter_2D_binned_for_time_interv = 0
    bool_count_intervals_2D_and_plot = 0
    bool_plot_2D_binned_time_int_filtered_data = 0
    bool_plot_2D_binned_valid_intervals = 0

## SETTINGS

# INFO: CHANGE PATHS TO FOLDERS ACCORDINGLY
# --- paths
path2dir_code = 'Code'
path2dir_data_base = 'Data'
path2dir_fig_base = 'Figures'
path2dir_in_base = path2dir_data_base + os.sep + 'processed'

path2dir_yaw_table = path2dir_code
fname_yaw_table = 'lookup_table.csv'

# INFO: CHANGE DATE RANGE
# --- dates to include
# incl. end_date, must be time ordered
start_date_list = ['2023-06-01', '2023-09-01']
end_date_list = ['2023-07-31', '2024-01-31']
# start_date_list = ['2023-07-01']
# end_date_list = ['2023-07-07']
# start_date_list = ['2023-09-01']
# end_date_list = ['2023-12-05']

date_range_total_str = start_date_list[0] + '_' + end_date_list[-1]

# INFO: CHANGE THE RESAMPLING INTERVAL
# --- resample interval of processed data to be loaded
resample_interval_s = 10
# resample_interval_s = 60

resample_str = str(resample_interval_s) + 's'

turb_keys_to_process = [
    'T3',
    'T4',
    'T5',
    'T6'
]

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

# INFO: Change the vars
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

n_selected_vars = len(var_keys_to_process)

# vars selected for binning process.
# these vars will be binned by bin criteria of 1 or 2 selected variables (see below)
var_keys_to_bin = [
    'Power',
    'WSpeed',
    'WDir',
    'Yaw',
    'NormPower'
]

ctrl_keys = ['on', 'off']

# turbine specs and other params
air_density = 1.225
rotor_diam = 122
rotor_rad = 0.5 * rotor_diam
rotor_area = np.pi * rotor_rad**2

## FILTER SETTINGS

filt_id = 1

# for below:
# the control approach leads to changed yaw lookup table in this range:
# wdir_min = 288
# wdir_max = 340
# wdir_min needs to be in [0, 360)
# wdir_max can be in [0, 720)
# "am Netz" = connected to power grid = normal operation:
# errorcode = 6
# partial load:
# pitch between -0.3 and 2.2
# rated Power is approx 5200 kW

if filt_id == 1:

    wdir_min_per_pair = [285, 287]
    wdir_max_per_pair = [342, 342]
    wspd_min = 4.5
    wspd_max = 12.5
    power_min = 200
    power_max = 4999
    powerref_min = 4999
    powerref_max = 7000
    errorcode_min = 5.5
    errorcode_max = 6.5
    pitch_min = -0.3
    pitch_max = 2.2

    #INFO: BIN WINDSPEED SETTINGS TO TWEAK
    # bin settings
    wspd_bin_min = 4.5
    wspd_bin_max = 12.5
    wspd_bin_width = 1.0

    # pair 1
    wdir_bin_min_1 = wdir_min_per_pair[0]
    wdir_bin_max_1 = wdir_max_per_pair[0]
    wdir_bins_pair_1 = np.array(
        [wdir_bin_min_1 - 0.1, 295.7, 306, 316, 326.3, wdir_bin_max_1 + 0.1])

    # pair 2
    wdir_bin_min_2 = wdir_min_per_pair[1]
    wdir_bin_max_2 = wdir_max_per_pair[1]
    wdir_bins_pair_2 = np.array(
        [wdir_bin_min_2 - 0.1, 298, 308.2, 318.2, 327.3, wdir_bin_max_2 + 0.1])

    # uninterrupted interval settings
    min_interval_duration_s = 8 * 60
    discard_time_at_beginning_s = 3 * 60
    # max gap between filtered valid time steps to be counted as the same interval
    max_gap_duration_s = 30

else:

    wdir_min_per_pair = [285, 287] # INFO: Interval filterning
    wdir_max_per_pair = [342, 342]
    wspd_min = 5.0
    wspd_max = 12.0
    power_min = 200
    power_max = 4999
    powerref_min = 4999
    powerref_max = 7000
    errorcode_min = 5.5
    errorcode_max = 6.5
    pitch_min = -0.3
    pitch_max = 2.2

    # bin settings
    wspd_bin_min = 5.5
    wspd_bin_max = 12.5
    wspd_bin_width = 1.0

    # pair 1
    wdir_bin_min_1 = wdir_min_per_pair[0]
    wdir_bin_max_1 = wdir_max_per_pair[0]
    wdir_bins_pair_1 = np.array(
        [wdir_bin_min_1 - 0.1, 295.7, 306, 316, 326.3, wdir_bin_max_1 + 0.1])

    # pair 2
    wdir_bin_min_2 = wdir_min_per_pair[1]
    wdir_bin_max_2 = wdir_max_per_pair[1]
    wdir_bins_pair_2 = np.array(
        [wdir_bin_min_2 - 0.1, 298, 308.2, 318.2, 327.3, wdir_bin_max_2 + 0.1])

    # uninterrupted interval settings
    min_interval_duration_s = 8 * 60
    discard_time_at_beginning_s = 3 * 60 # INFO: remove start of the yawing procedure
    # max gap between filtered valid time steps to be counted as the same interval
    max_gap_duration_s = 30

# T6 wake steering control is active over full range
# T4 wake steering control starts at 2023-06-29 08:00:00
time_index_min_dict = {
    'T3': None,
    'T4': '2023-06-29 08:00:00',
    'T5': None,
    'T6': None,
}

## DERIVED BINNING PARAMS

wspd_bins = np.arange(wspd_bin_min, wspd_bin_max + 0.5, wspd_bin_width)
wspd_bin_centers = 0.5 * (wspd_bins[:-1] + wspd_bins[1:])

wspd_bins_per_pair = [
    wspd_bins,
    wspd_bins
]

#INFO: BIN WINDDIRECTION SETTINGS
wdir_bins_per_pair = [
    wdir_bins_pair_1,
    wdir_bins_pair_2
]

wspd_bin_centers_per_pair = []
wdir_bin_centers_per_pair = []

for pair_n in range(n_turb_pairs):

    wspd_bins = wspd_bins_per_pair[pair_n]
    wspd_bin_centers = 0.5 * (wspd_bins[:-1] + wspd_bins[1:])
    wspd_bin_centers_per_pair.append(wspd_bin_centers)

    wdir_bins = wdir_bins_per_pair[pair_n]
    wdir_bin_centers = 0.5 * (wdir_bins[:-1] + wdir_bins[1:])
    wdir_bin_centers_per_pair.append(wdir_bin_centers)

# --- create yaw setpoint table per pair
wdir_setpoint_per_pair = []

for pair_n in range(n_turb_pairs):
    wdir_bins = wdir_bins_per_pair[pair_n]
    wdir_sp_ = np.linspace(wdir_bins[0] - 1, wdir_bins[-1] + 1, 1000)
    wdir_setpoint_per_pair.append(wdir_sp_)

yaw_setpoint_per_pair = proc.calc_yaw_table_setpoint_vs_wdir(
        path2dir_yaw_table,
        fname_yaw_table,
        wdir_setpoint_per_pair
)

## PATHS

path2dir_limit_filtered_base = path2dir_data_base + os.sep + 'limit_filtered'
path2dir_limit_filtered = (
        path2dir_limit_filtered_base + os.sep + date_range_total_str + f'_id_{filt_id:.0f}'
        + os.sep + resample_str
)
make_dir_if_not_exists(path2dir_limit_filtered)

path2dir_time_int_filtered_base = path2dir_data_base + os.sep + 'time_interval_filtered'
path2dir_time_int_filtered = (
        path2dir_time_int_filtered_base + os.sep + date_range_total_str + f'_id_{filt_id:.0f}'
        + os.sep + resample_str
)
make_dir_if_not_exists(path2dir_time_int_filtered)

make_dir_if_not_exists(path2dir_fig_base)

path2dir_fig_yaw_table = path2dir_fig_base + os.sep + 'yaw_table'
make_dir_if_not_exists(path2dir_fig_yaw_table)

path2dir_fig = (
        path2dir_fig_base + os.sep + date_range_total_str + f'_id_{filt_id:.0f}'
        + os.sep + resample_str
)

path2dir_fig_int_base = path2dir_fig + os.sep + 'filtered_intervals'
make_dir_if_not_exists(path2dir_fig_int_base)

path2dir_fig_limit_filtered = path2dir_fig + os.sep + 'limit_filtered'
make_dir_if_not_exists(path2dir_fig_limit_filtered)

path2dir_fig_time_int_filtered = path2dir_fig + os.sep + 'time_interval_filtered'
make_dir_if_not_exists(path2dir_fig_time_int_filtered)

## PLOT SETTINGS

# FIG_WINDOW_START_POS = np.array((0, 0))
FIG_WINDOW_START_POS = np.array((-1920, -230))
fig_window_pos = np.copy(FIG_WINDOW_START_POS)

color_dict_by_turb = config.color_dict_by_turb
marker_dict_by_turb = config.marker_dict_by_turb

# delta time index for unfiltered plotting (plot every ... timesteps)
dit_plot_unfilt = 60

# yaw table plot settings
wdir_min_yaw_table = 286
wdir_max_yaw_table = 342
d_wdir_plot_yaw_table = 10
figsize_yaw_table = cm2inch(10, 10)

## PLOT YAW TABLE

if bool_plot_yaw_table:

    plo.plot_yaw_table(
            path2dir_yaw_table,
            fname_yaw_table,
            path2dir_fig_yaw_table,
            figsize_yaw_table,
            wdir_min_yaw_table,
            wdir_max_yaw_table,
            d_wdir_plot_yaw_table
    )

## FILTER DATA FOR SET VARIABLE LIMITS

if bool_refilter_data_for_limits:

    df_dict = proc.load_processed_data(
        turb_keys_to_process,
        start_date_list,
        end_date_list,
        path2dir_in_base,
        resample_interval_s
    )

    df_filt_ctrl_turb_dict = \
        proc.filter_data_for_limits(
            df_dict,
            wdir_min_per_pair, wdir_max_per_pair,
            power_min, power_max,
            powerref_min, powerref_max,
            wspd_min, wspd_max,
            errorcode_min, errorcode_max,
            pitch_min, pitch_max,
            time_index_min_dict,
            turb_keys_split_by_pair,
            idx_upstream, idx_downstream,
            bool_save_filtered_data,
            resample_interval_s,
            path2dir_limit_filtered
        )

else:

    df_filt_ctrl_turb_dict = \
        proc.load_filtered_data(
            turb_keys_to_process,
            path2dir_limit_filtered,
            resample_interval_s
        )

if 'NormPower' in var_keys_to_bin:
    # --- define normalized power per turbine
    proc.add_norm_power_to_df_ctrl_turb_dict(
        df_filt_ctrl_turb_dict,
        turb_keys_split_by_pair,
        ctrl_keys,
        idx_upstream,
        rotor_diam,
        air_density
    )

## BIN 1D AND THEN FILTER FOR TIME INTERVAL CRITERIA: max gap and min duration

var_key_used_for_filter_binning = 'WDir'
bins_to_bin_by_per_pair = wdir_bins_per_pair

if bool_refilter_1D_binned_for_time_interv:

    df_filt_time_int_1D_ctrl_turb_dict = proc.filter_data_for_time_int_1D_binned(
        df_filt_ctrl_turb_dict,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_to_bin_by_per_pair,
        var_key_used_for_filter_binning,
        bool_use_std_of_mean,
        resample_interval_s,
        max_gap_duration_s,
        min_interval_duration_s,
        discard_time_at_beginning_s,
        ctrl_keys,
        path2dir_time_int_filtered
    )

elif bool_use_1D_binned_filtering:

    path2dir_time_int_filtered_ext = (
            path2dir_time_int_filtered + os.sep + f'1D_{var_key_used_for_filter_binning}'
    )

    df_filt_time_int_1D_ctrl_turb_dict = \
        proc.load_filtered_data(
            turb_keys_to_process,
            path2dir_time_int_filtered_ext,
            resample_interval_s
        )

## BIN 2D AND THEN FILTER FOR TIME INTERVAL CRITERIA: max gap and min duration

var_key_used_for_filter_binning_1 = 'WSpeed'
var_key_used_for_filter_binning_2 = 'WDir'

bins_to_bin_by_1_per_pair = wspd_bins_per_pair
bins_to_bin_by_2_per_pair = wdir_bins_per_pair

if bool_refilter_2D_binned_for_time_interv:

    df_filt_time_int_2D_ctrl_turb_dict = proc.filter_data_for_time_int_2D_binned(
        df_filt_ctrl_turb_dict,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_to_bin_by_1_per_pair,
        bins_to_bin_by_2_per_pair,
        var_key_used_for_filter_binning_1,
        var_key_used_for_filter_binning_2,
        bool_use_std_of_mean,
        resample_interval_s,
        max_gap_duration_s,
        min_interval_duration_s,
        discard_time_at_beginning_s,
        ctrl_keys,
        path2dir_time_int_filtered
    )

elif bool_use_2D_binned_filtering:

    path2dir_time_int_filtered_ext_2D = (
            path2dir_time_int_filtered + os.sep
            + f'2D_{var_key_used_for_filter_binning_1}_{var_key_used_for_filter_binning_2}'
    )

    df_filt_time_int_2D_ctrl_turb_dict = \
        proc.load_filtered_data(
            turb_keys_to_process,
            path2dir_time_int_filtered_ext_2D,
            resample_interval_s
        )

## PLOTTING

## PLOT LIMIT FILTERED DATA OVERVIEW

if bool_plot_limit_filtered_data_overview:

    wdir_min_overview = 286
    wdir_max_overview = 342

    plo.plot_df_ctrl_turb_dict_overview(
        df_filt_ctrl_turb_dict,
        var_keys_to_process,
        turb_keys_to_process,
        turb_keys_split_by_pair,
        resample_str,
        ctrl_keys,
        path2dir_fig_limit_filtered,
        marker_dict_by_turb,
        color_dict_by_turb,
        wdir_min_overview,
        wdir_max_overview,
        bool_plot_overview,
        bool_plot_overview_split,
        bool_plot_wdir_selected_range,
        bool_plot_wdir_vs_yaw,
        bool_plot_wdir_vs_yaw_selected_range
    )

## PLOT LIMIT FILTERED DATA BINNED

if bool_plot_limit_filtered_data_binned:

    # --- plot binned limit filtered pure (no interval duration filter)

    cp_min_plot_1D_per_turb = 0.25
    cp_max_plot_1D_per_turb = 0.5
    cp_min_plot_1D_per_pair = 0.25
    cp_max_plot_1D_per_pair = 0.5

    cp_min_down_plot_2D_per_turb = 0.2
    cp_max_down_plot_2D_per_turb = 0.55
    cp_min_up_plot_2D_per_turb = 0.3
    cp_max_up_plot_2D_per_turb = 0.5

    cp_min_plot_2D_per_pair = 0.25
    cp_max_plot_2D_per_pair = 0.55

    cp_plot_bin = 0.05

    power_min_plot_1D = 0
    power_max_plot_1D = 5000
    power_plot_bins = 500

    figsize_1D_per_turb = cm2inch(20, 8)
    figsize_1D_per_pair = cm2inch(10, 8)

    figsize_2D_per_turb = cm2inch(30, 16)
    figsize_2D_per_pair = cm2inch(16, 16)

    # --- plot 1D
    VLS = ':'
    VLW = 1
    ALPHA_ERR = 0.1
    MS = 4

    # --- plot norm power vs wspd
    var_key_to_plot = 'NormPower'
    var_label_y = 'Norm. power'
    var_unit_y = '-'
    var_key_to_bin_by = 'WSpeed'
    var_unit_x = 'm/s'
    bins_per_pair = wspd_bins_per_pair

    df_binned_1D_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict = \
        proc.bin_filt_ctrl_turb_dict_1D(
            df_filt_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            bins_per_pair,
            var_key_to_bin_by,
            bool_use_std_of_mean
        )

    # per turb
    plo.plot_binned_1D_per_turb(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_turb,
        cp_min_plot_1D_per_turb, cp_max_plot_1D_per_turb,
        cp_min_plot_1D_per_turb, cp_max_plot_1D_per_turb, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_limit_filtered,
    )

    # per pair
    plo.plot_binned_1D_per_pair(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_pair,
        cp_min_plot_1D_per_pair, cp_max_plot_1D_per_pair, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_limit_filtered,
        )

    # --- plot power vs wspd
    var_key_to_plot = 'Power'
    var_label_y = 'Power'
    var_unit_y = 'kW'

    # per turb
    plo.plot_binned_1D_per_turb(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_turb,
        power_min_plot_1D, power_max_plot_1D, power_min_plot_1D, power_max_plot_1D,
        power_plot_bins,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_limit_filtered,
    )

    # per pair
    plo.plot_binned_1D_per_pair(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_pair,
        power_min_plot_1D, power_max_plot_1D, power_plot_bins,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_limit_filtered,
        )

    # --- plot norm power vs wdir
    var_key_to_plot = 'NormPower'
    var_label_y = 'Norm. power'
    var_unit_y = '-'
    var_key_to_bin_by = 'WDir'
    var_unit_x = 'deg'
    bins_per_pair = wdir_bins_per_pair

    df_binned_1D_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict = \
        proc.bin_filt_ctrl_turb_dict_1D(
            df_filt_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            wdir_bins_per_pair,
            var_key_to_bin_by,
            bool_use_std_of_mean
        )

    # per turb
    plo.plot_binned_1D_per_turb(
            df_binned_1D_mean_dict,
            df_binned_1D_std_dict,
            bins_per_pair,
            var_key_to_bin_by,
            var_unit_x,
            var_key_to_plot,
            var_label_y,
            var_unit_y,
            turb_keys_split_by_pair,
            idx_upstream, idx_downstream,
            bool_save_fig,
            figsize_1D_per_turb,
            cp_min_plot_1D_per_turb, cp_max_plot_1D_per_turb,
            cp_min_plot_1D_per_turb, cp_max_plot_1D_per_turb, cp_plot_bin,
            VLS, VLW, MS, ALPHA_ERR,
            path2dir_fig_limit_filtered,
    )

    # --- plot 2D
    VLS = ':'
    VLW = 2
    ALPHA_ERR = 0.1
    MS = 6

    # --- plot norm power vs wspd
    var_key_to_plot = 'NormPower'
    var_label_y = 'Norm. Power'
    var_unit_y = '-'

    df_binned_2D_grouper_dict, df_binned_2D_mean_dict, df_binned_2D_std_dict = \
        proc.bin_filt_ctrl_turb_dict_2D(
            df_filt_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            wspd_bins_per_pair,
            wdir_bins_per_pair,
            'WSpeed',
            'WDir',
            bool_use_std_of_mean
        )

    # per turb
    plo.plot_binned_2D_per_turb(
        df_binned_2D_mean_dict,
        df_binned_2D_std_dict,
        wspd_bins_per_pair,
        wdir_bins_per_pair,
        wdir_setpoint_per_pair,
        yaw_setpoint_per_pair,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_2D_per_turb,
        cp_min_down_plot_2D_per_turb, cp_max_down_plot_2D_per_turb,
        cp_min_up_plot_2D_per_turb, cp_max_up_plot_2D_per_turb, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_limit_filtered,
    )

    # per pair
    plo.plot_binned_2D_per_pair(
        df_binned_2D_mean_dict,
        df_binned_2D_std_dict,
        wspd_bins_per_pair,
        wdir_bins_per_pair,
        wdir_setpoint_per_pair,
        yaw_setpoint_per_pair,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_2D_per_pair,
        cp_min_plot_2D_per_pair, cp_max_plot_2D_per_pair, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_limit_filtered,
    )

## PLOT 1D BINNED DATA, TIME INTERVAL FILTERED

if bool_plot_1D_binned_time_int_filtered_data:

    var_key_used_for_filter_binning = 'WDir'

    path2dir_fig_time_int_filtered_ext_1D = (
            path2dir_fig_time_int_filtered + os.sep + f'1D_{var_key_used_for_filter_binning}'
    )
    make_dir_if_not_exists(path2dir_fig_time_int_filtered_ext_1D)

    # ---
    cp_min_plot_1D_per_turb = 0.2
    cp_max_plot_1D_per_turb = 0.5
    cp_min_plot_1D_per_pair = 0.25
    cp_max_plot_1D_per_pair = 0.5

    cp_min_down_plot_2D_per_turb = 0.15
    cp_max_down_plot_2D_per_turb = 0.6
    cp_min_up_plot_2D_per_turb = 0.25
    cp_max_up_plot_2D_per_turb = 0.55

    cp_min_plot_2D_per_pair = 0.2
    cp_max_plot_2D_per_pair = 0.6

    cp_plot_bin = 0.05

    power_min_plot_1D = 0
    power_max_plot_1D = 5000
    power_plot_bins = 500

    figsize_1D_per_turb = cm2inch(20, 8)
    figsize_1D_per_pair = cm2inch(10, 8)

    figsize_2D_per_turb = cm2inch(30, 16)
    figsize_2D_per_pair = cm2inch(16, 22)

    # --- plot 1D
    VLS = ':'
    VLW = 1
    ALPHA_ERR = 0.1
    MS = 4

    # --- plot norm power vs wspd
    var_key_to_plot = 'NormPower'
    var_label_y = 'Norm. power'
    var_unit_y = '-'
    var_key_to_bin_by = 'WSpeed'
    var_unit_x = 'm/s'
    bins_per_pair = wspd_bins_per_pair

    df_binned_1D_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict = \
        proc.bin_filt_ctrl_turb_dict_1D(
            df_filt_time_int_1D_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            bins_per_pair,
            var_key_to_bin_by,
            bool_use_std_of_mean
        )

    # per turb
    plo.plot_binned_1D_per_turb(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_turb,
        cp_min_plot_1D_per_turb, cp_max_plot_1D_per_turb,
        cp_min_plot_1D_per_turb, cp_max_plot_1D_per_turb, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_1D,
    )

    # per pair
    plo.plot_binned_1D_per_pair(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_pair,
        cp_min_plot_1D_per_pair, cp_max_plot_1D_per_pair, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_1D,
        )

    # --- plot power vs wspd
    var_key_to_plot = 'Power'
    var_label_y = 'Power'
    var_unit_y = 'kW'

    # per turb
    plo.plot_binned_1D_per_turb(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_turb,
        power_min_plot_1D, power_max_plot_1D, power_min_plot_1D, power_max_plot_1D,
        power_plot_bins,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_1D,
    )

    # per pair
    plo.plot_binned_1D_per_pair(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_pair,
        power_min_plot_1D, power_max_plot_1D, power_plot_bins,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_1D,
        )

    # --- plot norm power vs wdir
    var_key_to_plot = 'NormPower'
    var_label_y = 'Norm. power'
    var_unit_y = '-'
    var_key_to_bin_by = 'WDir'
    var_unit_x = 'deg'
    bins_per_pair = wdir_bins_per_pair

    df_binned_1D_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict = \
        proc.bin_filt_ctrl_turb_dict_1D(
            df_filt_time_int_1D_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            wdir_bins_per_pair,
            var_key_to_bin_by,
            bool_use_std_of_mean
        )

    # per turb
    plo.plot_binned_1D_per_turb(
            df_binned_1D_mean_dict,
            df_binned_1D_std_dict,
            bins_per_pair,
            var_key_to_bin_by,
            var_unit_x,
            var_key_to_plot,
            var_label_y,
            var_unit_y,
            turb_keys_split_by_pair,
            idx_upstream, idx_downstream,
            bool_save_fig,
            figsize_1D_per_turb,
            cp_min_plot_1D_per_turb, cp_max_plot_1D_per_turb,
            cp_min_plot_1D_per_turb, cp_max_plot_1D_per_turb, cp_plot_bin,
            VLS, VLW, MS, ALPHA_ERR,
            path2dir_fig_time_int_filtered_ext_1D,
    )

    # per pair
    plo.plot_binned_1D_per_pair(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_to_bin_by,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_1D_per_pair,
        cp_min_plot_1D_per_pair, cp_max_plot_1D_per_pair, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_1D,
        )

    # --- plot 2D
    VLS = ':'
    VLW = 2
    ALPHA_ERR = 0.1
    MS = 6

    # --- plot norm power vs wspd
    var_key_used_for_filter_binning_1 = 'WSpeed'
    var_key_used_for_filter_binning_2 = 'WDir'
    var_key_to_plot = 'NormPower'
    var_label_y = 'Norm. Power'
    var_unit_y = '-'

    # careful here: the data was binned 1D, then time interval filtered. for plotting, the binning
    # is repeated in 2D
    df_binned_2D_grouper_dict, df_binned_2D_mean_dict, df_binned_2D_std_dict = \
        proc.bin_filt_ctrl_turb_dict_2D(
            df_filt_time_int_1D_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            wspd_bins_per_pair,
            wdir_bins_per_pair,
            var_key_used_for_filter_binning_1,
            var_key_used_for_filter_binning_2,
            bool_use_std_of_mean
        )

    # INFO: MAIN PLOTTING FUNCTIONS (2D)
    # per turb
    plo.plot_binned_2D_per_turb(
        df_binned_2D_mean_dict,
        df_binned_2D_std_dict,
        wspd_bins_per_pair,
        wdir_bins_per_pair,
        wdir_setpoint_per_pair,
        yaw_setpoint_per_pair,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_2D_per_turb,
        cp_min_down_plot_2D_per_turb, cp_max_down_plot_2D_per_turb,
        cp_min_up_plot_2D_per_turb, cp_max_up_plot_2D_per_turb, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_1D,
    )

    # per pair
    plo.plot_binned_2D_per_pair(
        df_binned_2D_mean_dict,
        df_binned_2D_std_dict,
        wspd_bins_per_pair,
        wdir_bins_per_pair,
        wdir_setpoint_per_pair,
        yaw_setpoint_per_pair,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_2D_per_pair,
        cp_min_plot_2D_per_pair, cp_max_plot_2D_per_pair, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_1D,
    )

## COUNT VALID TIME INTERVALS FOR 1D BINNED FILTERING

if bool_count_intervals_1D_and_plot:

    n_bin_dims = 1
    var_key_used_for_filter_binning_list = ['WDir']
    bins_per_pair_list = [wdir_bins_per_pair]

    int_cnt_dict_1D = proc.count_time_intervals_in_df_filt_dict(
        df_filt_time_int_1D_ctrl_turb_dict,
        n_bin_dims,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_per_pair_list,
        var_key_used_for_filter_binning_list,
        ctrl_keys,
        max_gap_duration_s,
        bool_use_std_of_mean
    )

    figsize_count_plot = cm2inch(12, 12)

    plo.plot_counted_intervals(
            int_cnt_dict_1D,
            n_bin_dims,
            bins_per_pair_list,
            var_key_used_for_filter_binning_list,
            ctrl_keys,
            n_turb_pairs,
            turb_keys_up,
            path2dir_fig_time_int_filtered,
            figsize_count_plot
    )

## PLOT TIME SERIES IN VALID TIME INTERVALS FOR 1D BINNED FILTERING

if bool_plot_1D_binned_valid_intervals:

    n_bin_dims = 1
    var_key_used_for_filter_binning_list = ['WDir']
    bins_per_pair_list = [wdir_bins_per_pair]

    bool_clear_interval_figs_from_dir = 0

    figsize_int_time_series = cm2inch(8, 8)

    plo.plot_valid_intervals_wdir_vs_t(
        df_filt_time_int_1D_ctrl_turb_dict,
        n_bin_dims,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_per_pair_list,
        var_key_used_for_filter_binning_list,
        wdir_bins_per_pair,
        ctrl_keys,
        max_gap_duration_s,
        bool_use_std_of_mean,
        path2dir_yaw_table,
        fname_yaw_table,
        path2dir_fig_int_base,
        bool_clear_interval_figs_from_dir,
        figsize_int_time_series,
        resample_interval_s
    )

## PLOT 2D BINNED, TIME INTERVAL FILTERED DATA

if bool_plot_2D_binned_time_int_filtered_data:

    var_key_used_for_filter_binning_1 = 'WSpeed'
    var_key_used_for_filter_binning_2 = 'WDir'

    path2dir_fig_time_int_filtered_ext_2D = (
            path2dir_fig_time_int_filtered + os.sep
            + f'2D_{var_key_used_for_filter_binning_1}_{var_key_used_for_filter_binning_2}'
    )
    make_dir_if_not_exists(path2dir_fig_time_int_filtered_ext_2D)

    cp_min_down_plot_2D_per_turb = 0.25
    cp_max_down_plot_2D_per_turb = 0.5
    cp_min_up_plot_2D_per_turb = 0.25
    cp_max_up_plot_2D_per_turb = 0.5

    cp_min_plot_2D_per_pair = 0.25
    cp_max_plot_2D_per_pair = 0.55

    cp_plot_bin = 0.05

    figsize_2D_per_turb = cm2inch(30, 16)
    figsize_2D_per_pair = cm2inch(16, 22)

    # --- plot 2D
    VLS = ':'
    VLW = 2
    ALPHA_ERR = 0.1
    MS = 6

    # --- plot norm power vs wspd
    var_key_to_plot = 'NormPower'
    var_label_y = 'Norm. Power'
    var_unit_y = '-'

    df_binned_2D_grouper_dict, df_binned_2D_mean_dict, df_binned_2D_std_dict = \
        proc.bin_filt_ctrl_turb_dict_2D(
            df_filt_time_int_2D_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            wspd_bins_per_pair,
            wdir_bins_per_pair,
            var_key_used_for_filter_binning_1,
            var_key_used_for_filter_binning_2,
            bool_use_std_of_mean
        )

    # per turb
    plo.plot_binned_2D_per_turb(
        df_binned_2D_mean_dict,
        df_binned_2D_std_dict,
        wspd_bins_per_pair,
        wdir_bins_per_pair,
        wdir_setpoint_per_pair,
        yaw_setpoint_per_pair,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_2D_per_turb,
        cp_min_down_plot_2D_per_turb, cp_max_down_plot_2D_per_turb,
        cp_min_up_plot_2D_per_turb, cp_max_up_plot_2D_per_turb, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_2D,
    )

    # per pair
    plo.plot_binned_2D_per_pair(
        df_binned_2D_mean_dict,
        df_binned_2D_std_dict,
        wspd_bins_per_pair,
        wdir_bins_per_pair,
        wdir_setpoint_per_pair,
        yaw_setpoint_per_pair,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize_2D_per_pair,
        cp_min_plot_2D_per_pair, cp_max_plot_2D_per_pair, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig_time_int_filtered_ext_2D,
    )

## COUNT VALID TIME INTERVALS FOR 2D BINNED FILTERING

if bool_count_intervals_2D_and_plot:

    n_bin_dims = 2
    var_key_used_for_filter_binning_list = ['WSpeed', 'WDir']
    bins_per_pair_list = [wspd_bins_per_pair, wdir_bins_per_pair]

    bool_clear_interval_figs_from_dir = 0

    int_cnt_dict_2D = proc.count_time_intervals_in_df_filt_dict(
        df_filt_time_int_2D_ctrl_turb_dict,
        n_bin_dims,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_per_pair_list,
        var_key_used_for_filter_binning_list,
        ctrl_keys,
        max_gap_duration_s,
        bool_use_std_of_mean
    )

    figsize_count_plot = cm2inch(12, 12)

    plo.plot_counted_intervals(
            int_cnt_dict_2D,
            n_bin_dims,
            bins_per_pair_list,
            var_key_used_for_filter_binning_list,
            ctrl_keys,
            n_turb_pairs,
            turb_keys_up,
            path2dir_fig_time_int_filtered,
            figsize_count_plot
    )

## PLOT TIME SERIES IN VALID TIME INTERVALS FOR 2D BINNED FILTERING

if bool_plot_2D_binned_valid_intervals:

    n_bin_dims = 2
    var_key_used_for_filter_binning_list = ['WSpeed', 'WDir']
    bins_per_pair_list = [wspd_bins_per_pair, wdir_bins_per_pair]

    bool_clear_interval_figs_from_dir = 0

    figsize_int_time_series = cm2inch(8, 8)

    plo.plot_valid_intervals_wdir_vs_t(
        df_filt_time_int_2D_ctrl_turb_dict,
        n_bin_dims,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_per_pair_list,
        var_key_used_for_filter_binning_list,
        wdir_bins_per_pair,
        ctrl_keys,
        max_gap_duration_s,
        bool_use_std_of_mean,
        path2dir_yaw_table,
        fname_yaw_table,
        path2dir_fig_int_base,
        bool_clear_interval_figs_from_dir,
        figsize_int_time_series,
        resample_interval_s
    )

## PROCESSED DATA PLOTS WITHOUT FILTERING

if bool_plot_unfiltered_data_overview or bool_plot_errorcode_counts:

    df_dict = proc.load_processed_data(
        turb_keys_to_process,
        start_date_list,
        end_date_list,
        path2dir_in_base,
        resample_interval_s
    )

if bool_plot_unfiltered_data_overview:

    wdir_min_overview = 286
    wdir_max_overview = 342

    plo.plot_df_turb_dict_overview(
        df_dict,
        var_keys_to_process,
        turb_keys_to_process,
        turb_keys_split_by_pair,
        resample_str,
        path2dir_fig,
        marker_dict_by_turb,
        color_dict_by_turb,
        wdir_min_overview,
        wdir_max_overview,
        dit_plot_unfilt,
        bool_plot_overview,
        bool_plot_overview_split,
        bool_plot_wdir_selected_range,
        bool_plot_wdir_vs_yaw,
        bool_plot_wdir_vs_yaw_selected_range
    )

if bool_plot_errorcode_counts:
    plo.plot_error_code_counts(
        df_dict,
        turb_keys_to_process,
        path2dir_data_base,
        path2dir_fig_base,
        resample_str,
        date_range_total_str,
        color_dict_by_turb
    )
    
############################################################
# INFO>>>
plo.plot_power_diff_vs_abs_yaw_mis(
    df_dict,
    turb_keys_split_by_pair,
    idx_upstream,
    idx_downstream, 
    bin_width=5.0,
    figsize=(8,6),
    bool_save_fig=bool_save_fig,
    path2dir_fig=path2dir_fig
)
############################################################
# INFO: PLOT YAW MISALIGNMENT VS WINDSPEED
# Plot wind speed vs yaw misalignment for error code 6
errorcode_val = 6
plo.plot_wspd_vs_yawmis(
    df_dict,
    turb_keys_to_process,
    turb_keys_split_by_pair,
    idx_upstream,
    idx_downstream,
    errorcode_val,
    bool_save_fig,
    path2dir_fig,
    figsize=(12, 8),
    marker_size=3,
    alpha=0.7,
)
############################################################

##
plt.close('all')
print('--- script finished:', os.path.basename(__file__), '---')
