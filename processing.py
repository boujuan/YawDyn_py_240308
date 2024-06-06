import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from plot_tools import make_dir_if_not_exists


## GENERAL FUNCTIONS

# INFO: Correct windspeed according to Vlaho's indications
def correct_wspd(wspd, date):
    if date < pd.Timestamp('2023-12-01'):
        return wspd * ((0.8274/0.9006) + 0.5645)
    else:
        return wspd

def gen_pandas_bin_keys_from_bins_1D(
    bins
):
    n_bins = bins.shape[0] - 1

    key_list = []

    for bin_n in range(n_bins):
        key_list.append(pd.Interval(
            left=bins[bin_n],
            right=bins[bin_n + 1],
            closed='right'
        ))

    return key_list


def gen_interval_plot_labels_from_bins(bins, closed='right'):

    n_bins = bins.shape[0] - 1
    labels = []

    for bin_n in range(n_bins):
        if closed == 'left':
            label = f'({bins[bin_n]:.1f}, {bins[bin_n + 1]:.1f}]'
        elif closed == 'right':
            label = f'[{bins[bin_n]:.1f}, {bins[bin_n + 1]:.1f})'

        labels.append(label)

    return labels


def gen_pandas_bin_keys_from_bins_2D(
    bins_1,
    bins_2
):
    n_bins_1 = bins_1.shape[0] - 1
    n_bins_2 = bins_2.shape[0] - 1

    key_list = []

    for bin_2_n in range(n_bins_2):
        for bin_1_n in range(n_bins_1):
            key_list.append(
                (
                    pd.Interval(
                    left=bins_1[bin_1_n],
                    right=bins_1[bin_1_n + 1],
                    closed='right'
                    ),
                    pd.Interval(
                    left=bins_2[bin_2_n],
                    right=bins_2[bin_2_n + 1],
                    closed='right'
                    )
                )
            )

    return key_list


def power_curve_ideal(u, cp, rho, A):
    # curve power in kW
    power = cp * 0.5 * rho * A * u**3
    return power / 1e3


def mask_first_n_rows(x, n_masked_rows):

    result = np.ones_like(x).astype(bool)
    # result[0:n_masked_rows] = False
    result[0:n_masked_rows] = 0

    return result


def mask_first_n_seconds(x, n_sec):

    time_series = x.index.to_series()

    t0 = time_series.iloc[0]

    mask = (time_series - t0).dt.total_seconds() > n_sec

    return mask


def add_norm_power_to_df_ctrl_turb_dict(
        df_filt_ctrl_turb_dict,
        turb_keys_split_by_pair,
        ctrl_keys,
        idx_upstream,
        rotor_diam,
        air_density
):

    rotor_rad = 0.5 * rotor_diam
    rotor_area = np.pi * rotor_rad**2
    c1 = 0.5 * air_density * rotor_area

    n_turb_pairs = len(turb_keys_split_by_pair)

    for ctrl_key in ctrl_keys:
        for pair_n in range(n_turb_pairs):

            turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
            turb_key_up = turb_keys_in_pair[idx_upstream]

            for turb_key in turb_keys_in_pair:

                df_filt_ctrl_turb_dict[ctrl_key][turb_key]['NormPower'] = (
                    1000 * df_filt_ctrl_turb_dict[ctrl_key][turb_key]['Power']
                    / (c1 * (df_filt_ctrl_turb_dict[ctrl_key][turb_key_up]['WSpeed'])**3)
                )


## YAW TABLE


def gen_yaw_table_interp(
        path2dir_yaw_table,
        fname_yaw_table
):

    yaw_table = np.loadtxt(path2dir_yaw_table + os.sep + fname_yaw_table, delimiter=',').T

    calc_yaw_setpoint_pair_1 = sp.interpolate.interp1d(
        yaw_table[0], yaw_table[1], assume_sorted=True)
    calc_yaw_setpoint_pair_2 = sp.interpolate.interp1d(
        yaw_table[0], yaw_table[2], assume_sorted=True)

    return calc_yaw_setpoint_pair_1, calc_yaw_setpoint_pair_2


def calc_yaw_table_setpoint_vs_wdir(
        path2dir_yaw_table,
        fname_yaw_table,
        wdir_setpoint_per_pair
):
    calc_yaw_setpoint_pairs = gen_yaw_table_interp(
        path2dir_yaw_table,
        fname_yaw_table,
    )

    yaw_setpoint_per_pair = []

    for pair_n in range(len(calc_yaw_setpoint_pairs)):
        yaw_setpoint_per_pair.append(
            calc_yaw_setpoint_pairs[pair_n](wdir_setpoint_per_pair[pair_n])
        )

    return yaw_setpoint_per_pair


## LOAD DATA


def load_processed_data(
        turb_keys_to_process,
        start_date_list,
        end_date_list,
        path2dir_in_base,
        resample_interval_s
):

    print('-- start loading processed data from disk')

    n_date_intervals_to_load = len(start_date_list)
    resample_str_to_load = f'{resample_interval_s:.0f}s'

    df_dict = {}

    for turb_key in turb_keys_to_process:

        print('- start loading processed data for', turb_key)

        df_list = []

        for date_int_n in range(n_date_intervals_to_load):
            start_date = start_date_list[date_int_n]
            end_date = end_date_list[date_int_n]

            id_str = start_date + '_' + end_date

            date_list = pd.date_range(start=start_date, end=end_date)
            date_list_str = date_list.astype(str)
            path2dir_in = path2dir_in_base + os.sep + resample_str_to_load

            for date_n, date_str in enumerate(date_list_str):
                fname = date_str + '_' + resample_str_to_load + '_' + turb_key
                df_ = pd.read_hdf(path2dir_in + os.sep + fname + '.h5')
                
                # INFO: Apply correction to 'WSpeed' column
                df_['WSpeed'] = df_.apply(lambda row: correct_wspd(row['WSpeed'], row.name), axis=1)
                
                df_list.append(df_)
                print('loaded', date_str)

        df_dict[turb_key] = pd.concat(df_list, keys=None, levels=None)

        # df_dict_list.append(df_dict)
    # for turb_key in turb_keys_to_process:

    print('-- finish loading processed data from disk')

    return df_dict


def load_filtered_data(
        turb_keys_to_process,
        path2dir_filtered,
        resample_interval_s,
):
    print('-- start loading filtered data from disk')
    print(path2dir_filtered)

    resample_str_to_load = f'{resample_interval_s:.0f}s'

    df_filt_ctrl_turb_dict = {
        'on': {},
        'off': {},
    }

    for ctrl_key in ['on', 'off']:
        for turb_key in turb_keys_to_process:

            fname = f'filtered_ctrl_{ctrl_key}_{resample_str_to_load}_{turb_key}'
            df_filt_ctrl_turb_dict[ctrl_key][turb_key] = \
                pd.read_hdf(path2dir_filtered + os.sep + fname + '.h5')

    print('-- finish loading filtered data from disk')

    return df_filt_ctrl_turb_dict


## DATA BINNING


def bin_df_1D_ext(df_to_bin, df_to_bin_by, bins, var_key):
    """
    Bin dataframe df_to_bin with mask created with another (external) dataframe df_to_bin_by
    :param df_to_bin:
    :param df_to_bin_by:
    :param bins_1:
    :param bins_2:
    :param var_key_1:
    :param var_key_2:
    :return:
    """

    bin_mask = pd.cut(df_to_bin_by[var_key], bins).dropna()

    time_mask = df_to_bin.index.to_series().isin(bin_mask.index)

    # keys = df_to_bin[time_mask].groupby(bin_mask, observed=False, dropna=True).groups.keys()
    # keys = list(keys)
    #
    # grp = df_to_bin[time_mask].groupby(bin_mask, observed=False, dropna=True)

    return df_to_bin[time_mask].groupby(bin_mask, observed=False, dropna=True)


def bin_df_2D_ext(df_to_bin, df_to_bin_by, bins_1, bins_2, var_key_1, var_key_2):
    """
    Bin dataframe df_to_bin with mask created with another (external) dataframe df_to_bin_by
    :param df_to_bin:
    :param df_to_bin_by:
    :param bins_1:
    :param bins_2:
    :param var_key_1:
    :param var_key_2:
    :return:
    """

    bin_mask_1 = pd.cut(df_to_bin_by[var_key_1], bins_1).dropna()
    bin_mask_2 = pd.cut(df_to_bin_by[var_key_2], bins_2).dropna()

    time_mask_1 = df_to_bin.index.to_series().isin(bin_mask_1.index)
    time_mask_2 = df_to_bin.index.to_series().isin(bin_mask_2.index)

    time_mask = time_mask_1 & time_mask_2

    aa = df_to_bin[time_mask].groupby([bin_mask_1, bin_mask_2], observed=False, dropna=True)

    return df_to_bin.groupby([bin_mask_1, bin_mask_2], observed=False, dropna=True)


def bin_filt_ctrl_turb_dict_1D(
        df_filt_ctrl_turb_dict,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        idx_upstream,
        bins_per_pair,
        var_key,
        bool_use_std_of_mean
):
    n_turb_pairs = len(turb_keys_split_by_pair)

    df_binned_1D_grouper_dict = {
        'on': {},
        'off': {},
    }

    df_binned_1D_mean_dict = {
        'on': {},
        'off': {},
    }

    df_binned_1D_std_dict = {
        'on': {},
        'off': {},
    }

    for ctrl_key in ['on', 'off']:
        for pair_n in range(n_turb_pairs):

            turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
            turb_key_up = turb_keys_in_pair[idx_upstream]

            df_x_up = df_filt_ctrl_turb_dict[ctrl_key][turb_key_up]

            bins = bins_per_pair[pair_n]

            for turb_key in turb_keys_in_pair:

                df_x = df_filt_ctrl_turb_dict[ctrl_key][turb_key][var_keys_to_bin]

                df_binned_1D_grouper_dict[ctrl_key][turb_key] = bin_df_1D_ext(
                    df_x, df_x_up, bins, var_key)

                df_binned_1D_mean_dict[ctrl_key][turb_key] = \
                    df_binned_1D_grouper_dict[ctrl_key][turb_key].mean()

                df_binned_1D_std_dict[ctrl_key][turb_key] = \
                    df_binned_1D_grouper_dict[ctrl_key][turb_key].std()

                if bool_use_std_of_mean:

                    count_x = df_binned_1D_grouper_dict[ctrl_key][turb_key].count()
                    df_binned_1D_std_dict[ctrl_key][turb_key] = \
                        df_binned_1D_std_dict[ctrl_key][turb_key] / np.sqrt(count_x)

    return df_binned_1D_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict


def bin_filt_ctrl_turb_dict_2D(
        df_filt_ctrl_turb_dict,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        idx_upstream,
        bins_1_per_pair,
        bins_2_per_pair,
        var_key_1,
        var_key_2,
        bool_use_std_of_mean
):
    n_turb_pairs = len(turb_keys_split_by_pair)

    df_binned_2D_grouper_dict = {
        'on': {},
        'off': {},
    }

    df_binned_2D_mean_dict = {
        'on': {},
        'off': {},
    }

    df_binned_2D_std_dict = {
        'on': {},
        'off': {},
    }

    for ctrl_key in ['on', 'off']:
        for pair_n in range(n_turb_pairs):

            turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
            turb_key_up = turb_keys_in_pair[idx_upstream]

            df_x_up = df_filt_ctrl_turb_dict[ctrl_key][turb_key_up]

            bins_1 = bins_1_per_pair[pair_n]
            bins_2 = bins_2_per_pair[pair_n]

            for turb_key in turb_keys_in_pair:

                df_x = df_filt_ctrl_turb_dict[ctrl_key][turb_key][var_keys_to_bin]

                df_binned_2D_grouper_dict[ctrl_key][turb_key] = bin_df_2D_ext(
                    df_x, df_x_up, bins_1, bins_2, var_key_1, var_key_2
                )

                df_binned_2D_mean_dict[ctrl_key][turb_key] = \
                    df_binned_2D_grouper_dict[ctrl_key][turb_key].mean()

                df_binned_2D_std_dict[ctrl_key][turb_key] = \
                    df_binned_2D_grouper_dict[ctrl_key][turb_key].std()

                if bool_use_std_of_mean:

                    count_x = df_binned_2D_grouper_dict[ctrl_key][turb_key].count()
                    df_binned_2D_std_dict[ctrl_key][turb_key] = \
                        df_binned_2D_std_dict[ctrl_key][turb_key] / np.sqrt(count_x)

    return df_binned_2D_grouper_dict, df_binned_2D_mean_dict, df_binned_2D_std_dict


## DATA FILTERING


def filter_data_for_limits(
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
        resample_interval_s=0,
        path2dir_filtered=''
):
    """

    :param df_dict: df_dict['T3'] = pandas dataframe with data for T3 etc.
    time_index_min_dict: index_min_dict['T4'] = '2023-06-29 08:00:00'
    = data will only be considered after this time
    turb_keys_split_by_pair: [['T3', 'T4'], ['T5', 'T6']]
    idx_upstream: index of the upstream turbine in each sublist of turb_keys_split_by_pair
    idx_downstream: index of the downstream turbine in each sublist of turb_keys_split_by_pair
    :return:
    """

    print('-- start filtering data for limits')

    ctrl_keys = ['on', 'off']
    
    filter_mask_ctrl_on_list = []
    filter_mask_ctrl_off_list = []
    # filter_mask_ctrl_combo_list = []

    n_turb_pairs = len(turb_keys_split_by_pair)

    for pair_n in range(n_turb_pairs):

        turb_key_up = turb_keys_split_by_pair[pair_n][idx_upstream]
        turb_key_down = turb_keys_split_by_pair[pair_n][idx_downstream]

        wdir_min = wdir_min_per_pair[pair_n]
        wdir_max = wdir_max_per_pair[pair_n]

        # --- filter turbine pair

        # --- start from minimum time index
        time_index_min_up = time_index_min_dict[turb_key_up]
        time_index_min_down = time_index_min_dict[turb_key_down]

        if time_index_min_up is None:
            time_index_min_up = '1000-01-01 00:00:00'
        if time_index_min_down is None:
            time_index_min_down = '1000-01-01 00:00:00'

        filter_mask_date = df_dict[turb_key_up].index >= time_index_min_up
        filter_mask_date = np.logical_and(
            filter_mask_date, df_dict[turb_key_down].index >= time_index_min_down
        )

        # --- filter upstream turbine for wind direction and wind speed

        # wdir of proc. data is in [0, 360).
        # if wdir_max > 360, need to split into two intervals
        if wdir_max >= 360.0:
            wdir_max1 = 360.0
            wdir_max2 = wdir_max - 360.0
            wdir_min1 = wdir_min
            wdir_min2 = 0.0

            filter_mask_pair_a = (
                df_dict[turb_key_up]['WDir'].between(wdir_min1, wdir_max1, inclusive='both')
            )

            filter_mask_pair_b = (
                df_dict[turb_key_up]['WDir'].between(wdir_min2, wdir_max2, inclusive='both')
            )

            filter_mask_pair = (filter_mask_pair_a | filter_mask_pair_b)

        else:
            filter_mask_pair = (
                df_dict[turb_key_up]['WDir'].between(wdir_min, wdir_max, inclusive='both')
            )

        filter_mask_pair = (
                filter_mask_pair
                & df_dict[turb_key_up]['WSpeed'].between(wspd_min, wspd_max, inclusive='both')
        )

        filter_mask_pair = (filter_mask_pair & filter_mask_date)

        # --- filter upstream and downstream for power, errorcode and pitch

        for turb_key in [turb_key_up, turb_key_down]:

            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Power'].between(power_min, power_max, inclusive='both')
            )
            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['PowerRef'].between(
                powerref_min, powerref_max, inclusive='both')
            )
            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Errorcode'].between(errorcode_min, errorcode_max,
                                                             inclusive='both')
            )
            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Pitch'].between(pitch_min, pitch_max, inclusive='both')
            )

        # --- split into yaw control on and off
        df_dict[turb_key_up]['ControlSwitch'][df_dict[turb_key_up]['ControlSwitch'] < -0.1] = np.nan

        control_switch_mask_pair = (
                ~df_dict[turb_key_up]['ControlSwitch'].ffill().isna()
                & df_dict[turb_key_up]['ControlSwitch'].ffill().astype(bool)
        )

        # --- control on only
        filter_mask_ctrl_on_pair = (filter_mask_pair & control_switch_mask_pair)

        # --- control off only
        filter_mask_ctrl_off_pair = (filter_mask_pair & ~control_switch_mask_pair)

        # --- append
        filter_mask_ctrl_on_list.append(filter_mask_ctrl_on_pair)
        filter_mask_ctrl_off_list.append(filter_mask_ctrl_off_pair)

        # filter_mask_ctrl_combo_list.append(filter_mask_ctrl_on_pair | filter_mask_ctrl_off_pair)

    # df_filt_ctrl_combo_dict = {}
    df_filt_ctrl_turb_dict = {
        'on': {},
        'off': {}
    }

    for ctrl_key in ctrl_keys:

        for pair_n in range(n_turb_pairs):

            for turb_key in turb_keys_split_by_pair[pair_n]:

                if ctrl_key == 'on':
                    df_filt_ctrl_turb_dict[ctrl_key][turb_key] = \
                        df_dict[turb_key][filter_mask_ctrl_on_list[pair_n]]

                elif ctrl_key == 'off':
                    df_filt_ctrl_turb_dict[ctrl_key][turb_key] = \
                        df_dict[turb_key][filter_mask_ctrl_off_list[pair_n]]

                # df_filt_ctrl_combo_dict[turb_key] = \
                #     df_dict[turb_key][filter_mask_ctrl_combo_list[pair_n]]

                if bool_save_filtered_data:

                    resample_str = f'{resample_interval_s:.0f}s'

                    fname = f'filtered_ctrl_{ctrl_key}_' + resample_str + '_' + turb_key
                    df_filt_ctrl_turb_dict[ctrl_key][turb_key].to_hdf(
                        path2dir_filtered + os.sep + fname + '.h5',
                        'scada', mode='w'
                    )

                    # fname = 'filtered_ctrl_on_' + resample_str + '_' + turb_key
                    # df_filt_ctrl_on_dict[turb_key].to_hdf(
                    #     path2dir_filtered + os.sep + fname + '.h5',
                    #     'scada', mode='w'
                    # )
                    #
                    # fname = 'filtered_ctrl_off_' + resample_str + '_' + turb_key
                    # df_filt_ctrl_off_dict[turb_key].to_hdf(
                    #     path2dir_filtered + os.sep + fname + '.h5',
                    #     'scada', mode='w'
                    # )
                    #
                    # fname = 'filtered_ctrl_combo_' + resample_str + '_' + turb_key
                    # df_filt_ctrl_combo_dict[turb_key].to_hdf(
                    #     path2dir_filtered + os.sep + fname + '.h5',
                    #     'scada', mode='w'
                    # )

                print('saved filtered data to hard drive for', turb_key, ctrl_key)

    print('-- finish filtering data for limits')

    return df_filt_ctrl_turb_dict


def filter_data_for_time_int_1D_binned(
        df_filt_ctrl_turb_dict,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_to_bin_by_per_pair,
        var_key_to_bin_by,
        bool_use_std_of_mean,
        resample_interval_s,
        max_gap_duration_s,
        min_interval_duration_s,
        discard_time_at_beginning_s,
        ctrl_keys,
        path2dir_time_int_filtered
):

    print('-- start filtering data for time intervals 1D binned')

    n_turb_pairs = len(turb_keys_split_by_pair)
    resample_str_to_apply = f'{resample_interval_s:.0f}s'

    # if resample_interval_s > 0:
    #     n_first_rows_to_mask = \
    #         int(discard_time_at_beginning_s / resample_interval_s)

    df_binned_1D_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict = \
        bin_filt_ctrl_turb_dict_1D(
            df_filt_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            bins_to_bin_by_per_pair,
            var_key_to_bin_by,
            bool_use_std_of_mean
        )

    df_filt_time_int_ctrl_turb_dict = {
        'on': {},
        'off': {},
    }

    for ctrl_key in ctrl_keys:

        for pair_n in range(n_turb_pairs):

            turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
            turb_key_up = turb_keys_up[pair_n]
            turb_key_down = turb_keys_down[pair_n]

            group_keys = list(df_binned_1D_grouper_dict[ctrl_key][turb_key_up].groups.keys())

            group_keys = [gk for gk in group_keys if not pd.isna(gk)]

            # remove empty keys:
            group_counter = df_binned_1D_grouper_dict[ctrl_key][turb_key_up].count()

            group_keys = [
                gk for gk in group_keys if group_counter.loc[gk].iloc[0] > 0
            ]

            df_up_list_per_group_key = []
            df_down_list_per_group_key = []

            # count_up = 0
            # count_down = 0

            for group_key in group_keys:

                df_up = df_binned_1D_grouper_dict[ctrl_key][turb_key_up].get_group(group_key)
                df_down = df_binned_1D_grouper_dict[ctrl_key][turb_key_down].get_group(group_key)

                index_for_filt = df_up.index.to_series()
                grouping_mask_time_intervals_pair = (
                        index_for_filt.diff().dt.total_seconds().bfill() > max_gap_duration_s
                ).cumsum()

                mask_time_intervals_pair = (
                        index_for_filt.groupby(
                            grouping_mask_time_intervals_pair
                        ).transform('count')
                        >= (min_interval_duration_s / resample_interval_s)
                )

                grouping_mask_time_intervals_pair = \
                    grouping_mask_time_intervals_pair[mask_time_intervals_pair]

                grouped_index_mask = grouping_mask_time_intervals_pair.groupby(
                        grouping_mask_time_intervals_pair
                    )

                mask_int_begin = grouped_index_mask.transform(
                    mask_first_n_seconds, discard_time_at_beginning_s)

                # mask_int_begin = \
                #     grouping_mask_time_intervals_pair.groupby(
                #         grouping_mask_time_intervals_pair
                #     ).transform(mask_first_n_rows, n_first_rows_to_mask)

                mask_time_intervals_pair = (mask_time_intervals_pair & mask_int_begin)

                df_up = df_up[mask_time_intervals_pair]
                df_down = df_down[mask_time_intervals_pair]

                df_up_list_per_group_key.append(df_up)
                df_down_list_per_group_key.append(df_down)

                # count_up += df_up.shape[0]
                # count_down += df_down.shape[0]

            df_up_concat = pd.concat(df_up_list_per_group_key)
            df_down_concat = pd.concat(df_down_list_per_group_key)

            df_up_concat.sort_index(inplace=True)
            df_down_concat.sort_index(inplace=True)

            df_filt_time_int_ctrl_turb_dict[ctrl_key][turb_key_down] = df_down_concat
            df_filt_time_int_ctrl_turb_dict[ctrl_key][turb_key_up] = df_up_concat

            resample_str = resample_str_to_apply
            path2dir_filtered = (
                    path2dir_time_int_filtered + os.sep
                    + f'1D_{var_key_to_bin_by}'
            )

            make_dir_if_not_exists(path2dir_filtered)

            fname = f'filtered_ctrl_{ctrl_key}_' + resample_str + '_' + turb_key_up
            df_up_concat.to_hdf(
                path2dir_filtered + os.sep + fname + '.h5',
                'scada', mode='w'
            )
            fname = f'filtered_ctrl_{ctrl_key}_' + resample_str + '_' + turb_key_down
            df_down_concat.to_hdf(
                path2dir_filtered + os.sep + fname + '.h5',
                'scada', mode='w'
            )
    print('-- finish filtering data for time intervals 1D binned')

    return df_filt_time_int_ctrl_turb_dict


def filter_data_for_time_int_2D_binned(
        df_filt_ctrl_turb_dict,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_to_bin_by_1_per_pair,
        bins_to_bin_by_2_per_pair,
        var_key_to_bin_by_1,
        var_key_to_bin_by_2,
        bool_use_std_of_mean,
        resample_interval_s,
        max_gap_duration_s,
        min_interval_duration_s,
        discard_time_at_beginning_s,
        ctrl_keys,
        path2dir_time_int_filtered
):

    print('-- start filtering data for time intervals 2D binned')

    n_turb_pairs = len(turb_keys_split_by_pair)
    resample_str_to_apply = f'{resample_interval_s:.0f}s'

    df_binned_2D_grouper_dict, df_binned_2D_mean_dict, df_binned_2D_std_dict = \
        bin_filt_ctrl_turb_dict_2D(
            df_filt_ctrl_turb_dict,
            var_keys_to_bin,
            turb_keys_split_by_pair,
            idx_upstream,
            bins_to_bin_by_1_per_pair,
            bins_to_bin_by_2_per_pair,
            var_key_to_bin_by_1,
            var_key_to_bin_by_2,
            bool_use_std_of_mean
        )

    df_filt_time_int_ctrl_turb_dict = {
        'on': {},
        'off': {},
    }

    for ctrl_key in ctrl_keys:

        for pair_n in range(n_turb_pairs):

            turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
            turb_key_up = turb_keys_up[pair_n]
            turb_key_down = turb_keys_down[pair_n]

            group_keys = list(df_binned_2D_grouper_dict[ctrl_key][turb_key_up].groups.keys())

            group_keys = [gk for gk in group_keys if not pd.isna(gk[0]) and not pd.isna(gk[1])]

            # remove empty keys:
            group_counter = df_binned_2D_grouper_dict[ctrl_key][turb_key_up].count()

            group_keys = [
                gk for gk in group_keys if group_counter.loc[gk].iloc[0] > 0
            ]

            df_up_list_per_group_key = []
            df_down_list_per_group_key = []

            # count_up = 0
            # count_down = 0

            for group_key in group_keys:

                df_up = df_binned_2D_grouper_dict[ctrl_key][turb_key_up].get_group(group_key)
                df_down = df_binned_2D_grouper_dict[ctrl_key][turb_key_down].get_group(group_key)

                index_for_filt = df_up.index.to_series()
                grouping_mask_time_intervals_pair = (
                        index_for_filt.diff().dt.total_seconds().bfill() > max_gap_duration_s
                ).cumsum()

                mask_time_intervals_pair = (
                        index_for_filt.groupby(
                            grouping_mask_time_intervals_pair
                        ).transform('count')
                        >= (min_interval_duration_s / resample_interval_s)
                )

                grouping_mask_time_intervals_pair = \
                    grouping_mask_time_intervals_pair[mask_time_intervals_pair]

                grouped_index_mask = grouping_mask_time_intervals_pair.groupby(
                        grouping_mask_time_intervals_pair
                    )

                mask_int_begin = grouped_index_mask.transform(
                    mask_first_n_seconds, discard_time_at_beginning_s)

                # mask_int_begin = \
                #     grouping_mask_time_intervals_pair.groupby(
                #         grouping_mask_time_intervals_pair
                #     ).transform(mask_first_n_rows, n_first_rows_to_mask)

                mask_time_intervals_pair = (mask_time_intervals_pair & mask_int_begin)

                df_up = df_up[mask_time_intervals_pair]
                df_down = df_down[mask_time_intervals_pair]

                df_up_list_per_group_key.append(df_up)
                df_down_list_per_group_key.append(df_down)

                # count_up += df_up.shape[0]
                # count_down += df_down.shape[0]

            df_up_concat = pd.concat(df_up_list_per_group_key)
            df_down_concat = pd.concat(df_down_list_per_group_key)

            df_up_concat.sort_index(inplace=True)
            df_down_concat.sort_index(inplace=True)

            df_filt_time_int_ctrl_turb_dict[ctrl_key][turb_key_down] = df_down_concat
            df_filt_time_int_ctrl_turb_dict[ctrl_key][turb_key_up] = df_up_concat

            resample_str = resample_str_to_apply
            path2dir_filtered = (
                    path2dir_time_int_filtered + os.sep
                    + f'2D_{var_key_to_bin_by_1}_{var_key_to_bin_by_2}'
            )

            make_dir_if_not_exists(path2dir_filtered)

            fname = f'filtered_ctrl_{ctrl_key}_' + resample_str + '_' + turb_key_up
            df_up_concat.to_hdf(
                path2dir_filtered + os.sep + fname + '.h5',
                'scada', mode='w'
            )
            fname = f'filtered_ctrl_{ctrl_key}_' + resample_str + '_' + turb_key_down
            df_down_concat.to_hdf(
                path2dir_filtered + os.sep + fname + '.h5',
                'scada', mode='w'
            )
    print('-- finish filtering data for time intervals 2D binned')

    return df_filt_time_int_ctrl_turb_dict


def filter_data_incl_valid_time_intervals(
        df_dict,
        wdir_min, wdir_max,
        power_min, power_max,
        wspd_min, wspd_max,
        errorcode_min, errorcode_max,
        pitch_min, pitch_max,
        time_index_min_dict,
        turb_keys_split,
        idx_upstream, idx_downstream,
        bool_save_filtered_data,
        resample_interval_s=0,
        path2dir_filtered=''
):
    """

    :param df_dict: df_dict['T3'] = pandas dataframe with data for T3 etc.
    time_index_min_dict: index_min_dict['T4'] = '2023-06-29 08:00:00' =
    data will only be considered after this time
    turb_keys_split: [['T3', 'T4'], ['T5', 'T6']]
    idx_upstream: index of the upstream turbine in each sublist of turb_keys_split
    idx_downstream: index of the downstream turbine in each sublist of turb_keys_split
    :return:
    """

    print('start filtering data')

    filter_mask_ctrl_on_list = []
    filter_mask_ctrl_off_list = []
    filter_mask_ctrl_combo_list = []

    n_turb_pairs = len(turb_keys_split)

    for pair_n in range(n_turb_pairs):

        turb_key_up = turb_keys_split[pair_n][idx_upstream]
        turb_key_down = turb_keys_split[pair_n][idx_downstream]

        # --- filter turbine pair

        # --- start from minimum time index
        time_index_min_up = time_index_min_dict[turb_key_up]
        time_index_min_down = time_index_min_dict[turb_key_down]

        if time_index_min_up is None:
            time_index_min_up = '1000-01-01 00:00:00'
        if time_index_min_down is None:
            time_index_min_down = '1000-01-01 00:00:00'

        filter_mask_date = df_dict[turb_key_up].index >= time_index_min_up
        filter_mask_date = np.logical_and(
            filter_mask_date, df_dict[turb_key_down].index >= time_index_min_down
        )

        # --- filter upstream turbine for wind direction and wind speed

        # wdir of proc. data is in [0, 360).
        # if wdir_max > 360, need to split into two intervals
        if wdir_max >= 360.0:
            wdir_max1 = 360.0
            wdir_max2 = wdir_max - 360.0
            wdir_min1 = wdir_min
            wdir_min2 = 0.0

            filter_mask_pair_a = (
                df_dict[turb_key_up]['WDir'].between(wdir_min1, wdir_max1, inclusive='both')
            )

            filter_mask_pair_b = (
                df_dict[turb_key_up]['WDir'].between(wdir_min2, wdir_max2, inclusive='both')
            )

            filter_mask_pair = (filter_mask_pair_a | filter_mask_pair_b)

        else:
            filter_mask_pair = (
                df_dict[turb_key_up]['WDir'].between(wdir_min, wdir_max, inclusive='both')
            )

        filter_mask_pair = (
                filter_mask_pair
                & df_dict[turb_key_up]['WSpeed'].between(wspd_min, wspd_max, inclusive='both')
        )

        filter_mask_pair = (filter_mask_pair & filter_mask_date)

        # --- filter upstream and downstream for power, errorcode and pitch

        for turb_key in [turb_key_up, turb_key_down]:

            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Power'].between(power_min, power_max, inclusive='both')
            )
            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Errorcode'].between(errorcode_min, errorcode_max,
                                                             inclusive='both')
            )
            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Pitch'].between(pitch_min, pitch_max, inclusive='both')
            )

        # --- split into yaw control on and off
        df_dict[turb_key_up]['ControlSwitch'][df_dict[turb_key_up]['ControlSwitch'] < -0.1] \
            = np.nan

        control_switch_mask_pair = (
                ~df_dict[turb_key_up]['ControlSwitch'].ffill().isna()
                & df_dict[turb_key_up]['ControlSwitch'].ffill().astype(bool)
        )

        # --- control on only
        filter_mask_ctrl_on_pair = (filter_mask_pair & control_switch_mask_pair)

        # --- control off only
        filter_mask_ctrl_off_pair = (filter_mask_pair & ~control_switch_mask_pair)

        # --- append
        filter_mask_ctrl_on_list.append(filter_mask_ctrl_on_pair)
        filter_mask_ctrl_off_list.append(filter_mask_ctrl_off_pair)

        filter_mask_ctrl_combo_list.append(filter_mask_ctrl_on_pair | filter_mask_ctrl_off_pair)

    df_filt_ctrl_on_dict = {}
    df_filt_ctrl_off_dict = {}
    df_filt_ctrl_combo_dict = {}

    for pair_n in range(n_turb_pairs):
        for turb_key in turb_keys_split[pair_n]:
            df_filt_ctrl_on_dict[turb_key] = \
                df_dict[turb_key][filter_mask_ctrl_on_list[pair_n]]
            df_filt_ctrl_off_dict[turb_key] = \
                df_dict[turb_key][filter_mask_ctrl_off_list[pair_n]]
            df_filt_ctrl_combo_dict[turb_key] = \
                df_dict[turb_key][filter_mask_ctrl_combo_list[pair_n]]

            if bool_save_filtered_data:

                resample_str = f'{resample_interval_s:.0f}s'

                fname = 'filtered_ctrl_on_' + resample_str + '_' + turb_key
                df_filt_ctrl_on_dict[turb_key].to_hdf(
                    path2dir_filtered + os.sep + fname + '.h5',
                    'scada', mode='w'
                )

                fname = 'filtered_ctrl_off_' + resample_str + '_' + turb_key
                df_filt_ctrl_off_dict[turb_key].to_hdf(
                    path2dir_filtered + os.sep + fname + '.h5',
                    'scada', mode='w'
                )

                fname = 'filtered_ctrl_combo_' + resample_str + '_' + turb_key
                df_filt_ctrl_combo_dict[turb_key].to_hdf(
                    path2dir_filtered + os.sep + fname + '.h5',
                    'scada', mode='w'
                )

            print('saved filtered data to hard drive for ctrl', turb_key)

    print('finish filtering data')

    return df_filt_ctrl_on_dict, df_filt_ctrl_off_dict, df_filt_ctrl_combo_dict


## TIME INTERVAL SPLITTING


def split_df_filt_dict_into_time_intervals(
        df_dict,
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
):

    n_turb_pairs = len(turb_keys_split_by_pair)

    if n_bin_dims == 1:

        bins_per_pair = bins_per_pair_list[0]
        var_key_used_for_filter_binning = var_key_used_for_filter_binning_list[0]

        df_binned_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict = \
            bin_filt_ctrl_turb_dict_1D(
                df_dict,
                var_keys_to_bin,
                turb_keys_split_by_pair,
                idx_upstream,
                bins_per_pair,
                var_key_used_for_filter_binning,
                bool_use_std_of_mean
            )

    elif n_bin_dims == 2:

        bins_1_per_pair = bins_per_pair_list[0]
        bins_2_per_pair = bins_per_pair_list[1]

        var_key_1_used_for_filter_binning = var_key_used_for_filter_binning_list[0]
        var_key_2_used_for_filter_binning = var_key_used_for_filter_binning_list[1]

        df_binned_grouper_dict, df_binned_2D_mean_dict, df_binned_2D_std_dict = \
            bin_filt_ctrl_turb_dict_2D(
                df_dict,
                var_keys_to_bin,
                turb_keys_split_by_pair,
                idx_upstream,
                bins_1_per_pair,
                bins_2_per_pair,
                var_key_1_used_for_filter_binning,
                var_key_2_used_for_filter_binning,
                bool_use_std_of_mean
            )

    interval_dict = {
        'on': {},
        'off': {},
    }

    for ctrl_key in ctrl_keys:

        for pair_n in range(n_turb_pairs):

            turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
            turb_key_up = turb_keys_up[pair_n]
            turb_key_down = turb_keys_down[pair_n]

            group_keys = list(df_binned_grouper_dict[ctrl_key][turb_key_up].groups.keys())

            if n_bin_dims == 1:

                n_bins_1 = bins_per_pair[pair_n].shape[0]
                group_keys = [gk for gk in group_keys if not pd.isna(gk)]

            elif n_bin_dims == 2:

                group_keys = [
                    gk for gk in group_keys if not pd.isna(gk[0]) and not pd.isna(gk[1])]

                n_bins_1 = bins_1_per_pair[pair_n].shape[0]
                # n_bins_2 = bins_2_per_pair[pair_n].shape[0]

            # remove empty keys:
            group_counter = df_binned_grouper_dict[ctrl_key][turb_key_up].count()

            group_keys = [
                gk for gk in group_keys if group_counter.loc[gk].iloc[0] > 0
            ]

            interval_dict[ctrl_key][turb_key_up] = []
            interval_dict[ctrl_key][turb_key_down] = []

            # bin_n_1 = 0
            # bin_n_2 = 0

            for group_n, group_key in enumerate(group_keys):

                if n_bin_dims == 1:

                    bin_n_1 = group_n

                elif n_bin_dims == 2:

                    bin_n_1 = group_n % n_bins_1
                    bin_n_2 = group_n // n_bins_1

                df_up = df_binned_grouper_dict[ctrl_key][turb_key_up].get_group(group_key)
                df_down = df_binned_grouper_dict[ctrl_key][turb_key_down].get_group(group_key)

                index_for_filt = df_up.index.to_series()
                grouping_mask_time_intervals_pair = (
                        index_for_filt.diff().dt.total_seconds().bfill() > max_gap_duration_s
                ).cumsum()

                for k, v in df_up.groupby(grouping_mask_time_intervals_pair):
                    interval_dict[ctrl_key][turb_key_up].append(v)

                for k, v in df_down.groupby(grouping_mask_time_intervals_pair):
                    interval_dict[ctrl_key][turb_key_down].append(v)

                group_n += 1

    return interval_dict


def split_df_filt_dict_into_time_intervals_BAK(
        df_dict,
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
):

    n_turb_pairs = len(turb_keys_split_by_pair)

    int_cnt_dict = {
        'on': {},
        'off': {},
    }

    if n_bin_dims == 1:

        bins_per_pair = bins_per_pair_list[0]
        var_key_used_for_filter_binning = var_key_used_for_filter_binning_list[0]

        for ctrl_key in ctrl_keys:

            for pair_n in range(n_turb_pairs):
                int_cnt_arr = np.zeros((bins_per_pair[pair_n].shape[0] - 1))

                for turb_key in turb_keys_split_by_pair[pair_n]:
                    int_cnt_dict[ctrl_key][turb_key] = int_cnt_arr.copy()

        df_binned_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict = \
            bin_filt_ctrl_turb_dict_1D(
                df_dict,
                var_keys_to_bin,
                turb_keys_split_by_pair,
                idx_upstream,
                bins_per_pair,
                var_key_used_for_filter_binning,
                bool_use_std_of_mean
            )

    elif n_bin_dims == 2:

        bins_1_per_pair = bins_per_pair_list[0]
        bins_2_per_pair = bins_per_pair_list[1]

        var_key_1_used_for_filter_binning = var_key_used_for_filter_binning_list[0]
        var_key_2_used_for_filter_binning = var_key_used_for_filter_binning_list[1]

        for ctrl_key in ctrl_keys:

            for pair_n in range(n_turb_pairs):
                int_cnt_arr = np.zeros((
                    bins_1_per_pair[pair_n].shape[0] - 1,
                    bins_2_per_pair[pair_n].shape[0] - 1
                ))

                for turb_key in turb_keys_split_by_pair[pair_n]:
                    int_cnt_dict[ctrl_key][turb_key] = int_cnt_arr.copy()

        df_binned_grouper_dict, df_binned_2D_mean_dict, df_binned_2D_std_dict = \
            bin_filt_ctrl_turb_dict_2D(
                df_dict,
                var_keys_to_bin,
                turb_keys_split_by_pair,
                idx_upstream,
                bins_1_per_pair,
                bins_2_per_pair,
                var_key_1_used_for_filter_binning,
                var_key_2_used_for_filter_binning,
                bool_use_std_of_mean
            )

    interval_dict = {
        'on': {},
        'off': {},
    }

    for ctrl_key in ctrl_keys:

        for pair_n in range(n_turb_pairs):

            turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
            turb_key_up = turb_keys_up[pair_n]
            turb_key_down = turb_keys_down[pair_n]

            group_keys = list(df_binned_grouper_dict[ctrl_key][turb_key_up].groups.keys())

            if n_bin_dims == 1:

                n_bins_1 = bins_per_pair[pair_n].shape[0]
                group_keys = [gk for gk in group_keys if not pd.isna(gk)]

            elif n_bin_dims == 2:

                group_keys = [
                    gk for gk in group_keys if not pd.isna(gk[0]) and not pd.isna(gk[1])]

                n_bins_1 = bins_1_per_pair[pair_n].shape[0]
                # n_bins_2 = bins_2_per_pair[pair_n].shape[0]

            # remove empty keys:
            group_counter = df_binned_grouper_dict[ctrl_key][turb_key_up].count()

            group_keys = [
                gk for gk in group_keys if group_counter.loc[gk].iloc[0] > 0
            ]

            interval_dict[ctrl_key][turb_key_up] = []
            interval_dict[ctrl_key][turb_key_down] = []

            # bin_n_1 = 0
            # bin_n_2 = 0

            for group_n, group_key in enumerate(group_keys):

                if n_bin_dims == 1:

                    bin_n_1 = group_n

                elif n_bin_dims == 2:

                    bin_n_1 = group_n % n_bins_1
                    bin_n_2 = group_n // n_bins_1

                df_up = df_binned_grouper_dict[ctrl_key][turb_key_up].get_group(group_key)
                df_down = df_binned_grouper_dict[ctrl_key][turb_key_down].get_group(group_key)

                index_for_filt = df_up.index.to_series()
                grouping_mask_time_intervals_pair = (
                        index_for_filt.diff().dt.total_seconds().bfill() > max_gap_duration_s
                ).cumsum()

                for k, v in df_up.groupby(grouping_mask_time_intervals_pair):
                    interval_dict[ctrl_key][turb_key_up].append(v)
                    if n_bin_dims == 1:
                        int_cnt_dict[ctrl_key][turb_key_up][bin_n_1] += 1
                    elif n_bin_dims == 2:
                        int_cnt_dict[ctrl_key][turb_key_up][bin_n_1][bin_n_2] += 1

                # for bin_n, (k, v) in enumerate(df_up.groupby(grouping_mask_time_intervals_pair)):
                for k, v in df_down.groupby(grouping_mask_time_intervals_pair):
                    interval_dict[ctrl_key][turb_key_down].append(v)
                    if n_bin_dims == 1:
                        int_cnt_dict[ctrl_key][turb_key_down][bin_n_1] += 1
                    elif n_bin_dims == 2:
                        int_cnt_dict[ctrl_key][turb_key_down][bin_n_1][bin_n_2] += 1

                group_n += 1

    return interval_dict


def count_time_intervals_in_df_filt_dict(
        df_dict,
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
):

    n_turb_pairs = len(turb_keys_split_by_pair)

    int_cnt_dict = {
        'on': {},
        'off': {},
    }

    if n_bin_dims == 1:

        bins_per_pair = bins_per_pair_list[0]
        var_key_used_for_filter_binning = var_key_used_for_filter_binning_list[0]

        df_binned_grouper_dict, df_binned_1D_mean_dict, df_binned_1D_std_dict = \
            bin_filt_ctrl_turb_dict_1D(
                df_dict,
                var_keys_to_bin,
                turb_keys_split_by_pair,
                idx_upstream,
                bins_per_pair,
                var_key_used_for_filter_binning,
                bool_use_std_of_mean
            )

        for ctrl_key in ctrl_keys:

            for pair_n in range(n_turb_pairs):

                turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
                turb_key_up = turb_keys_up[pair_n]
                turb_key_down = turb_keys_down[pair_n]

                bins_1 = bins_per_pair[pair_n]

                n_bins_1 = bins_1.shape[0] - 1

                int_cnt_arr = np.zeros(n_bins_1)

                for turb_key in turb_keys_split_by_pair[pair_n]:
                    int_cnt_dict[ctrl_key][turb_key] = int_cnt_arr.copy()

                group_keys = list(df_binned_grouper_dict[ctrl_key][turb_key_up].groups.keys())
                group_keys = [gk for gk in group_keys if not pd.isna(gk)]

                # remove empty keys:
                group_counter = df_binned_grouper_dict[ctrl_key][turb_key_up].count()

                group_keys = [
                    gk for gk in group_keys if group_counter.loc[gk].iloc[0] > 0
                ]

                group_keys_total = gen_pandas_bin_keys_from_bins_1D(bins_1)

                for bin_n_1 in range(n_bins_1):

                    group_key_x = group_keys_total[bin_n_1]

                    if group_key_x in group_keys:

                        df_up = df_binned_grouper_dict[ctrl_key][turb_key_up].get_group(
                            group_key_x)
                        index_for_filt = df_up.index.to_series()
                        grouping_mask_time_intervals_pair = (
                                index_for_filt.diff().dt.total_seconds().bfill() > max_gap_duration_s
                                ).cumsum()
                        cnt_ = grouping_mask_time_intervals_pair.unique().shape[0]

                        int_cnt_dict[ctrl_key][turb_key_up][bin_n_1] = cnt_
                        int_cnt_dict[ctrl_key][turb_key_down][bin_n_1] = cnt_

    elif n_bin_dims == 2:

        bins_1_per_pair = bins_per_pair_list[0]
        bins_2_per_pair = bins_per_pair_list[1]

        var_key_1_used_for_filter_binning = var_key_used_for_filter_binning_list[0]
        var_key_2_used_for_filter_binning = var_key_used_for_filter_binning_list[1]

        df_binned_grouper_dict, df_binned_2D_mean_dict, df_binned_2D_std_dict = \
            bin_filt_ctrl_turb_dict_2D(
                df_dict,
                var_keys_to_bin,
                turb_keys_split_by_pair,
                idx_upstream,
                bins_1_per_pair,
                bins_2_per_pair,
                var_key_1_used_for_filter_binning,
                var_key_2_used_for_filter_binning,
                bool_use_std_of_mean
            )

        for ctrl_key in ctrl_keys:

            for pair_n in range(n_turb_pairs):

                turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
                turb_key_up = turb_keys_up[pair_n]
                turb_key_down = turb_keys_down[pair_n]

                bins_1 = bins_1_per_pair[pair_n]
                bins_2 = bins_2_per_pair[pair_n]

                n_bins_1 = bins_1.shape[0] - 1
                n_bins_2 = bins_2.shape[0] - 1

                int_cnt_arr = np.zeros((
                    n_bins_1,
                    n_bins_2
                ))

                for turb_key in turb_keys_split_by_pair[pair_n]:
                    int_cnt_dict[ctrl_key][turb_key] = int_cnt_arr.copy()

                group_keys = list(df_binned_grouper_dict[ctrl_key][turb_key_up].groups.keys())
                group_keys = [
                    gk for gk in group_keys if not pd.isna(gk[0]) and not pd.isna(gk[1])]

                # remove empty keys:
                group_counter = df_binned_grouper_dict[ctrl_key][turb_key_up].count()

                group_keys = [
                    gk for gk in group_keys if group_counter.loc[gk].iloc[0] > 0
                ]

                group_keys_total = gen_pandas_bin_keys_from_bins_2D(bins_1, bins_2)

                group_n = 0

                for bin_n_2 in range(n_bins_2):
                    for bin_n_1 in range(n_bins_1):

                        group_key_x = group_keys_total[group_n]

                        if group_key_x in group_keys:

                            df_up = df_binned_grouper_dict[ctrl_key][turb_key_up].get_group(
                                group_key_x)
                            index_for_filt = df_up.index.to_series()
                            grouping_mask_time_intervals_pair = (
                                    index_for_filt.diff().dt.total_seconds().bfill() > max_gap_duration_s
                            ).cumsum()

                            cnt_ = grouping_mask_time_intervals_pair.unique().shape[0]

                            int_cnt_dict[ctrl_key][turb_key_up][bin_n_1, bin_n_2] = cnt_

                        group_n += 1

    return int_cnt_dict


## SCRATCH


def filter_data_BAK(
        df_dict,
        wdir_min, wdir_max,
        power_min, power_max,
        wspd_min, wspd_max,
        errorcode_min, errorcode_max,
        pitch_min, pitch_max,
        time_index_min_dict,
        discard_time_at_beginning_s,
        max_gap_duration_s,
        min_interval_duration_s,
        turb_keys_split,
        idx_upstream, idx_downstream,
        bool_save_filtered_data,
        resample_interval_s=0,
        path2dir_filtered=''
):
    """

    :param df_dict: df_dict['T3'] = pandas dataframe with data for T3 etc.
    time_index_min_dict: index_min_dict['T4'] = '2023-06-29 08:00:00' =
    data will only be considered after this time
    turb_keys_split: [['T3', 'T4'], ['T5', 'T6']]
    idx_upstream: index of the upstream turbine in each sublist of turb_keys_split
    idx_downstream: index of the downstream turbine in each sublist of turb_keys_split
    :return:
    """

    print('start filtering data')

    if resample_interval_s > 0:
        n_first_rows_to_mask = int(discard_time_at_beginning_s / resample_interval_s)

    filter_mask_ctrl_on_list = []
    filter_mask_ctrl_off_list = []
    filter_mask_ctrl_combo_list = []

    n_turb_pairs = len(turb_keys_split)

    for pair_n in range(n_turb_pairs):

        turb_key_up = turb_keys_split[pair_n][idx_upstream]
        turb_key_down = turb_keys_split[pair_n][idx_downstream]

        # --- filter turbine pair

        # --- start from minimum time index
        time_index_min_up = time_index_min_dict[turb_key_up]
        time_index_min_down = time_index_min_dict[turb_key_down]

        if time_index_min_up is None:
            time_index_min_up = '1000-01-01 00:00:00'
        if time_index_min_down is None:
            time_index_min_down = '1000-01-01 00:00:00'

        filter_mask_date = df_dict[turb_key_up].index >= time_index_min_up
        filter_mask_date = np.logical_and(
            filter_mask_date, df_dict[turb_key_down].index >= time_index_min_down
        )

        # --- filter upstream turbine for wind direction and wind speed

        # wdir of proc. data is in [0, 360).
        # if wdir_max > 360, need to split into two intervals
        if wdir_max >= 360.0:
            wdir_max1 = 360.0
            wdir_max2 = wdir_max - 360.0
            wdir_min1 = wdir_min
            wdir_min2 = 0.0

            filter_mask_pair_a = (
                df_dict[turb_key_up]['WDir'].between(wdir_min1, wdir_max1, inclusive='both')
            )

            filter_mask_pair_b = (
                df_dict[turb_key_up]['WDir'].between(wdir_min2, wdir_max2, inclusive='both')
            )

            filter_mask_pair = (filter_mask_pair_a | filter_mask_pair_b)

        else:
            filter_mask_pair = (
                df_dict[turb_key_up]['WDir'].between(wdir_min, wdir_max, inclusive='both')
            )

        filter_mask_pair = (
                filter_mask_pair
                & df_dict[turb_key_up]['WSpeed'].between(wspd_min, wspd_max, inclusive='both')
        )

        filter_mask_pair = (filter_mask_pair & filter_mask_date)

        # --- filter upstream and downstream for power, errorcode and pitch

        for turb_key in [turb_key_up, turb_key_down]:

            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Power'].between(power_min, power_max, inclusive='both')
            )
            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Errorcode'].between(errorcode_min, errorcode_max,
                                                             inclusive='both')
            )
            filter_mask_pair = (
                    filter_mask_pair
                    & df_dict[turb_key]['Pitch'].between(pitch_min, pitch_max, inclusive='both')
            )

        # --- split into yaw control on and off
        df_dict[turb_key_up]['ControlSwitch'][df_dict[turb_key_up]['ControlSwitch'] < -0.1] \
            = np.nan

        control_switch_mask_pair = (
                ~df_dict[turb_key_up]['ControlSwitch'].ffill().isna()
                & df_dict[turb_key_up]['ControlSwitch'].ffill().astype(bool)
        )

        # --- control on only
        filter_mask_ctrl_on_pair = (filter_mask_pair & control_switch_mask_pair)

        # --- control off only
        filter_mask_ctrl_off_pair = (filter_mask_pair & ~control_switch_mask_pair)

        # --- restrict valid time intervals ctrl on
        index_filt = df_dict[turb_key_up].index.to_series()[filter_mask_ctrl_on_pair]
        grouping_mask_time_intervals_pair = (
                index_filt.diff().dt.total_seconds().bfill() > max_gap_duration_s
        )
        grouping_mask_time_intervals_pair = grouping_mask_time_intervals_pair.cumsum()

        mask_time_intervals_pair = (
                index_filt.groupby(grouping_mask_time_intervals_pair).transform('count')
                >= min_interval_duration_s
        )

        grouping_mask_time_intervals_pair = \
            grouping_mask_time_intervals_pair[mask_time_intervals_pair]
        grouped_mask = grouping_mask_time_intervals_pair.groupby(grouping_mask_time_intervals_pair)
        mask_int_begin = grouped_mask.transform(mask_first_n_rows, n_first_rows_to_mask)

        mask_time_intervals_pair = mask_time_intervals_pair & mask_int_begin

        # --- complete filter mask control on
        filter_mask_ctrl_on_pair = filter_mask_ctrl_on_pair & mask_time_intervals_pair

        # --- restrict valid time intervals ctrl off
        index_filt = df_dict[turb_key_up].index.to_series()[filter_mask_ctrl_off_pair]
        grouping_mask_time_intervals_pair = (
                index_filt.diff().dt.total_seconds().bfill() > max_gap_duration_s
        )
        grouping_mask_time_intervals_pair = grouping_mask_time_intervals_pair.cumsum()

        mask_time_intervals_pair = (
                index_filt.groupby(grouping_mask_time_intervals_pair).transform('count')
                >= min_interval_duration_s
        )

        grouping_mask_time_intervals_pair = \
            grouping_mask_time_intervals_pair[mask_time_intervals_pair]
        grouped_mask = grouping_mask_time_intervals_pair.groupby(grouping_mask_time_intervals_pair)
        mask_int_begin = grouped_mask.transform(mask_first_n_rows, n_first_rows_to_mask)

        mask_time_intervals_pair = mask_time_intervals_pair & mask_int_begin

        # --- complete filter mask control off
        filter_mask_ctrl_off_pair = filter_mask_ctrl_off_pair & mask_time_intervals_pair

        # --- append
        filter_mask_ctrl_on_list.append(filter_mask_ctrl_on_pair)
        filter_mask_ctrl_off_list.append(filter_mask_ctrl_off_pair)

        filter_mask_ctrl_combo_list.append(filter_mask_ctrl_on_pair | filter_mask_ctrl_off_pair)

    df_filt_ctrl_on_dict = {}
    df_filt_ctrl_off_dict = {}
    df_filt_ctrl_combo_dict = {}

    for pair_n in range(n_turb_pairs):
        for turb_key in turb_keys_split[pair_n]:
            df_filt_ctrl_on_dict[turb_key] = \
                df_dict[turb_key][filter_mask_ctrl_on_list[pair_n]]
            df_filt_ctrl_off_dict[turb_key] = \
                df_dict[turb_key][filter_mask_ctrl_off_list[pair_n]]
            df_filt_ctrl_combo_dict[turb_key] = \
                df_dict[turb_key][filter_mask_ctrl_combo_list[pair_n]]

            if bool_save_filtered_data:

                resample_str = f'{resample_interval_s:.0f}s'

                fname = 'filtered_ctrl_on_' + resample_str + '_' + turb_key
                df_filt_ctrl_on_dict[turb_key].to_hdf(
                    path2dir_filtered + os.sep + fname + '.h5',
                    'scada', mode='w'
                )

                fname = 'filtered_ctrl_off_' + resample_str + '_' + turb_key
                df_filt_ctrl_off_dict[turb_key].to_hdf(
                    path2dir_filtered + os.sep + fname + '.h5',
                    'scada', mode='w'
                )

                fname = 'filtered_ctrl_combo_' + resample_str + '_' + turb_key
                df_filt_ctrl_combo_dict[turb_key].to_hdf(
                    path2dir_filtered + os.sep + fname + '.h5',
                    'scada', mode='w'
                )

            print('saved filtered data to hard drive for', turb_key)

    print('finish filtering data')

    return df_filt_ctrl_on_dict, df_filt_ctrl_off_dict, df_filt_ctrl_combo_dict


# BACKUP
# def load_limit_filtered_data(
#         turb_keys_to_process,
#         path2dir_filtered,
#         resample_interval_s,
# ):
#     print('-- start loading filtered data from disk')
#
#     resample_str_to_load = f'{resample_interval_s:.0f}s'
#
#     df_filt_ctrl_on_dict = {}
#     df_filt_ctrl_off_dict = {}
#     df_filt_ctrl_combo_dict = {}
#
#     for turb_key in turb_keys_to_process:
#
#         fname = 'filtered_ctrl_on_' + resample_str_to_load + '_' + turb_key
#         df_filt_ctrl_on_dict[turb_key] = pd.read_hdf(path2dir_filtered + os.sep + fname + '.h5')
#
#         fname = 'filtered_ctrl_off_' + resample_str_to_load + '_' + turb_key
#         df_filt_ctrl_off_dict[turb_key] = pd.read_hdf(path2dir_filtered + os.sep + fname + '.h5')
#
#         fname = 'filtered_ctrl_combo_' + resample_str_to_load + '_' + turb_key
#         df_filt_ctrl_combo_dict[turb_key] = pd.read_hdf(
#         path2dir_filtered + os.sep + fname + '.h5')
#
#     print('-- finish loading filtered data from disk')
#
#     # filter_mask_pair_1 = pd.read_hdf(path2dir_filtered + os.sep + 'filter_mask_pair_1.h5')
#     # filter_mask_pair_2 = pd.read_hdf(path2dir_filtered + os.sep + 'filter_mask_pair_2.h5')
#
#     return df_filt_ctrl_on_dict, df_filt_ctrl_off_dict, df_filt_ctrl_combo_dict

