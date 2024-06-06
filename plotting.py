## IMPORTS

import os
import matplotlib as mpl
# INFO: CHOOSE BACKEND (TkAgg is for Windows)
MPL_BACKEND = 'TkAgg'
# MPL_BACKEND = 'QtAgg'
mpl.use(MPL_BACKEND)
import ipython_tools
ipython_tools.set_mpl_magic(MPL_BACKEND)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# careful with removing warnings
pd.options.mode.chained_assignment = None  # default='warn'
# import scipy as sp

import config
import analysis as ana
import processing as proc
from plot_tools import cm2inch, move_figure_window, make_dir_if_not_exists, marker_list, \
    delete_all_files_from_dir
from my_rc_params import rc_params_dict
# rc_params_dict['savefig.dpi'] = 300
mpl.rcParams.update(mpl.rcParamsDefault) # reset to default settings
mpl.rcParams.update(rc_params_dict) # set to custom settings

from scipy.stats import weibull_min # For fitting weibull distribution
import scipy.interpolate as interp
from scipy.optimize import curve_fit


## FUNCTIONS
#########################################################
# INFO:
# TASK 1: Find positioning of wind turines vs wind vane measurements

def plot_norm_power_ratio_vs_wdir(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        wdir_bins_per_pair,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize,
        path2dir_fig,
):
    n_turb_pairs = len(turb_keys_split_by_pair)

    fig, axes = plt.subplots(nrows=2, ncols=n_turb_pairs, figsize=figsize, sharex=True)

    for pair_n, (ax_ratio, ax_yaw) in enumerate(zip(axes[0], axes[1])):
        turb_up = turb_keys_split_by_pair[pair_n][idx_upstream]
        turb_down = turb_keys_split_by_pair[pair_n][idx_downstream]

        wdir_bins = wdir_bins_per_pair[pair_n]
        wdir_bin_centers = 0.5 * (wdir_bins[:-1] + wdir_bins[1:])

        norm_power_ratio_mean = df_binned_1D_mean_dict['off'][turb_down]['NormPower'] / df_binned_1D_mean_dict['off'][turb_up]['NormPower']
        norm_power_ratio_std = np.sqrt(
            (df_binned_1D_std_dict['off'][turb_down]['NormPower'] / df_binned_1D_mean_dict['off'][turb_up]['NormPower'])**2
            + (df_binned_1D_std_dict['off'][turb_up]['NormPower'] * df_binned_1D_mean_dict['off'][turb_down]['NormPower'] / df_binned_1D_mean_dict['off'][turb_up]['NormPower']**2)**2
        )

        # Plot normalized power ratio
        ax_ratio.plot(wdir_bin_centers, norm_power_ratio_mean, marker='o', label=f'Pair {pair_n + 1}', markersize=1.5)
        ax_ratio.fill_between(wdir_bin_centers, norm_power_ratio_mean - norm_power_ratio_std, norm_power_ratio_mean + norm_power_ratio_std, alpha=0.5)

        # Fit a polynomial function to the data
        poly_degree = 8  # Adjust the degree as needed
        poly_coeffs = np.polyfit(wdir_bin_centers, norm_power_ratio_mean, poly_degree)
        poly_func = np.poly1d(poly_coeffs)

        # Generate points for the fitted polynomial curve
        wdir_fit = np.linspace(wdir_bin_centers.min(), wdir_bin_centers.max(), 100)
        norm_power_ratio_fit = poly_func(wdir_fit)

        # Find the minimum point of the fitted polynomial curve
        min_idx = np.argmin(norm_power_ratio_fit)
        min_wdir = wdir_fit[min_idx]
        min_norm_power_ratio = norm_power_ratio_fit[min_idx]
        
        # Extract the yaw angles at the minimum power point wind direction
        closest_bin_idx = np.argmin(np.abs(wdir_bin_centers - min_wdir))
        closest_wdir = wdir_bin_centers[closest_bin_idx]
        print("Debug: Closest WDir:", closest_wdir)  # Debug output
         
        yaw_up_at_min_series = df_binned_1D_mean_dict['off'][turb_up]['Yaw'].loc[df_binned_1D_mean_dict['off'][turb_up]['Yaw'].index == closest_wdir]
        yaw_down_at_min_series = df_binned_1D_mean_dict['off'][turb_down]['Yaw'].loc[df_binned_1D_mean_dict['off'][turb_down]['Yaw'].index == closest_wdir]
         
        print("Debug: Yaw Up Series:", yaw_up_at_min_series)  # Debug output
        print("Debug: Yaw Down Series:", yaw_down_at_min_series)  # Debug output
         
        yaw_up_at_min = yaw_up_at_min_series.iloc[0] if not yaw_up_at_min_series.empty else np.nan
        yaw_down_at_min = yaw_down_at_min_series.iloc[0] if not yaw_down_at_min_series.empty else np.nan
        
        # Compare the yaw angles with the minimum power point wind direction
        print(f"Pair {pair_n + 1}:")
        print(f"Minimum power point wind direction: {min_wdir:.1f} degrees")
        print(f"{turb_up} yaw angle: {yaw_up_at_min:.1f} degrees")
        print(f"{turb_down} yaw angle: {yaw_down_at_min:.1f} degrees")
        print()

        # Plot the fitted polynomial curve and minimum point
        ax_ratio.plot(wdir_fit, norm_power_ratio_fit, 'r--', linewidth=0.8, label='Fitted Polynomial')
        ax_ratio.plot(min_wdir, min_norm_power_ratio, 'ro', markersize=2, label=f'Minimum at {min_wdir:.1f}°')

        ax_ratio.set_xlabel('Wind Direction [deg]', fontsize=5)
        if pair_n == 0:
            ax_ratio.set_ylabel('Norm. Power Ratio (Down/Up)', fontsize=5)
        ax_ratio.legend(fontsize=5)
        ax_ratio.grid()

        # Increase the number of ticks on the x and y axes
        ax_ratio.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax_ratio.yaxis.set_major_locator(plt.MaxNLocator(10))
        ax_ratio.tick_params(axis='x', labelsize=5)  # Adjust fontsize for x-axis labels
        ax_ratio.tick_params(axis='y', labelsize=5)  # Adjust fontsize for y-axis labels

        # Plot yaw angles
        ax_yaw.plot(wdir_bin_centers, df_binned_1D_mean_dict['off'][turb_up]['Yaw'], marker='o', label=f'{turb_up} Yaw')
        ax_yaw.plot(wdir_bin_centers, df_binned_1D_mean_dict['off'][turb_down]['Yaw'], marker='o', label=f'{turb_down} Yaw')
        ax_yaw.plot(wdir_bin_centers, wdir_bin_centers, '--', color='red', label='Ideal Yaw')

        ax_yaw.set_xlabel('Wind Direction [deg]', fontsize=5)
        if pair_n == 0:
            ax_yaw.set_ylabel('Yaw Angle [deg]', fontsize=5)
        ax_yaw.legend(fontsize=5)
        ax_yaw.grid()

        # Increase the number of ticks on the x and y axes
        ax_yaw.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax_yaw.yaxis.set_major_locator(plt.MaxNLocator(10))
        ax_yaw.tick_params(axis='x', labelsize=5)  # Adjust fontsize for x-axis labels
        ax_yaw.tick_params(axis='y', labelsize=5)  # Adjust fontsize for y-axis labels

    fig.suptitle('Normalized Power Ratio (Down/Up) vs Wind Direction (Ctrl Off)', fontsize=6)
    fig.tight_layout()

    if bool_save_fig:
        figname = 'norm_power_ratio_vs_wdir_ctrl_off.png'
        fig.savefig(os.path.join(path2dir_fig, figname), bbox_inches='tight')
        plt.close(fig)
        
# INFO: TASK 2
def plot_wspd_dist_upstream_turbs(
        df_filt_ctrl_turb_dict,
        turb_keys_up,
        ctrl_keys,
        path2dir_fig,
        figsize=(8, 6)
):
    """
    Plots the Weibull-like distribution curves of wind speed for the two upstream turbines (T4 and T6)
    for each control mode.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for ctrl_key in ctrl_keys:
        for turb_key in turb_keys_up:
            df = df_filt_ctrl_turb_dict[ctrl_key][turb_key]
            wspd = df['WSpeed'].values
            wspd = wspd[~np.isnan(wspd)]  # Remove NaN values
            wspd_mean = wspd.mean()

            # Fit Weibull distribution
            shape, loc, scale = weibull_min.fit(wspd, floc=0)

            # Calculate Weibull PDF
            x = np.linspace(wspd.min(), wspd.max(), 1000)
            pdf = weibull_min.pdf(x, shape, loc, scale)

            label = f'{turb_key}, Ctrl {ctrl_key}'
            line, = ax.plot(x, pdf, label=label)

            # Add vertical bar for mean wind speed
            ax.axvline(wspd_mean, color=line.get_color(), linestyle='--', label=f'Mean WSpeed ({turb_key})')

            # Shade area under curve for "off" control mode
            if ctrl_key == 'off':
                ax.fill_between(x, pdf, alpha=0.2, color=line.get_color())

    ax.set_xlabel('Wind Speed [m/s]')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid()
    
    title = 'Weibull distribution curves of wind speed for the two upstream turbines for each control mode'
    fig.suptitle(title)
    fig.tight_layout()

    figname = 'wspd_dist_upstream_turbs.png'
    fig.savefig(os.path.join(path2dir_fig, figname), bbox_inches='tight')
    plt.close(fig)

#######################################################
def plot_power_diff_vs_abs_yaw_mis(
        df_dict, 
        turb_keys_split_by_pair,
        idx_upstream, 
        idx_downstream,
        path2dir_yaw_table,
        fname_yaw_table,
        bin_width=1.0,
        figsize=(8, 6),
        bool_save_fig=False,
        path2dir_fig=None
):
    calc_yaw_setpoint_per_pair = proc.gen_yaw_table_interp(
            path2dir_yaw_table,
            fname_yaw_table,
    )
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    for pair_n, (ax, turb_pair) in enumerate(zip(axes, turb_keys_split_by_pair)):
        turb_up = turb_pair[idx_upstream]
        turb_down = turb_pair[idx_downstream]
        
        df_up = df_dict[turb_up]
        df_down = df_dict[turb_down]
        
        # Calculate yaw misalignment using the yaw table
        calc_yaw_setpoint_x = calc_yaw_setpoint_per_pair[pair_n]
        yaw_mis_up = df_up['Yaw'] - calc_yaw_setpoint_x(df_up['WDir'])
        
        bins = np.arange(-10, 10+bin_width, bin_width)  # Adjust the bin range as needed
        
        if pair_n == 0:
            power_diff_pct = (df_dict['T3']['Power']/df_dict['T3']['Power'].mean() - 
                              df_dict['T5']['Power']/df_dict['T5']['Power'].mean()) * 100
        else:
            power_diff_pct = (df_dict['T5']['Power']/df_dict['T5']['Power'].mean() - 
                              df_dict['T3']['Power']/df_dict['T3']['Power'].mean()) * 100
        
        df_binned = pd.DataFrame({
            'yaw_mis_up': pd.cut(yaw_mis_up.abs(), bins),
            'power_diff_pct': power_diff_pct
        })
        
        power_diff_binned = df_binned.groupby('yaw_mis_up').mean()
        
        ax.bar(power_diff_binned.index.astype(str), power_diff_binned['power_diff_pct'], 
               width=0.8, align='center', alpha=0.7)
        
        if pair_n == 0:
            ax.set_ylabel(f'Power Diff T3 vs T5 [%]')
        else:
            ax.set_ylabel(f'Power Diff T5 vs T3 [%]')
        ax.grid()
        ax.tick_params(axis='x', labelsize=5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(15))
        
    axes[1].set_xlabel('Abs Yaw Misalignment Upstream [deg]')
    
    fig.tight_layout()
    
    if bool_save_fig:
        fig.savefig(path2dir_fig + 'power_diff_vs_abs_yaw_mis.png')
##########################################################
def plot_wspd_vs_yawmis(
        df_dict,
        turb_keys_to_process,
        turb_keys_split_by_pair,
        idx_upstream,
        idx_downstream,
        errorcode_val,
        bool_save_fig,
        path2dir_fig,
        figsize=(8, 6),
        marker_size=2,
        alpha=0.5,
):
    """
    Plots wind speed vs yaw misalignment for a given error code value.

    Args:
        df_dict (dict): Dictionary containing pandas DataFrames for each turbine.
        turb_keys_to_process (list): List of turbine keys to process.
        turb_keys_split_by_pair (list): List of turbine key pairs.
        idx_upstream (int): Index of the upstream turbine in each pair.
        idx_downstream (int): Index of the downstream turbine in each pair.
        errorcode_val (float): Error code value to filter the data.
        bool_save_fig (bool): Whether to save the figure or not.
        path2dir_fig (str): Path to the directory where the figure will be saved.
        figsize (tuple, optional): Figure size (width, height) in inches. Default is (8, 6).
        marker_size (int, optional): Marker size for the scatter plot. Default is 4.
        alpha (float, optional): Alpha value for the scatter plot. Default is 0.5.
    """
    n_turb_pairs = len(turb_keys_split_by_pair)

    fig, axes = plt.subplots(nrows=1, ncols=n_turb_pairs, figsize=figsize, sharey=True)

    for pair_n in range(n_turb_pairs):
        ax = axes[pair_n]

        turb_key_up = turb_keys_split_by_pair[pair_n][idx_upstream]
        turb_key_down = turb_keys_split_by_pair[pair_n][idx_downstream]

        df_up = df_dict[turb_key_up]
        df_down = df_dict[turb_key_down]

        mask_errorcode = (df_up['Errorcode'] == errorcode_val) & (df_down['Errorcode'] == errorcode_val)

        ax.scatter(
            df_up['WSpeed'][mask_errorcode],
            df_up['Yaw'][mask_errorcode] - df_up['WDir'][mask_errorcode],
            s=marker_size,
            alpha=alpha,
            label=turb_key_up
        )

        ax.scatter(
            df_down['WSpeed'][mask_errorcode],
            df_down['Yaw'][mask_errorcode] - df_down['WDir'][mask_errorcode],
            s=marker_size,
            alpha=alpha,
            label=turb_key_down
        )

        ax.set_xlabel('Wind Speed [m/s]')
        ax.set_ylabel('Yaw Misalignment [deg]')
        ax.legend()
        ax.grid()
        ax.set_title(f'Pair {pair_n + 1}')

    fig.suptitle(f'Error Code: {errorcode_val}')
    fig.tight_layout()

    if bool_save_fig:
        figname = f'wspd_vs_yawmis_errorcode_{errorcode_val:.0f}.png'
        fig.savefig(os.path.join(path2dir_fig, figname), bbox_inches='tight')
        plt.close(fig)
#################################################


def plot_df_turb_dict_overview(
        df_dict,
        var_keys_to_process,
        turb_keys_to_process,
        turb_keys_split_by_pair,
        resample_str,
        path2dir_fig,
        marker_dict_by_turb,
        color_dict_by_turb,
        wdir_min_plot,
        wdir_max_plot,
        dit_plot_unfilt,
        bool_plot_overview,
        bool_plot_overview_split,
        bool_plot_wdir_selected_range,
        bool_plot_wdir_vs_yaw,
        bool_plot_wdir_vs_yaw_selected_range
):

    print('-- start plotting unfiltered data')

    n_selected_vars = len(var_keys_to_process)
    n_turb_pairs = len(turb_keys_split_by_pair)

    if 'Pitch' in var_keys_to_process:
        for turb_key in turb_keys_to_process:
            df_dict[turb_key]['Pitch'] = (df_dict[turb_key]['Pitch'] + 180.) % 360. - 180.

    date_format_str = "%m-%d %H:%M"

    title_add_str = 'unfilt'
    path2dir_fig_unfilt = (path2dir_fig + os.sep + 'no_filter')
    make_dir_if_not_exists(path2dir_fig_unfilt)

    # PLOT ONE BIG FIGURE WITH ALL TURBINES IN ONE SUBPLOT PER QUANTITY
    if bool_plot_overview:
        n_cols = 1
        n_rows = n_selected_vars

        figsize_cm = [20, 4 * n_rows]
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True,
                                 figsize=cm2inch(figsize_cm))
        if n_selected_vars == 1:
            axes = [axes]

        for var_n, var_key in enumerate(var_keys_to_process):

            ax = axes[var_n]

            for turb_n, turb_key in enumerate(turb_keys_to_process):
                if var_key == 'ControlSwitch' and turb_key not in ['T4', 'T6']:
                    pass
                else:
                    ax.plot(df_dict[turb_key].index[::dit_plot_unfilt],
                            df_dict[turb_key][var_key][::dit_plot_unfilt],
                            marker_dict_by_turb[turb_key], color=color_dict_by_turb[turb_key],
                            fillstyle='none',
                            mew=0.3, ms=2,
                            # label=var_key + '_' + turb_key
                            label=turb_key
                            )

                if var_key in ['Power']:
                    ax.set_ylim([0, 5500])

            if var_n == n_selected_vars - 1:
                myFmt = mpl.dates.DateFormatter(date_format_str)
                ax.xaxis.set_major_formatter(myFmt)
                ax.tick_params(axis='x', labelrotation=90)

            ax.set_ylabel(var_key)

            ax.grid()
            ax.legend()

        fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
        fig.tight_layout()

        figname = 'Overview_plot_' + resample_str
        fig.savefig(path2dir_fig_unfilt + os.sep + figname + '.png')
        plt.close('all')

    # PLOT TURBINES PAIRS SEPARATELY
    # plot one big figure with all turbines in one subplot per quantity
    if bool_plot_overview_split:
        n_cols = 2
        n_rows = n_selected_vars

        figsize_cm = [20, 4 * n_rows]
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True,
                                 figsize=cm2inch(figsize_cm))
        if n_selected_vars == 1:
            axes = [axes]

        for var_n, var_key in enumerate(var_keys_to_process):
            for pair_n in range(n_turb_pairs):
                ax = axes[var_n][pair_n]

                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                    if var_key == 'ControlSwitch' and turb_key not in ['T4', 'T6']:
                        pass
                    else:
                        ax.plot(df_dict[turb_key].index[::dit_plot_unfilt],
                                df_dict[turb_key][var_key][::dit_plot_unfilt],
                                marker_dict_by_turb[turb_key], color=color_dict_by_turb[turb_key],
                                fillstyle='none',
                                mew=0.3, ms=2,
                                # label=var_key + '_' + turb_key
                                label=turb_key
                                )

                if var_key in ['Power']:
                    ax.set_ylim([0, 5500])
                if var_n == n_selected_vars - 1:
                    myFmt = mpl.dates.DateFormatter(date_format_str)
                    ax.xaxis.set_major_formatter(myFmt)
                    ax.tick_params(axis='x', labelrotation=90)

                ax.set_ylabel(var_key)

                ax.grid()
                ax.legend()

        fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
        fig.tight_layout()

        figname = 'Overview_plot_split_pairs_' + resample_str
        fig.savefig(path2dir_fig_unfilt + os.sep + figname + '.png')
        plt.close('all')

    # PLOT WDIR AND YAW ONLY FOR TURBINES PAIRS SEPARATELY
    if bool_plot_wdir_selected_range:
        var_keys_to_process2 = [
            'WDir',
            'Yaw',
            'Power'
        ]

        n_cols = 2
        n_rows = len(var_keys_to_process2)

        figsize_cm = [16, 8 * n_rows]
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True,
                                 figsize=cm2inch(figsize_cm))

        for var_n, var_key in enumerate(var_keys_to_process2):
            for pair_n in range(n_turb_pairs):
                ax = axes[var_n][pair_n]

                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                    # label = label_dict[turb_key][var_key]
                    if var_key == 'ControlSwitch' and turb_key not in ['T4', 'T6']:
                        pass
                    else:
                        ax.plot(df_dict[turb_key].index,
                                df_dict[turb_key][var_key],
                                marker_dict_by_turb[turb_key], color=color_dict_by_turb[turb_key],
                                fillstyle='none',
                                mew=0.3, ms=2,
                                # label=var_key + '_' + turb_key
                                label=turb_key
                                )

                if var_key in ['Power']:
                    ax.set_ylim([0, 5500])
                else:
                    ax.set_ylim([295, 325])

                if var_n == n_rows - 1:
                    myFmt = mpl.dates.DateFormatter(date_format_str)
                    ax.xaxis.set_major_formatter(myFmt)
                    ax.tick_params(axis='x', labelrotation=90)

                ax.set_ylabel(var_key)

                ax.grid()
                ax.legend()

        fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
        fig.tight_layout()

        figname = 'WDir_Yaw_selected_range_' + resample_str
        fig.savefig(path2dir_fig_unfilt + os.sep + figname + '.png')
        plt.close('all')

    # PLOT WDIR VS. YAW FOR TURB PAIRS SEPARATELY
    if bool_plot_wdir_vs_yaw:

        n_cols = 1
        n_rows = 2

        figsize_cm = [12, 24]
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=cm2inch(figsize_cm))

        for pair_n in range(n_turb_pairs):
            ax = axes[pair_n]

            for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                ax.plot(df_dict[turb_key]['WDir'],
                        df_dict[turb_key]['Yaw'],
                        marker_dict_by_turb[turb_key], color=color_dict_by_turb[turb_key],
                        fillstyle='none',
                        mew=0.3, ms=2,
                        # label=var_key + '_' + turb_key
                        label=turb_key
                        )

            ax.plot(np.linspace(0, 360, 100),
                    np.linspace(0, 360, 100),
                    '--', color='grey')

            ax.set_xlabel('WDir')
            ax.set_ylabel('Yaw')

            ax.grid()
            ax.legend()

        fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
        fig.tight_layout()

        figname = 'WDir_vs_Yaw_' + resample_str
        fig.savefig(path2dir_fig_unfilt + os.sep + figname + '.png')
        plt.close('all')

    # PLOT WDIR VS. YAW SELECTED RANGE FOR TURB PAIRS SEPARATELY
    if bool_plot_wdir_vs_yaw_selected_range:

        n_cols = 1
        n_rows = 2

        figsize_cm = [12, 24]
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=cm2inch(figsize_cm))

        for pair_n in range(n_turb_pairs):
            ax = axes[pair_n]

            for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                ax.plot(df_dict[turb_key]['WDir'],
                        df_dict[turb_key]['Yaw'],
                        marker_dict_by_turb[turb_key], color=color_dict_by_turb[turb_key],
                        fillstyle='none',
                        mew=0.3, ms=2,
                        # label=var_key + '_' + turb_key
                        label=turb_key
                        )

            ax.plot(np.linspace(0, 360, 100),
                    np.linspace(0, 360, 100),
                    '--', color='grey')

            ax.set_xlim([wdir_min_plot, wdir_max_plot])
            ax.set_ylim([wdir_min_plot, wdir_max_plot])

            ax.set_xlabel('WDir')
            ax.set_ylabel('Yaw')

            ax.grid()
            ax.legend()

        fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
        fig.tight_layout()

        figname = 'WDir_vs_Yaw_select_range_' + resample_str
        fig.savefig(path2dir_fig_unfilt + os.sep + figname + '.png')
        plt.close('all')

    print('-- finish plotting unfiltered data')


def plot_df_ctrl_turb_dict_overview(
        df_ctrl_turb_dict,
        var_keys_to_process,
        turb_keys_to_process,
        turb_keys_split_by_pair,
        resample_str,
        ctrl_keys,
        path2dir_fig,
        marker_dict_by_turb,
        color_dict_by_turb,
        wdir_min_plot,
        wdir_max_plot,
        bool_plot_overview,
        bool_plot_overview_split,
        bool_plot_wdir_selected_range,
        bool_plot_wdir_vs_yaw,
        bool_plot_wdir_vs_yaw_selected_range
):

    print('-- start plotting data overview')

    date_format_str = "%m-%d %H:%M"

    n_selected_vars = len(var_keys_to_process)
    n_turb_pairs = len(turb_keys_split_by_pair)

    for ctrl_key in ctrl_keys:

        df_filt_dict = df_ctrl_turb_dict[ctrl_key]

        title_add_str = f'ctrl_{ctrl_key}'

        path2dir_fig = path2dir_fig

        if bool_plot_overview:
            n_cols = 1
            n_rows = n_selected_vars

            figsize_cm = [16, 4 * n_rows]
            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, sharex=True, figsize=cm2inch(figsize_cm))
            if n_selected_vars == 1:
                axes = [axes]

            for var_n, var_key in enumerate(var_keys_to_process):

                ax = axes[var_n]

                for turb_n, turb_key in enumerate(turb_keys_to_process):
                    if var_key == 'ControlSwitch' and turb_key not in ['T4', 'T6']:
                        pass
                    else:
                        ax.plot(df_filt_dict[turb_key].index,
                                df_filt_dict[turb_key][var_key],
                                marker_dict_by_turb[turb_key], color=color_dict_by_turb[turb_key],
                                fillstyle='none',
                                mew=0.3, ms=2,
                                # label=var_key + '_' + turb_key
                                label=turb_key
                                )

                    if var_key in ['Power']:
                        ax.set_ylim([0, 5500])
                if var_n == n_selected_vars - 1:
                    myFmt = mpl.dates.DateFormatter(date_format_str)
                    ax.xaxis.set_major_formatter(myFmt)
                    ax.tick_params(axis='x', labelrotation=90)

                ax.set_ylabel(var_key)

                ax.grid()
                ax.legend()

            fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
            fig.tight_layout()

            figname = 'Filt_Overview_plot_' + resample_str + '_' + title_add_str
            fig.savefig(path2dir_fig + os.sep + figname + '.png')
            plt.close('all')

            print('plotted overview')

        # ------------------------------------------
        if bool_plot_overview_split:
            n_cols = 2
            n_rows = n_selected_vars

            figsize_cm = [16, 4 * n_rows]
            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, sharex=True,
                figsize=cm2inch(figsize_cm)
            )

            if n_selected_vars == 1:
                axes = [axes]

            for var_n, var_key in enumerate(var_keys_to_process):
                for pair_n in range(n_turb_pairs):
                    ax = axes[var_n][pair_n]

                    for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                        if var_key == 'ControlSwitch' and turb_key not in ['T4', 'T6']:
                            pass
                        else:
                            ax.plot(df_filt_dict[turb_key].index,
                                    df_filt_dict[turb_key][var_key],
                                    marker_dict_by_turb[turb_key],
                                    color=color_dict_by_turb[turb_key],
                                    fillstyle='none',
                                    mew=0.3, ms=2,
                                    # label=var_key + '_' + turb_key
                                    label=turb_key
                                    )

                    if var_key in ['Power']:
                        ax.set_ylim([0, 5500])
                    if var_n == n_selected_vars - 1:
                        myFmt = mpl.dates.DateFormatter(date_format_str)
                        ax.xaxis.set_major_formatter(myFmt)
                        ax.tick_params(axis='x', labelrotation=90)

                    ax.set_ylabel(var_key)

                    ax.grid()
                    ax.legend()

            fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
            fig.tight_layout()

            figname = ('Filt_Overview_plot_split_pairs_' + resample_str + '_'
                       + title_add_str)
            fig.savefig(path2dir_fig + os.sep + figname + '.png')
            plt.close('all')

            print('plotted overview split')

        # ---------------------------------------
        if bool_plot_wdir_selected_range:
            var_keys_to_process2 = [
                'WDir',
                'Yaw',
                'Power'
            ]

            n_cols = 2
            n_rows = len(var_keys_to_process2)

            figsize_cm = [16, 8 * n_rows]
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True,
                                     figsize=cm2inch(figsize_cm))

            for var_n, var_key in enumerate(var_keys_to_process2):
                for pair_n in range(n_turb_pairs):
                    ax = axes[var_n][pair_n]

                    for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                        if var_key == 'ControlSwitch' and turb_key not in ['T4', 'T6']:
                            pass
                        else:
                            ax.plot(df_filt_dict[turb_key].index,
                                    df_filt_dict[turb_key][var_key],
                                    marker_dict_by_turb[turb_key],
                                    color=color_dict_by_turb[turb_key],
                                    fillstyle='none',
                                    mew=0.3, ms=2,
                                    # label=var_key + '_' + turb_key
                                    label=turb_key
                                    )

                    if var_key in ['Power']:
                        ax.set_ylim([0, 5500])
                    else:
                        ax.set_ylim([285, 340])

                    if var_n == n_rows - 1:
                        myFmt = mpl.dates.DateFormatter(date_format_str)
                        ax.xaxis.set_major_formatter(myFmt)
                        ax.tick_params(axis='x', labelrotation=90)

                    ax.set_ylabel(var_key)

                    ax.grid()
                    ax.legend()

            fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
            fig.tight_layout()

            figname = ('Filt_WDir_Yaw_selected_range_' + resample_str
                       + '_' + title_add_str)
            fig.savefig(path2dir_fig + os.sep + figname + '.png')
            plt.close('all')

            print('plotted Wdir and Yaw for selected range')

        # ----------------------------------------
        if bool_plot_wdir_vs_yaw:
            n_cols = 1
            n_rows = 2

            figsize_cm = [12, 24]
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=cm2inch(figsize_cm))

            for pair_n in range(n_turb_pairs):
                ax = axes[pair_n]

                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):

                    ax.plot(df_filt_dict[turb_key]['WDir'],
                            df_filt_dict[turb_key]['Yaw'],
                            marker_dict_by_turb[turb_key], color=color_dict_by_turb[turb_key],
                            fillstyle='none',
                            mew=0.3, ms=2,
                            # label=var_key + '_' + turb_key
                            label=turb_key
                            )

                ax.plot(np.linspace(0, 360, 100),
                        np.linspace(0, 360, 100),
                        '--', color='grey')

                ax.set_xlabel('WDir')
                ax.set_ylabel('Yaw')

                ax.grid()
                ax.legend()

            fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
            fig.tight_layout()

            figname = 'Filt_WDir_vs_Yaw_' + resample_str + '_' + title_add_str
            fig.savefig(path2dir_fig + os.sep + figname + '.png')
            plt.close('all')

            print('plotted Wdir vs. Yaw')

        # -------------------------------------------------------
        if bool_plot_wdir_vs_yaw_selected_range:
            n_cols = 1
            n_rows = 2

            figsize_cm = [12, 24]
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=cm2inch(figsize_cm))

            for pair_n in range(n_turb_pairs):
                ax = axes[pair_n]

                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                    ax.plot(df_filt_dict[turb_key]['WDir'],
                            df_filt_dict[turb_key]['Yaw'],
                            marker_dict_by_turb[turb_key], color=color_dict_by_turb[turb_key],
                            fillstyle='none',
                            mew=0.3, ms=2,
                            # label=var_key + '_' + turb_key
                            label=turb_key
                            )

                ax.plot(np.linspace(0, 360, 100),
                        np.linspace(0, 360, 100),
                        '--', color='grey')

                ax.set_xlim([wdir_min_plot, wdir_max_plot])
                ax.set_ylim([wdir_min_plot, wdir_max_plot])

                ax.set_xlabel('WDir')
                ax.set_ylabel('Yaw')

                ax.grid()
                ax.legend()

            fig.suptitle(f'Sampling {resample_str}, ' + title_add_str)
            fig.tight_layout()

            figname = ('Filt_WDir_vs_Yaw_select_range_' + resample_str
                       + '_' + title_add_str)
            fig.savefig(path2dir_fig + os.sep + figname + '.png')
            plt.close('all')

            print('plotted Wdir vs. Yaw for selected range')

    print('--- finish plotting data overview')


def plot_binned_1D_per_turb(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_x,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize,
        cp_min_down, cp_max_down, cp_min_up, cp_max_up, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig,
):
    dash_type_per_ctrl_dict = {
        'on': '-',
        'off': '--'
    }

    # start plotting from this index
    start_idx = 0

    n_turb_pairs = len(turb_keys_split_by_pair)

    n_turbs = 2 * n_turb_pairs

    turb_keys_down = [tk[idx_downstream] for tk in turb_keys_split_by_pair]

    bin_centers_per_pair = []

    for pair_n in range(n_turb_pairs):

        bins = bins_per_pair[pair_n]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_centers_per_pair.append(bin_centers)

    fig, axes = plt.subplots(
        nrows=1, ncols=n_turbs, sharex=False,
        figsize=figsize,
    )

    turb_n = 0

    for pair_n in range(n_turb_pairs):

        bins = bins_per_pair[pair_n]
        bin_centers = bin_centers_per_pair[pair_n]

        for turb_key in turb_keys_split_by_pair[pair_n]:

            ax = axes[turb_n]

            if turb_key in turb_keys_down:
                ax.vlines(bins, cp_min_down, cp_max_down,
                          colors='k', linestyles=VLS, linewidths=VLW,
                          alpha=0.5, zorder=0)
            else:
                ax.vlines(bins, cp_min_up, cp_max_up,
                          colors='k', linestyles=VLS, linewidths=VLW,
                          alpha=0.5, zorder=0)

            m_cnt = 0

            for ctrl_key in ['on', 'off']:

                var_binned_mean = df_binned_1D_mean_dict[ctrl_key][turb_key][var_key_to_plot]
                var_binned_std = df_binned_1D_std_dict[ctrl_key][turb_key][var_key_to_plot]

                COLOR = config.color_list[m_cnt]
                ax.plot(
                    bin_centers[start_idx:],
                    var_binned_mean.to_numpy(),
                    dash_type_per_ctrl_dict[ctrl_key] + marker_list[m_cnt],
                    color=COLOR,
                    ms=MS,
                    fillstyle='none', mew=0.4,
                    label=f'ctrl {ctrl_key}'
                )

                ax.fill_between(
                    bin_centers[start_idx:],
                    (var_binned_mean.to_numpy()[start_idx:]
                     - var_binned_std.to_numpy()[start_idx:]),
                    (var_binned_mean.to_numpy()[start_idx:]
                     + var_binned_std.to_numpy()[start_idx:]),
                    color=COLOR,
                    alpha=ALPHA_ERR,
                )

                m_cnt += 1

            ax.set_xlim([bins[0] - 1, bins[-1] + 1])
            # ax.set_xticks(bin_centers)

            if turb_key in turb_keys_down:
                ax.set_ylim([cp_min_down, cp_max_down])
                ax.set_yticks(np.arange(cp_min_down, cp_max_down + 0.01, cp_plot_bin))
            else:
                ax.set_ylim([cp_min_up, cp_max_up])
                ax.set_yticks(np.arange(cp_min_up, cp_max_up + 0.01, cp_plot_bin))

            ax.set_xlabel(f'{var_key_x} [{var_unit_x}]')

            ax.grid(axis='y')
            ax.legend()

            if turb_n == 0:
                ax.set_ylabel(f'{var_label_y} [{var_unit_y}]')
                # fig.legend(
                #     ncols=len(bin_centers), loc='upper center',
                #     bbox_to_anchor=(0.5, 1.03)
                # )

            ax.set_title(f'{turb_key}')

            turb_n += 1

    # save fig
    fig.tight_layout()

    figname = f'{var_key_to_plot}_binned_1D_vs_{var_key_x}_turbs'
    if bool_save_fig:
        fig.savefig(path2dir_fig + os.sep + figname + '.png',
                    bbox_inches='tight')
        plt.close('all')


def plot_binned_2D_per_turb(
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
        figsize,
        cp_min_down, cp_max_down, cp_min_up, cp_max_up, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig,
):
    dash_type_per_ctrl_dict = {
        'on': '-',
        'off': '--'
    }

    n_turb_pairs = len(turb_keys_split_by_pair)

    n_turbs = 2 * n_turb_pairs

    turb_keys_down = [tk[idx_downstream] for tk in turb_keys_split_by_pair]
    turb_keys_up = [tk[idx_upstream] for tk in turb_keys_split_by_pair]

    wspd_bin_centers_per_pair = []
    wdir_bin_centers_per_pair = []

    for pair_n in range(n_turb_pairs):

        wspd_bins = wspd_bins_per_pair[pair_n]
        wspd_bin_centers = 0.5 * (wspd_bins[:-1] + wspd_bins[1:])
        wspd_bin_centers_per_pair.append(wspd_bin_centers)

        wdir_bins = wdir_bins_per_pair[pair_n]
        wdir_bin_centers = 0.5 * (wdir_bins[:-1] + wdir_bins[1:])
        wdir_bin_centers_per_pair.append(wdir_bin_centers)

    fig, axes = plt.subplots(
        nrows=2, ncols=n_turbs, sharex=False,
        figsize=figsize,
        height_ratios=[4, 1]
    )

    turb_n = 0

    for pair_n in range(n_turb_pairs):

        turb_key_up = turb_keys_up[pair_n]

        wdir_bins = wdir_bins_per_pair[pair_n]
        # wspd_bins = wspd_bins_per_pair[pair_n]

        wdir_bin_centers = wdir_bin_centers_per_pair[pair_n]
        wspd_bin_centers = wspd_bin_centers_per_pair[pair_n]

        for turb_key in turb_keys_split_by_pair[pair_n]:

            ax = axes[0, turb_n]

            if turb_key in turb_keys_down:
                ax.vlines(wdir_bins, cp_min_down, cp_max_down,
                          colors='k', linestyles=VLS, linewidths=VLW,
                          alpha=0.5, zorder=0)
            else:
                ax.vlines(wdir_bins, cp_min_up, cp_max_up,
                          colors='k', linestyles=VLS, linewidths=VLW,
                          alpha=0.5, zorder=0)

            for ctrl_key in ['on', 'off']:

                str_add = f'ctrl_{ctrl_key}'

                var_binned_mean = df_binned_2D_mean_dict[ctrl_key][turb_key][var_key_to_plot]
                var_binned_std = df_binned_2D_std_dict[ctrl_key][turb_key][var_key_to_plot]

                # prepare lists
                var_binned_mean_vs_wdir_per_wspd_list = []
                var_binned_std_vs_wdir_per_wspd_list = []

                wspd_index = df_binned_2D_mean_dict[ctrl_key][turb_key].index.get_level_values(
                    level='WSpeed').unique()
                wdir_index = df_binned_2D_std_dict[ctrl_key][turb_key].index.get_level_values(
                    level='WDir').unique()

                for wspd_bin_n in range(len(wspd_bin_centers)):
                    int_x = wspd_index[wspd_bin_n]
                    var_binned_mean_vs_wdir = var_binned_mean.loc[[int_x]]
                    var_binned_mean_vs_wdir_per_wspd_list.append(var_binned_mean_vs_wdir)

                    int_x = var_binned_std.index.get_level_values(0).unique()[wspd_bin_n]
                    var_binned_std_vs_wdir = var_binned_std.loc[[int_x]]
                    var_binned_std_vs_wdir_per_wspd_list.append(var_binned_std_vs_wdir)

                # plot Cp vs wdir for several wspd regimes
                start_idx = 0
                m_cnt = 0

                for wspd_bin_n in range(len(wspd_bin_centers)):

                    var_binned_mean_vs_wdir = var_binned_mean_vs_wdir_per_wspd_list[wspd_bin_n]
                    var_binned_std_vs_wdir = var_binned_std_vs_wdir_per_wspd_list[wspd_bin_n]

                    COLOR = config.color_list[m_cnt]
                    ax.plot(
                        wdir_bin_centers[start_idx:],
                        var_binned_mean_vs_wdir.to_numpy()[start_idx:],
                        dash_type_per_ctrl_dict[ctrl_key] + marker_list[m_cnt],
                        color=COLOR,
                        ms=MS,
                        fillstyle='none', mew=0.4,
                        label=f'$u$={wspd_bin_centers[wspd_bin_n]:.1f} m/s, {str_add}'
                    )

                    ax.fill_between(
                        wdir_bin_centers[start_idx:],
                        (var_binned_mean_vs_wdir.to_numpy()[start_idx:]
                         - var_binned_std_vs_wdir.to_numpy()[start_idx:]),
                        (var_binned_mean_vs_wdir.to_numpy()[start_idx:]
                         + var_binned_std_vs_wdir.to_numpy()[start_idx:]),
                        color=COLOR,
                        alpha=ALPHA_ERR,
                    )

                    m_cnt += 1

            ax.set_xlim([wdir_bins[0]-1, wdir_bins[-1]+1])

            if turb_key in turb_keys_down:
                ax.set_ylim([cp_min_down, cp_max_down])
                ax.set_yticks(np.arange(cp_min_down, cp_max_down + 0.01, cp_plot_bin))
            else:
                ax.set_ylim([cp_min_up, cp_max_up])
                ax.set_yticks(np.arange(cp_min_up, cp_max_up + 0.01, cp_plot_bin))

            if turb_n == 0:
                ax.set_ylabel(f'{var_label_y} [{var_unit_y}]')
                fig.legend(
                    ncols=len(wspd_bin_centers), loc='upper center',
                    bbox_to_anchor=(0.5, 1.06)
                )

            ax.set_title(f'{turb_key}')

            # plot yaw_mis from yaw_table
            ax = axes[1, turb_n]

            ax.plot(
                wdir_setpoint_per_pair[pair_n],
                yaw_setpoint_per_pair[pair_n] - wdir_setpoint_per_pair[pair_n],
                label=turb_key_up)
            ax.vlines(wdir_bins, -10, 10, colors='k',
                      linestyles=VLS, linewidths=VLW, alpha=0.5)
            ax.set_yticks(np.arange(-10, 11, 5))
            ax.legend()
            ax.grid()
            ax.set_xlabel(r'WDir [deg]')
            if turb_n == 0:
                ax.set_ylabel(r'Yaw mis. table [deg]')

            turb_n += 1

    # save fig
    fig.tight_layout()

    figname = f'{var_key_to_plot}_binned_2D_vs_wdir_per_wspd_turbs'
    if bool_save_fig:
        fig.savefig(path2dir_fig + os.sep + figname + '.png',
                    bbox_inches='tight')
        plt.close('all')

def plot_binned_1D_per_pair(
        df_binned_1D_mean_dict,
        df_binned_1D_std_dict,
        bins_per_pair,
        var_key_x,
        var_unit_x,
        var_key_to_plot,
        var_label_y,
        var_unit_y,
        turb_keys_split_by_pair,
        idx_upstream, idx_downstream,
        bool_save_fig,
        figsize,
        cp_min, cp_max, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig,
):
    dash_type_per_ctrl_dict = {
        'on': '-',
        'off': '--'
    }

    # start plotting from this index
    start_idx = 0

    n_turb_pairs = len(turb_keys_split_by_pair)

    turb_keys_up = [tk[idx_upstream] for tk in turb_keys_split_by_pair]
    turb_keys_down = [tk[idx_downstream] for tk in turb_keys_split_by_pair]

    bin_centers_per_pair = []

    for pair_n in range(n_turb_pairs):

        bins = bins_per_pair[pair_n]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_centers_per_pair.append(bin_centers)

    fig, axes = plt.subplots(
        nrows=1, ncols=n_turb_pairs, sharex=False,
        figsize=figsize,
    )

    # turb_n = 0

    for pair_n in range(n_turb_pairs):

        ax = axes[pair_n]

        bins = bins_per_pair[pair_n]
        bin_centers = bin_centers_per_pair[pair_n]

        ax.vlines(
            bins, cp_min, cp_max,
            colors='k', linestyles=VLS, linewidths=VLW,
            alpha=0.5, zorder=0
        )

        m_cnt = 0

        for ctrl_key in ['on', 'off']:

            turb_key_up = turb_keys_up[pair_n]
            turb_key_down = turb_keys_down[pair_n]

            var_binned_mean = 0.5 * (
                df_binned_1D_mean_dict[ctrl_key][turb_key_up][var_key_to_plot]
                + df_binned_1D_mean_dict[ctrl_key][turb_key_down][var_key_to_plot]
            )

            var_binned_std = np.sqrt(
                df_binned_1D_std_dict[ctrl_key][turb_key_up][var_key_to_plot]**2
                + df_binned_1D_std_dict[ctrl_key][turb_key_down][var_key_to_plot]**2
            )

            COLOR = config.color_list[m_cnt]
            ax.plot(
                bin_centers[start_idx:],
                var_binned_mean.to_numpy(),
                dash_type_per_ctrl_dict[ctrl_key] + marker_list[m_cnt],
                color=COLOR,
                ms=MS,
                fillstyle='none', mew=0.4,
                label=f'ctrl {ctrl_key}'
            )

            ax.fill_between(
                bin_centers[start_idx:],
                (var_binned_mean.to_numpy()[start_idx:]
                 - var_binned_std.to_numpy()[start_idx:]),
                (var_binned_mean.to_numpy()[start_idx:]
                 + var_binned_std.to_numpy()[start_idx:]),
                color=COLOR,
                alpha=ALPHA_ERR,
            )

            m_cnt += 1

            ax.set_xlim([bins[0] - 1, bins[-1] + 1])
            # ax.set_xticks(bin_centers)

            ax.set_ylim([cp_min, cp_max])
            ax.set_yticks(np.arange(cp_min, cp_max + 0.01, cp_plot_bin))

            ax.set_xlabel(f'{var_key_x} [{var_unit_x}]')

            ax.grid(axis='y')
            ax.legend()

            if pair_n == 0:
                ax.set_ylabel(f'{var_label_y} [{var_unit_y}]')
                # fig.legend(
                #     ncols=len(bin_centers), loc='upper center',
                #     bbox_to_anchor=(0.5, 1.03)
                # )

            ax.set_title(f'Pair {pair_n + 1:.0f}')

            # turb_n += 1

    # save fig
    fig.tight_layout()

    figname = f'{var_key_to_plot}_binned_1D_vs_{var_key_x}_pairs'
    if bool_save_fig:
        fig.savefig(path2dir_fig + os.sep + figname + '.png',
                    bbox_inches='tight')
        plt.close('all')


def plot_binned_2D_per_pair(
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
        figsize,
        cp_min, cp_max, cp_plot_bin,
        VLS, VLW, MS, ALPHA_ERR,
        path2dir_fig,
):
    dash_type_per_ctrl_dict = {
        'on': '-',
        'off': '--'
    }

    n_turb_pairs = len(turb_keys_split_by_pair)

    n_turbs = 2 * n_turb_pairs

    turb_keys_down = [tk[idx_downstream] for tk in turb_keys_split_by_pair]
    turb_keys_up = [tk[idx_upstream] for tk in turb_keys_split_by_pair]

    wspd_bin_centers_per_pair = []
    wdir_bin_centers_per_pair = []

    for pair_n in range(n_turb_pairs):

        wspd_bins = wspd_bins_per_pair[pair_n]
        wspd_bin_centers = 0.5 * (wspd_bins[:-1] + wspd_bins[1:])
        wspd_bin_centers_per_pair.append(wspd_bin_centers)

        wdir_bins = wdir_bins_per_pair[pair_n]
        wdir_bin_centers = 0.5 * (wdir_bins[:-1] + wdir_bins[1:])
        wdir_bin_centers_per_pair.append(wdir_bin_centers)

    fig, axes = plt.subplots(
        nrows=2, ncols=n_turb_pairs, sharex=False,
        figsize=figsize,
        height_ratios=[4, 1]
    )

    # turb_n = 0

    for pair_n in range(n_turb_pairs):

        ax = axes[0, pair_n]

        turb_key_down = turb_keys_down[pair_n]
        turb_key_up = turb_keys_up[pair_n]

        wdir_bins = wdir_bins_per_pair[pair_n]
        wspd_bins = wspd_bins_per_pair[pair_n]

        wdir_bin_centers = wdir_bin_centers_per_pair[pair_n]
        wspd_bin_centers = wspd_bin_centers_per_pair[pair_n]

        ax.vlines(wdir_bins, cp_min, cp_max,
                  colors='k', linestyles=VLS, linewidths=VLW,
                  alpha=0.5, zorder=0)

        for ctrl_key in ['on', 'off']:

            str_add = f'ctrl {ctrl_key}'

            # var_binned_mean = df_binned_2D_mean_dict[ctrl_key][turb_key][var_key_to_plot]
            # var_binned_std = df_binned_2D_std_dict[ctrl_key][turb_key][var_key_to_plot]

            var_binned_mean = 0.5 * (
                df_binned_2D_mean_dict[ctrl_key][turb_key_up][var_key_to_plot]
                + df_binned_2D_mean_dict[ctrl_key][turb_key_down][var_key_to_plot]
            )

            var_binned_std = np.sqrt(
                df_binned_2D_std_dict[ctrl_key][turb_key_up][var_key_to_plot]**2
                + df_binned_2D_std_dict[ctrl_key][turb_key_down][var_key_to_plot]**2
            )

            # prepare lists
            var_binned_mean_vs_wdir_per_wspd_list = []
            var_binned_std_vs_wdir_per_wspd_list = []

            wspd_index = df_binned_2D_mean_dict[ctrl_key][turb_key_up].index.get_level_values(
                level='WSpeed').unique()
            # wdir_index = df_binned_2D_std_dict[ctrl_key][turb_key_up].index.get_level_values(
            #     level='WDir').unique()

            for wspd_bin_n in range(len(wspd_bin_centers)):
                int_x = wspd_index[wspd_bin_n]
                var_binned_mean_vs_wdir = var_binned_mean.loc[[int_x]]
                var_binned_mean_vs_wdir_per_wspd_list.append(var_binned_mean_vs_wdir)

                int_x = var_binned_std.index.get_level_values(0).unique()[wspd_bin_n]
                var_binned_std_vs_wdir = var_binned_std.loc[[int_x]]
                var_binned_std_vs_wdir_per_wspd_list.append(var_binned_std_vs_wdir)

            # plot Cp vs wdir for several wspd regimes
            start_idx = 0
            m_cnt = 0

            for wspd_bin_n in range(len(wspd_bin_centers)):

                var_binned_mean_vs_wdir = var_binned_mean_vs_wdir_per_wspd_list[wspd_bin_n]
                var_binned_std_vs_wdir = var_binned_std_vs_wdir_per_wspd_list[wspd_bin_n]

                COLOR = config.color_list[m_cnt]
                ax.plot(
                    wdir_bin_centers[start_idx:],
                    var_binned_mean_vs_wdir.to_numpy()[start_idx:],
                    dash_type_per_ctrl_dict[ctrl_key] + marker_list[m_cnt],
                    color=COLOR,
                    ms=MS,
                    fillstyle='none', mew=0.4,
                    label=f'$u$={wspd_bin_centers[wspd_bin_n]:.1f} m/s, {str_add}'
                )

                ax.fill_between(
                    wdir_bin_centers[start_idx:],
                    (var_binned_mean_vs_wdir.to_numpy()[start_idx:]
                     - var_binned_std_vs_wdir.to_numpy()[start_idx:]),
                    (var_binned_mean_vs_wdir.to_numpy()[start_idx:]
                     + var_binned_std_vs_wdir.to_numpy()[start_idx:]),
                    color=COLOR,
                    alpha=ALPHA_ERR,
                )

                m_cnt += 1

            ax.set_xlim([wdir_bins[0] - 1, wdir_bins[-1] + 1])

            ax.set_ylim([cp_min, cp_max])
            ax.set_yticks(np.arange(cp_min, cp_max + 0.01, cp_plot_bin))

        if pair_n == 0:
            ax.set_ylabel(f'{var_label_y} [{var_unit_y}]')
            fig.legend(
                ncols=1, loc='upper left',
                bbox_to_anchor=(1.01, 0.95)
            )

        ax.set_title(f'Pair {pair_n + 1:.0f}')

        # plot yaw_mis from yaw_table
        ax = axes[1, pair_n]

        ax.plot(
            wdir_setpoint_per_pair[pair_n],
            yaw_setpoint_per_pair[pair_n] - wdir_setpoint_per_pair[pair_n],
            label=turb_key_up)
        ax.vlines(wdir_bins, -10, 10, colors='k',
                  linestyles=VLS, linewidths=VLW, alpha=0.5)
        ax.set_yticks(np.arange(-10, 11, 5))
        ax.legend()
        ax.grid()
        ax.set_xlabel(r'WDir [deg]')
        if pair_n == 0:
            ax.set_ylabel(r'Yaw mis. table [deg]')

            # turb_n += 1

    # save fig
    fig.tight_layout()

    figname = f'{var_key_to_plot}_binned_2D_vs_wdir_per_wspd_pairs'
    if bool_save_fig:
        fig.savefig(path2dir_fig + os.sep + figname + '.png',
                    bbox_inches='tight')
        plt.close('all')


def plot_valid_intervals_wdir_vs_t(
        df_filt_dict,
        n_bin_dims,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_used_for_filter_binning_per_pair_list,
        var_key_used_for_filter_binning_list,
        wdir_bins_per_pair,
        ctrl_keys,
        max_gap_duration_s,
        bool_use_std_of_mean,
        path2dir_yaw_table,
        fname_yaw_table,
        path2dir_fig_int_base,
        bool_clear_interval_figs_from_dir,
        figsize_int_ts,
        resample_interval_s
):

    print(f'-- start plotting intervals {n_bin_dims:.0f}D binned')

    n_turb_pairs = len(turb_keys_split_by_pair)

    yawmis_plot_lim = [-15, 15]
    date_format_str = "%m-%d %H:%M:%S"
    plot_limit_shift_wdir = 5
    VLS = ':'
    VLW = 1

    interval_dict = proc.split_df_filt_dict_into_time_intervals(
        df_filt_dict,
        n_bin_dims,
        var_keys_to_bin,
        turb_keys_split_by_pair,
        turb_keys_up,
        turb_keys_down,
        idx_upstream,
        bins_used_for_filter_binning_per_pair_list,
        var_key_used_for_filter_binning_list,
        ctrl_keys,
        max_gap_duration_s,
        bool_use_std_of_mean
    )

    # -- plot

    calc_yaw_setpoint_per_pair = proc.gen_yaw_table_interp(
            path2dir_yaw_table,
            fname_yaw_table,
    )

    for ctrl_key in ctrl_keys:

        title_add_str = 'ctrl_' + ctrl_key

        if n_bin_dims == 1:
            var_key_used_for_filter_binning = var_key_used_for_filter_binning_list[0]
            p2d_str_add = f'1D_{var_key_used_for_filter_binning}'
        if n_bin_dims == 2:
            var_key_1_used_for_filter_binning = var_key_used_for_filter_binning_list[0]
            var_key_2_used_for_filter_binning = var_key_used_for_filter_binning_list[1]
            p2d_str_add = \
                f'2D_{var_key_1_used_for_filter_binning}_{var_key_2_used_for_filter_binning}'

        path2dir_fig_int = (
                path2dir_fig_int_base + os.sep + p2d_str_add
                + os.sep + title_add_str)
        make_dir_if_not_exists(path2dir_fig_int)

        if bool_clear_interval_figs_from_dir:
            delete_all_files_from_dir(path2dir_fig_int)

        for pair_n in range(n_turb_pairs):

            wdir_bins = wdir_bins_per_pair[pair_n]

            wdir_min = wdir_bins[0]
            wdir_max = wdir_bins[-1]

            turb_keys_in_pair = turb_keys_split_by_pair[pair_n]
            calc_yaw_setpoint_x = calc_yaw_setpoint_per_pair[pair_n]

            for turb_key in [turb_keys_up[pair_n]]:

                df_list = interval_dict[ctrl_key][turb_key]

                for int_n, df_x in enumerate(df_list):

                    wspd_mean = df_x['WSpeed'].mean()

                    # wdir and yaw vs t
                    fig, ax = plt.subplots(1, 1, figsize=figsize_int_ts)

                    time_index_x = df_x.index.to_series()

                    ax.plot(
                        time_index_x,
                        df_x['WDir'],
                        label='WDir'
                    )

                    ax.plot(
                        time_index_x,
                        df_x['Yaw'],
                        label='Yaw'
                    )

                    ax.plot(
                        time_index_x,
                        calc_yaw_setpoint_x(df_x['WDir']),
                        '--', c='k', alpha=0.6,
                        label='Yaw table'
                    )

                    ax.hlines(wdir_bins, time_index_x.iloc[0], time_index_x.iloc[-1],
                              colors='grey', linestyles=VLS, linewidths=VLW,
                              alpha=0.7, zorder=0)

                    myFmt = mpl.dates.DateFormatter(date_format_str)
                    ax.xaxis.set_major_formatter(myFmt)
                    ax.tick_params(axis='x', labelrotation=90)
                    ax.set_ylim(wdir_min - plot_limit_shift_wdir, wdir_max + plot_limit_shift_wdir)
                    ax.legend()
                    ax.set_ylabel('Angle [deg]')
                    ax.set_title(
                        f'{turb_key}, sampling = {resample_interval_s:.0f} s,\n'
                        + f'WSpeed mean = {wspd_mean:.1f} m/s'
                    )

                    fig.tight_layout()
                    fig.savefig(
                        path2dir_fig_int + os.sep + f'WDir_{turb_key}_interval_{int_n:.0f}'
                    )
                    plt.close()

                    # yaw misalignment plot
                    fig, ax = plt.subplots(1, 1, figsize=figsize_int_ts)

                    time_index_x = df_x.index.to_series()

                    ax.plot(
                        time_index_x,
                        df_x['Yaw'] - df_x['WDir'],
                        label='Yaw mis. data'
                    )

                    ax.plot(
                        time_index_x,
                        calc_yaw_setpoint_x(df_x['WDir']) - df_x['WDir'],
                        '--', c='k', alpha=0.6,
                        label='Yaw mis. setp. table'
                    )

                    myFmt = mpl.dates.DateFormatter(date_format_str)
                    ax.xaxis.set_major_formatter(myFmt)
                    ax.tick_params(axis='x', labelrotation=90)
                    ax.set_ylim(yawmis_plot_lim)
                    ax.legend()
                    ax.set_ylabel('Angle [deg]')
                    # ax.set_title(f'{turb_key}, sampling = {resample_interval_s:.0f} s')
                    ax.set_title(
                        f'{turb_key}, sampling = {resample_interval_s:.0f} s,\n'
                        + f'WSpeed mean = {wspd_mean:.1f} m/s'
                    )

                    fig.tight_layout()
                    fig.savefig(
                        path2dir_fig_int + os.sep + f'YawMis_{turb_key}_interval_{int_n:.0f}'
                    )
                    plt.close()

    print(f'-- finish plotting intervals {n_bin_dims:.0f}D binned')


def plot_yaw_table(
        path2dir_yaw_table,
        fname_yaw_table,
        path2dir_fig_yaw_table,
        figsize_yaw_table,
        wdir_min,
        wdir_max,
        d_wdir_plot
):

    print('-- start plotting yaw table')

    calc_yaw_setpoint_pair_1, calc_yaw_setpoint_pair_2 = proc.gen_yaw_table_interp(
        path2dir_yaw_table,
        fname_yaw_table
    )

    wdir_ = np.linspace(0, 360, 1000)

    fig, ax = plt.subplots(figsize=figsize_yaw_table)

    ax.plot(wdir_, calc_yaw_setpoint_pair_1(wdir_), label='T4')
    ax.plot(wdir_, calc_yaw_setpoint_pair_2(wdir_), label='T6')

    ax.plot(
        [0, 360],
        [0, 360],
        '--', c='grey', alpha=0.6
    )

    ax.set_aspect('equal')

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)

    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(0, 361, 30))

    ax.set_xlabel('WDir')
    ax.set_ylabel('Yaw')

    ax.legend()
    ax.grid()
    fig.tight_layout()

    fig.savefig(path2dir_fig_yaw_table + os.sep + 'yaw_lookup_table_full_range.png')

    ax.set_xlim(wdir_min, wdir_max)
    ax.set_ylim(wdir_min, wdir_max)

    ax.set_xticks(np.arange(wdir_min, wdir_max + 0.1, d_wdir_plot))
    ax.set_yticks(np.arange(wdir_min, wdir_max + 0.1, d_wdir_plot))

    ax.set_aspect('equal')

    fig.tight_layout()

    fig.savefig(path2dir_fig_yaw_table + os.sep + 'yaw_lookup_table_select_range.png')

    # --- plot yaw misalignment
    fig, ax = plt.subplots(figsize=figsize_yaw_table)

    ax.plot(wdir_, calc_yaw_setpoint_pair_1(wdir_) - wdir_, label='T4')
    ax.plot(wdir_, calc_yaw_setpoint_pair_2(wdir_) - wdir_, label='T6')

    ax.set_xlim(wdir_min, wdir_max)

    ax.set_xticks(np.arange(wdir_min, wdir_max + 0.1, d_wdir_plot))
    # ax.tick_params(axis='x', which='major', labelsize=6)
    ax.set_yticks(np.arange(-10, 11, 1))

    ax.set_xlabel('WDir')
    ax.set_ylabel('Yaw')
    ax.grid()

    fig.tight_layout()

    fig.savefig(path2dir_fig_yaw_table + os.sep + 'yaw_misalign_lookup_table_select_range.png')

    # --- plot yaw misalignment fine
    figsize_yaw_table_fine = cm2inch(30, 16)
    fig, ax = plt.subplots(figsize=figsize_yaw_table_fine)

    ax.plot(wdir_, calc_yaw_setpoint_pair_1(wdir_) - wdir_, label='T4')
    ax.plot(wdir_, calc_yaw_setpoint_pair_2(wdir_) - wdir_, label='T6')

    ax.set_xlim(wdir_min, wdir_max)

    ax.set_xticks(np.arange(wdir_min, wdir_max + 0.1, 1))
    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.set_yticks(np.arange(-10, 11, 1))

    ax.set_xlabel('WDir')
    ax.set_ylabel('Yaw')
    ax.grid()

    fig.tight_layout()

    fig.savefig(path2dir_fig_yaw_table + os.sep + 'yaw_misalign_lookup_table_select_range_fine.png')

    print('-- finish plotting yaw table')


def my_err_formatter(x):
    if x >= 1e6:
        s = f'{x:1.1e}'
    else:
        s = f'{x:.0f}'
    return s.rjust(8)


def plot_error_code_counts(
        df_dict,
        turb_keys_to_process,
        path2dir_data_base,
        path2dir_fig_base,
        resample_str,
        date_range_total_str,
        color_dict_by_turb,
):
    n_turbs = len(turb_keys_to_process)

    path2dir_errcode = (
            path2dir_data_base + os.sep + 'error_code' + os.sep + date_range_total_str
            + os.sep + resample_str
    )
    make_dir_if_not_exists(path2dir_errcode)

    path2dir_fig_errcode = (
            path2dir_fig_base + os.sep + 'error_code' + os.sep + date_range_total_str
            + os.sep + resample_str)
    make_dir_if_not_exists(path2dir_fig_errcode)

    path2file_err_code_count = path2dir_errcode + os.sep + 'err_val_cnt.txt'
    
    err_val_cnt = {}
    for turb_n, turb_key in enumerate(turb_keys_to_process):
        df_x = df_dict[turb_key]
        err_val_cnt[turb_key] = df_x['Errorcode'].value_counts()
    
        if turb_n == 0:
            err_val_cnt[turb_key].to_csv(
                path2file_err_code_count,
                sep=',',
                mode='w'
            )
    
        else:
            err_val_cnt[turb_key].to_csv(
                path2file_err_code_count,
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


def plot_counted_intervals(
        int_cnt_dict,
        n_bin_dims,
        bins_per_pair_list,
        var_key_used_for_filter_binning_list,
        ctrl_keys,
        n_turb_pairs,
        turb_keys_up,
        path2dir_fig,
        figsize_count_plot
):

    # init
    figname = 'count_time_interv.png'
    p2d = ''

    # get max interval count over all turbs and control modes
    cnt_max = 0
    for ctrl_n, ctrl_key in enumerate(ctrl_keys):

        for pair_n in range(n_turb_pairs):

            turb_key_up = turb_keys_up[pair_n]

            cnt_arr = int_cnt_dict[ctrl_key][turb_key_up]
            cnt_max_ = cnt_arr.max()
            if cnt_max_ > cnt_max: cnt_max = cnt_max_

    fig, axes = plt.subplots(nrows=2, ncols=n_turb_pairs, figsize=figsize_count_plot)

    # --- 1D binning
    if n_bin_dims == 1:

        bins_per_pair = bins_per_pair_list[0]
        var_key_used_for_filter_binning = var_key_used_for_filter_binning_list[0]

        for ctrl_n, ctrl_key in enumerate(ctrl_keys):

            for pair_n in range(n_turb_pairs):

                turb_key_up = turb_keys_up[pair_n]

                labels_1 = proc.gen_interval_plot_labels_from_bins(bins_per_pair[pair_n])

                cnt_arr = int_cnt_dict[ctrl_key][turb_key_up]

                n_bins = cnt_arr.shape[0]

                ax = axes[ctrl_n, pair_n]

                ax.imshow(cnt_arr.reshape(1, n_bins), cmap='Purples', vmin=0, vmax=cnt_max, alpha=0.8)

                for (i, j), z in np.ndenumerate(cnt_arr.reshape(1, -1)):
                    ax.text(j, i, f'{z:.0f}', ha='center', va='center', size=10)

                ax.yaxis.set_visible(False)
                ax.set_xticks(np.arange(n_bins), labels_1)
                ax.tick_params(axis='x', labelrotation=90)
                ax.tick_params(axis='x', bottom=False)

        for pair_n in range(n_turb_pairs):
            axes[0, pair_n].set_title(f'Pair {pair_n:.0f}', pad=20, size=10)

        for ctrl_n, ctrl_key in enumerate(ctrl_keys):
            axes[ctrl_n, 0].text(-2, 0, f'Ctrl. {ctrl_key}', ha='center', va='center', size=10)

        x_pos_ = -1.2
        axes[0, 0].text(x_pos_, 2, f'{var_key_used_for_filter_binning}:',
                            ha='center', va='center', size=10)
        axes[1, 0].text(x_pos_, 2, f'{var_key_used_for_filter_binning}:',
                            ha='center', va='center', size=10)

        figname = 'count_time_interv_1D.png'
        p2d = path2dir_fig + os.sep + f'1D_{var_key_used_for_filter_binning}'

    # --- 2D
    elif n_bin_dims == 2:

        bins_per_pair_1 = bins_per_pair_list[0]
        bins_per_pair_2 = bins_per_pair_list[1]

        var_key_used_for_filter_binning_1 = var_key_used_for_filter_binning_list[0]
        var_key_used_for_filter_binning_2 = var_key_used_for_filter_binning_list[1]

        fig, axes = plt.subplots(nrows=2, ncols=n_turb_pairs, figsize=figsize_count_plot)

        for ctrl_n, ctrl_key in enumerate(ctrl_keys):

            for pair_n in range(n_turb_pairs):

                turb_key_up = turb_keys_up[pair_n]

                labels_1 = proc.gen_interval_plot_labels_from_bins(bins_per_pair_1[pair_n])
                labels_2 = proc.gen_interval_plot_labels_from_bins(bins_per_pair_2[pair_n])

                cnt_arr = int_cnt_dict[ctrl_key][turb_key_up]

                n_bins_1 = cnt_arr.shape[0]
                n_bins_2 = cnt_arr.shape[1]

                ax = axes[ctrl_n, pair_n]

                ax.imshow(cnt_arr, cmap='Purples', vmin=0, vmax=cnt_max, alpha=0.8)

                for (i, j), z in np.ndenumerate(cnt_arr):
                    ax.text(j, i, f'{z:.0f}', ha='center', va='center', size=10)

                if ctrl_n == 1:
                    ax.set_xticks(np.arange(n_bins_2), labels_2)
                else:
                    ax.xaxis.set_visible(False)

                if pair_n == 0:
                    ax.set_yticks(np.arange(n_bins_1), labels_1)
                else:
                    ax.yaxis.set_visible(False)

                ax.tick_params(axis='x', labelrotation=90)
                ax.tick_params(axis='x', bottom=False)
                ax.tick_params(axis='y', left=False)

        for pair_n in range(n_turb_pairs):
            axes[0, pair_n].set_title(f'Pair {pair_n:.0f}', pad=20, size=10)

        for ctrl_n, ctrl_key in enumerate(ctrl_keys):
            axes[ctrl_n, 0].text(-8, 0.5 * n_bins_1, f'Ctrl. {ctrl_key}', size=10)

        x_pos_ = -2.8
        axes[0, 0].text(x_pos_, -1, f'{var_key_used_for_filter_binning_1}:',
                        ha='center', va='center', size=10)
        axes[1, 0].text(x_pos_, n_bins_1 + 2, f'{var_key_used_for_filter_binning_2}:',
                        ha='center', va='center', size=10)

        figname = 'count_time_interv_2D.png'
        p2d = (
                path2dir_fig + os.sep
                + f'2D_{var_key_used_for_filter_binning_1}'
                + f'_{var_key_used_for_filter_binning_2}'
        )

    make_dir_if_not_exists(p2d)
    fig.tight_layout()
    fig.savefig(p2d + os.sep + figname)
