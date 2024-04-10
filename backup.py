filt_id = 4

if filt_id == 1:
    # wdir_min = 288
    # wdir_max = 340
    wspd_min = 5.0
    wspd_max = 12.0
    power_min = 200
    power_max = 4999
    errorcode_min = 5.5
    errorcode_max = 6.5
    pitch_min = -0.1
    pitch_max = 0.1
    # uninterrupted interval settings
    min_interval_duration_s = 10 * 60
    discard_time_at_beginning_s = 5 * 60
    # max gap between filtered valid time steps to be counted as the same interval
    max_gap_duration_s = 60

# broader range to check Power curves
elif filt_id == 2:
    # wdir_min = 260
    # wdir_max = 360
    wspd_min = 2.5
    wspd_max = 15.5
    power_min = 100
    power_max = 5500
    errorcode_min = 5.5
    errorcode_max = 6.5
    pitch_min = -0.1
    pitch_max = 0.1
    # uninterrupted interval settings
    min_interval_duration_s = 10 * 60
    discard_time_at_beginning_s = 5 * 60
    # max gap between filtered valid time steps to be counted as the same interval
    max_gap_duration_s = 60

# broader range to check Power curves
elif filt_id == 3:
    # wdir_min = 260
    # wdir_max = 405
    wspd_min = 2.5
    wspd_max = 15.5
    power_min = 100
    power_max = 5500
    errorcode_min = 5.5
    errorcode_max = 6.5
    pitch_min = -0.1
    pitch_max = 0.1
    # uninterrupted interval settings
    min_interval_duration_s = 10 * 60
    discard_time_at_beginning_s = 5 * 60
    # max gap between filtered valid time steps to be counted as the same interval
    max_gap_duration_s = 60

if filt_id == 4:
    wdir_min_per_pair = [285, 287]
    wdir_max_per_pair = [342, 342]
    # wdir_min = 286
    # wdir_max = 342
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
    # uninterrupted interval settings
    min_interval_duration_s = 8 * 60
    discard_time_at_beginning_s = 3 * 60
    # max gap between filtered valid time steps to be counted as the same interval
    max_gap_duration_s = 30


## BACKUP

## analyze and plot time intervals, where filtered data is available
bool_plot_filtered_time_intervals = 0

if bool_plot_filtered_time_intervals:
    date_format_str = "%m-%d %H:%M"

    plot_limit_shift_wdir = 5

    print('start plotting filtered time intervals')

    calc_yaw_setpoint_pair_1, calc_yaw_setpoint_pair_2 = \
        proc.gen_yaw_table_interp(
            path2dir_yaw_table,
            fname_yaw_table,
        )

    # --- ctrl on
    grouping_mask_time_intervals_pair_1 = (
            df_filt_ctrl_on_dict['T4'].index.to_series().diff().dt.total_seconds().bfill()
            > max_gap_duration_s
    )
    grouping_mask_time_intervals_pair_2 = (
            df_filt_ctrl_on_dict['T6'].index.to_series().diff().dt.total_seconds().bfill()
            > max_gap_duration_s
    )

    grouping_mask_time_intervals_pair_1 = grouping_mask_time_intervals_pair_1.cumsum()
    grouping_mask_time_intervals_pair_2 = grouping_mask_time_intervals_pair_2.cumsum()

    # assign interval lists to dict
    grouping_mask_time_intervals_split = [grouping_mask_time_intervals_pair_1,
                                          grouping_mask_time_intervals_pair_2]
    #
    interval_dict_ctrl_on = {}
    for pair_n in range(n_turb_pairs):
        for turb_key in turb_keys_split_by_pair[pair_n]:
            interval_dict_ctrl_on[turb_key] = \
                [v for k, v in df_filt_ctrl_on_dict[turb_key].groupby(
                grouping_mask_time_intervals_split[pair_n])]

    # --- ctrl off
    grouping_mask_time_intervals_pair_1 = (
            df_filt_ctrl_off_dict['T4'].index.to_series().diff().dt.total_seconds().bfill()
            > max_gap_duration_s
    )
    grouping_mask_time_intervals_pair_2 = (
            df_filt_ctrl_off_dict['T6'].index.to_series().diff().dt.total_seconds().bfill()
            > max_gap_duration_s
    )

    grouping_mask_time_intervals_pair_1 = grouping_mask_time_intervals_pair_1.cumsum()
    grouping_mask_time_intervals_pair_2 = grouping_mask_time_intervals_pair_2.cumsum()

    # assign interval lists to dict
    grouping_mask_time_intervals_split = [
        grouping_mask_time_intervals_pair_1,
        grouping_mask_time_intervals_pair_2
    ]

    #
    interval_dict_ctrl_off = {}
    for pair_n in range(n_turb_pairs):
        for turb_key in turb_keys_split_by_pair[pair_n]:
            interval_dict_ctrl_off[turb_key] = [
                v for k, v in df_filt_ctrl_off_dict[turb_key].groupby(
                    grouping_mask_time_intervals_split[pair_n])
            ]

    n_intervals_ctrl_on = [0] * 4
    interval_length_ctrl_on = []
    # interval_length_tot_ctrl_on = []

    n_intervals_ctrl_off = [0] * 4
    interval_length_ctrl_off = []

    for n in range(4):
        turb_key = turb_keys_to_process[n]
        n_intervals_ctrl_on[n] = len(interval_dict_ctrl_on[turb_key])
        n_intervals_ctrl_off[n] = len(interval_dict_ctrl_off[turb_key])

        ll_ = []
        for nn in range(len(interval_dict_ctrl_on[turb_key])):
            ll_.append(interval_dict_ctrl_on[turb_key][nn].shape[0])
        interval_length_ctrl_on.append(ll_)

        ll_ = []
        for nn in range(len(interval_dict_ctrl_off[turb_key])):
            ll_.append(interval_dict_ctrl_off[turb_key][nn].shape[0])
        interval_length_ctrl_off.append(ll_)

    # plots

    interval_dict_list = [
        interval_dict_ctrl_on,
        interval_dict_ctrl_off
    ]

    title_add_str_list = ['ctrl_on', 'ctrl_off']
    for ctrl_n, interval_dict in enumerate(interval_dict_list):
        title_add_str = title_add_str_list[ctrl_n]
        path2dir_fig_int = path2dir_fig_int_base + os.sep + title_add_str
        make_dir_if_not_exists(path2dir_fig_int)
        delete_all_files_from_dir(path2dir_fig_int)

        # T4
        for int_n in range(len(interval_dict['T4'])):
            # wdir and yaw vs t
            fig, axes = plt.subplots(1, 1, figsize=cm2inch(8, 8))
            axes = [axes]
            axes[0].plot(
                interval_dict['T4'][int_n].index.to_series(),
                interval_dict['T4'][int_n]['WDir'],
                label='WDir'
            )
            axes[0].plot(
                interval_dict['T4'][int_n].index.to_series(),
                interval_dict['T4'][int_n]['Yaw'],
                label='Yaw'
            )
            axes[0].plot(
                interval_dict['T4'][int_n].index.to_series(),
                calc_yaw_setpoint_pair_1(interval_dict['T4'][int_n]['WDir']),
                '--', c='grey', alpha=0.5,
                label='Yaw_table'
            )

            for ax in axes:
                myFmt = mpl.dates.DateFormatter(date_format_str)
                ax.xaxis.set_major_formatter(myFmt)
                ax.tick_params(axis='x', labelrotation=90)
                ax.set_ylim(wdir_min-plot_limit_shift_wdir, wdir_max+plot_limit_shift_wdir)
                ax.legend()
                ax.set_ylabel('Angle [deg]')

            fig.tight_layout()
            fig.savefig(path2dir_fig_int + os.sep + f'WDir_T4_interval_{int_n:.0f}')
            plt.close()

            # yaw vs wdir
            fig, ax = plt.subplots(1, 1, figsize=cm2inch(8, 8))
            ax.plot(
                interval_dict['T4'][int_n]['WDir'],
                interval_dict['T4'][int_n]['Yaw'],
                'o-',
                label='Meas'
            )
            ax.plot(
                interval_dict['T4'][int_n]['WDir'],
                calc_yaw_setpoint_pair_1(interval_dict['T4'][int_n]['WDir']),
                'o', c='grey', alpha=0.5,
                label='Yaw_table'
            )

            ax.set_xlim(wdir_min-plot_limit_shift_wdir, wdir_max+plot_limit_shift_wdir)
            ax.set_ylim(wdir_min-plot_limit_shift_wdir, wdir_max+plot_limit_shift_wdir)
            # ax.tick_params(axis='x', labelrotation=90)

            ax.set_aspect('equal')
            ax.set_xlabel('WDir [deg]')
            ax.set_ylabel('Yaw [deg]')
            ax.legend()

            fig.tight_layout()
            fig.savefig(path2dir_fig_int + os.sep + f'Yaw_vs_WDir_T4_interval_{int_n:.0f}')
            plt.close()

        # T6
        for int_n in range(len(interval_dict['T6'])):
            # wdir and yaw vs t
            fig, axes = plt.subplots(1, 1, figsize=cm2inch(8, 8))
            axes = [axes]
            axes[0].plot(
                interval_dict['T6'][int_n].index.to_series(),
                interval_dict['T6'][int_n]['WDir'],
                label='WDir'
            )
            axes[0].plot(
                interval_dict['T6'][int_n].index.to_series(),
                interval_dict['T6'][int_n]['Yaw'],
                label='Yaw'
            )
            axes[0].plot(
                interval_dict['T6'][int_n].index.to_series(),
                calc_yaw_setpoint_pair_2(interval_dict['T6'][int_n]['WDir']),
                '--', c='grey', alpha=0.5,
                label='Yaw_table'
            )

            for ax in axes:
                # myFmt = mpl.dates.DateFormatter("%m-%d/%H")
                myFmt = mpl.dates.DateFormatter(date_format_str)
                ax.xaxis.set_major_formatter(myFmt)
                ax.tick_params(axis='x', labelrotation=90)
                ax.set_ylim(wdir_min-plot_limit_shift_wdir, wdir_max+plot_limit_shift_wdir)
                ax.legend()

            fig.tight_layout()
            fig.savefig(path2dir_fig_int + os.sep + f'WDir_T6_interval_{int_n:.0f}')
            plt.close()

            # yaw vs wdir
            fig, ax = plt.subplots(1, 1, figsize=cm2inch(8, 8))
            ax.plot(
                interval_dict['T6'][int_n]['WDir'],
                interval_dict['T6'][int_n]['Yaw'],
                'o-',
                label='Meas'
            )
            ax.plot(
                interval_dict['T6'][int_n]['WDir'],
                calc_yaw_setpoint_pair_2(interval_dict['T6'][int_n]['WDir']),
                'o', c='grey', alpha=0.5,
                label='Yaw_table'
            )

            ax.set_xlim(wdir_min-plot_limit_shift_wdir, wdir_max+plot_limit_shift_wdir)
            ax.set_ylim(wdir_min-plot_limit_shift_wdir, wdir_max+plot_limit_shift_wdir)
            # ax.tick_params(axis='x', labelrotation=90)

            ax.set_aspect('equal')
            ax.set_xlabel('WDir [deg]')
            ax.set_ylabel('Yaw [deg]')
            ax.legend()

            fig.tight_layout()
            fig.savefig(path2dir_fig_int + os.sep + f'Yaw_vs_WDir_T6_interval_{int_n:.0f}')
            plt.close()

    print('finish plotting filtered time intervals')


bool_analyze_data = 0
if bool_analyze_data:

    # IN WORK

    dash_list = ['--', '-', '--', '-']

    cp_min = 0.2
    cp_max = 0.8

    # for plotting:

    cp_min_plot = 0.25
    cp_max_plot = 0.55

    cp_min2 = 0.2
    cp_max2 = 0.55

    cp_min_down = 0.2
    cp_max_down = 0.55
    cp_min_up = 0.3
    cp_max_up = 0.5

    df_filt_ctrl_list = [
        df_filt_ctrl_on_dict,
        df_filt_ctrl_off_dict,
    ]

    # first check: plot P/u**3 vs time -> should be constant to have operation at constant Cp
    air_density = 1.225
    rotor_diam = 122
    z_hub = 90

    rotor_rad = 0.5 * rotor_diam
    rotor_area = np.pi * rotor_rad**2

    # P = Cp * 0.5 * rho * A * u**3 = Cp * c1 * u**3
    # Cp = P / (c1 * u**3)
    c1 = 0.5 * air_density * rotor_area

    make_dir_if_not_exists(path2dir_fig_power)

    for pair_n in range(n_turb_pairs):
        for turb_key in turb_keys_split_by_pair[pair_n]:

            turb_key_up = turb_keys_split_by_pair[pair_n][idx_upstream]

            df_filt_ctrl_on_dict[turb_key]['NormPower'] = (
                1000 * df_filt_ctrl_on_dict[turb_key]['Power']
                / (c1 * (df_filt_ctrl_on_dict[turb_key_up]['WSpeed'])**3)
            )
            df_filt_ctrl_off_dict[turb_key]['NormPower'] = (
                1000 * df_filt_ctrl_off_dict[turb_key]['Power']
                / (c1 * (df_filt_ctrl_off_dict[turb_key_up]['WSpeed'])**3)
            )

    # bool_bin_data_2D = 0
    if bool_bin_data_2D:

        u_bin_min = 5.5
        u_bin_max = 12.5
        u_bin_width = 1.0
        u_bins = np.arange(u_bin_min, u_bin_max + 0.5, u_bin_width)
        u_bin_centers = 0.5 * (u_bins[0:-1] + u_bins[1:])

        # wdir_bin_min = 280
        # wdir_bin_max = 340
        # wdir_min = 288
        # wdir_max = 340

        wdir_bins = np.array([wdir_min - 1.0, 296, 308, 316, 328, wdir_max + 1.0])
        wdir_bin_centers = 0.5 * (wdir_bins[0:-1] + wdir_bins[1:])

        df_dict_ = df_filt_ctrl_on_dict
        # df = df_dict

        turb_key = 'T6'
        turb_key_up = 'T6'
        turb_key_down = 'T5'

        df_up = df_dict_[turb_key_up]
        df_down = df_dict_[turb_key_down]
        df_x = df_dict_[turb_key]

        u = df_up['WSpeed']
        bin_mask_u = pd.cut(u, u_bins)

        wdir = df_down['WDir']
        bin_mask_wdir = pd.cut(wdir, wdir_bins)

        power_norm = df_x['NormPower']
        var_binned_mean = power_norm.groupby([bin_mask_u, bin_mask_wdir], observed=False).mean()
        var_binned_std = power_norm.groupby([bin_mask_u, bin_mask_wdir], observed=False).std()

        if bool_use_std_of_mean:
            var_binned_std = (
                    var_binned_std / np.sqrt(
                power_norm.groupby([bin_mask_u, bin_mask_wdir]).count())
            )

    # plot P vs wspd (power curve)
    # IN WORK
    if bool_plot_P_vs_wspd_general:

        u_bin_min = 2.5
        u_bin_max = 15.5
        # u_bins = np.array([6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
        u_bins = np.arange(u_bin_min, u_bin_max + 0.5, 1.0)
        u_bin_centers = 0.5 * (u_bins[0:-1] + u_bins[1:])

        cp_ideal = 0.4
        power_ideal = proc.power_curve_ideal(u_bins, cp_ideal, air_density, rotor_area)

        # plot big plot with 1 plot per turbine
        figsize_cm = [36, 18]
        fig, axes = plt.subplots(
            nrows=1, ncols=4, sharex=True,
            figsize=cm2inch(figsize_cm),
        )

        str_add_list = ['ctrl_on', 'ctrl_off']

        # cp_min_down = 0.2
        # cp_max_down = 0.8
        # cp_min_up = 0.2
        # cp_max_up = 0.6
        power_min_plot = 0
        power_max_plot = 5200

        VLS = ':'
        VLW = 2
        turb_n = 0
        ALPHA_ERR = 0.1

        for pair_n in range(n_turb_pairs):
            turb_key_up = turb_keys_split_by_pair[pair_n][idx_upstream]
            # turb_key_down = turb_keys_split_by_pair[pair_n][idx_downstream]

            for turb_key in turb_keys_split_by_pair[pair_n]:

                ax = axes[turb_n]

                # if turb_key in ['T3', 'T5']:
                #     ax.vlines(wdir_bins, cp_min_down, cp_max_down,
                #               colors='k', linestyles=VLS, linewidths=VLW,
                #               alpha=0.5, zorder=0)
                # else:
                #     ax.vlines(wdir_bins, cp_min_up, cp_max_up,
                #               colors='k', linestyles=VLS, linewidths=VLW,
                #               alpha=0.5, zorder=0)

                for ctrl_n in range(2):
                    df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
                    str_add = str_add_list[ctrl_n]

                    df_x_up = df_filt_ctrl_dict[turb_key_up]
                    df_x = df_filt_ctrl_dict[turb_key]

                    u = df_x_up['WSpeed']
                    bin_mask_u = pd.cut(u, u_bins)

                    # wdir = df_x_up['WDir']
                    # bin_mask_wdir = pd.cut(wdir, wdir_bins)

                    # power_norm_up = df_x_up['NormPower']
                    # power_norm_down = df_x_down['NormPower']
                    # power_norm = 0.5 * (power_norm_up + power_norm_down)
                    power = df_x['Power']

                    var_binned_mean = power.groupby(bin_mask_u, observed=False).mean()
                    var_binned_std = power.groupby(bin_mask_u, observed=False).std()

                    if bool_use_std_of_mean:
                        var_binned_std = (
                                var_binned_std / np.sqrt(power.groupby(bin_mask_u,
                                                                       observed=False).count())
                        )

                    # print('turb_key, var_binned_mean.index')
                    # print(turb_key, var_binned_mean.index)
                    # print('turb_key, var_binned_std.index')
                    # print(turb_key, var_binned_std.index)

                    # var_binned_mean_vs_wdir_per_wspd_list = []
                    # var_binned_std_vs_wdir_per_wspd_list = []
                    # for u_bin_n in range(len(u_bin_centers)):
                    #     int_x = var_binned_mean.index.get_level_values(0).unique()[u_bin_n]
                    #     var_binned_mean_vs_wdir = var_binned_mean.loc[[int_x]]
                    #     var_binned_mean_vs_wdir_per_wspd_list.append(var_binned_mean_vs_wdir)
                    #
                    #     int_x = var_binned_std.index.get_level_values(0).unique()[u_bin_n]
                    #     var_binned_std_vs_wdir = var_binned_std.loc[[int_x]]
                    #     var_binned_std_vs_wdir_per_wspd_list.append(var_binned_std_vs_wdir)

                    # plot Cp vs wdir for several wspd regimes
                    # plot Cp vs wdir
                    # ax = axes[0]
                    start_idx = 0
                    m_cnt = 0

                    # for u_bin_n in range(len(u_bin_centers)):

                        # var_binned_mean_vs_wdir = var_binned_mean_vs_wdir_per_wspd_list[u_bin_n]
                        # var_binned_std_vs_wdir = var_binned_std_vs_wdir_per_wspd_list[u_bin_n]

                    MS = 6
                    COLOR = config.color_list[ctrl_n]
                    ax.errorbar(
                        u_bin_centers[start_idx:],
                        # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                        var_binned_mean.to_numpy(),
                        var_binned_std.to_numpy(),
                        # fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                        fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                        color=COLOR,
                        ms=MS,
                        capsize=0,
                        elinewidth=0.0,
                        fillstyle='none', mew=0.4,
                        label=f'{str_add}'
                    )

                    ax.fill_between(
                        u_bin_centers[start_idx:],
                        (var_binned_mean.to_numpy()[start_idx:]
                         - var_binned_std.to_numpy()[start_idx:]),
                        (var_binned_mean.to_numpy()[start_idx:]
                         + var_binned_std.to_numpy()[start_idx:]),
                        color=COLOR,
                        alpha=ALPHA_ERR,
                    )

                    # m_cnt += 1

                ax.plot(
                    u_bins,
                    power_ideal,
                    '-',
                    color='k',
                    label='ideal'
                )


                ax.set_xlim([u_bins[0], u_bins[-1]])

                # if turb_key in ['T3', 'T5']:
                ax.set_ylim([power_min_plot, power_max_plot])
                # else:
                #     ax.set_ylim([cp_min_up, cp_max_up])

                # ax.tick_params(axis='x', labelrotation=90)
                ax.set_xlabel(r'WSpeed [m/s]')
                if turb_n == 0:
                    ax.set_ylabel(r'Power [kW]')
                ax.legend(ncols=1, loc='upper left')
                ax.grid()
                ax.set_title(f'{turb_key}')

                turb_n += 1

        # save fig
        fig.tight_layout()

        # resample_time_P = '1s'
        figname = 'Power_binned_vs_wspd_turbs'
        if bool_save_fig:
            fig.savefig(path2dir_fig_power + os.sep + figname + '.png',
                        bbox_inches='tight')
            plt.close('all')

    # plot Cp vs wspd (normalized power curve)
    # IN WORK
    if bool_plot_Cp_vs_wspd_general:

        u_bin_min = 2.5
        u_bin_max = 15.5
        # u_bins = np.array([6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
        u_bins = np.arange(u_bin_min, u_bin_max + 0.5, 1.0)
        u_bin_centers = 0.5 * (u_bins[0:-1] + u_bins[1:])

        # plot big plot with 1 plot per turbine
        figsize_cm = [36, 18]
        fig, axes = plt.subplots(
            nrows=1, ncols=4, sharex=True,
            figsize=cm2inch(figsize_cm),
        )

        str_add_list = ['ctrl_on', 'ctrl_off']

        VLS = ':'
        VLW = 2
        turb_n = 0
        ALPHA_ERR = 0.1

        for pair_n in range(n_turb_pairs):
            turb_key_up = turb_keys_split_by_pair[pair_n][idx_upstream]
            # turb_key_down = turb_keys_split_by_pair[pair_n][idx_downstream]

            for turb_key in turb_keys_split_by_pair[pair_n]:

                ax = axes[turb_n]

                # if turb_key in ['T3', 'T5']:
                #     ax.vlines(wdir_bins, cp_min_down, cp_max_down,
                #               colors='k', linestyles=VLS, linewidths=VLW,
                #               alpha=0.5, zorder=0)
                # else:
                #     ax.vlines(wdir_bins, cp_min_up, cp_max_up,
                #               colors='k', linestyles=VLS, linewidths=VLW,
                #               alpha=0.5, zorder=0)

                for ctrl_n in range(2):
                    df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
                    str_add = str_add_list[ctrl_n]

                    df_x_up = df_filt_ctrl_dict[turb_key_up]
                    df_x = df_filt_ctrl_dict[turb_key]

                    u = df_x_up['WSpeed']
                    bin_mask_u = pd.cut(u, u_bins)

                    # wdir = df_x_up['WDir']
                    # bin_mask_wdir = pd.cut(wdir, wdir_bins)

                    # power_norm_up = df_x_up['NormPower']
                    # power_norm_down = df_x_down['NormPower']
                    # power_norm = 0.5 * (power_norm_up + power_norm_down)
                    power_norm = df_x['NormPower']

                    var_binned_mean = power_norm.groupby(bin_mask_u, observed=False).mean()
                    var_binned_std = power_norm.groupby(bin_mask_u, observed=False).std()

                    if bool_use_std_of_mean:
                        var_binned_std = (
                                var_binned_std / np.sqrt(
                            power_norm.groupby(bin_mask_u, observed=False).count())
                        )

                    # print('turb_key, var_binned_mean.index')
                    # print(turb_key, var_binned_mean.index)
                    # print('turb_key, var_binned_std.index')
                    # print(turb_key, var_binned_std.index)

                    # var_binned_mean_vs_wdir_per_wspd_list = []
                    # var_binned_std_vs_wdir_per_wspd_list = []
                    # for u_bin_n in range(len(u_bin_centers)):
                    #     int_x = var_binned_mean.index.get_level_values(0).unique()[u_bin_n]
                    #     var_binned_mean_vs_wdir = var_binned_mean.loc[[int_x]]
                    #     var_binned_mean_vs_wdir_per_wspd_list.append(var_binned_mean_vs_wdir)
                    #
                    #     int_x = var_binned_std.index.get_level_values(0).unique()[u_bin_n]
                    #     var_binned_std_vs_wdir = var_binned_std.loc[[int_x]]
                    #     var_binned_std_vs_wdir_per_wspd_list.append(var_binned_std_vs_wdir)

                    # plot Cp vs wdir for several wspd regimes
                    # plot Cp vs wdir
                    # ax = axes[0]
                    start_idx = 0
                    m_cnt = 0

                    # for u_bin_n in range(len(u_bin_centers)):

                        # var_binned_mean_vs_wdir = var_binned_mean_vs_wdir_per_wspd_list[u_bin_n]
                        # var_binned_std_vs_wdir = var_binned_std_vs_wdir_per_wspd_list[u_bin_n]

                    MS = 6
                    COLOR = config.color_list[ctrl_n]
                    ax.errorbar(
                        u_bin_centers[start_idx:],
                        # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                        var_binned_mean.to_numpy(),
                        var_binned_std.to_numpy(),
                        # fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                        fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                        color=COLOR,
                        ms=MS,
                        capsize=0,
                        elinewidth=0.0,
                        fillstyle='none', mew=0.4,
                        label=f'{str_add}'
                    )

                    ax.fill_between(
                        u_bin_centers[start_idx:],
                        (var_binned_mean.to_numpy()[start_idx:]
                         - var_binned_std.to_numpy()[start_idx:]),
                        (var_binned_mean.to_numpy()[start_idx:]
                         + var_binned_std.to_numpy()[start_idx:]),
                        color=COLOR,
                        alpha=ALPHA_ERR,
                    )

                    m_cnt += 1

                ax.set_xlim([u_bins[0], u_bins[-1]])

                # if turb_key in ['T3', 'T5']:
                ax.set_ylim([cp_min_plot, cp_max_plot])
                # else:
                #     ax.set_ylim([cp_min_up, cp_max_up])

                # ax.tick_params(axis='x', labelrotation=90)
                ax.set_xlabel(r'WSpeed [m/s]')
                if turb_n == 0:
                    ax.set_ylabel(r'Cp [-]')
                ax.legend(ncols=1, loc='upper left')
                ax.grid()
                ax.set_title(f'{turb_key}')

                turb_n += 1

        # save fig
        fig.tight_layout()

        # resample_time_P = '1s'
        figname = 'Cp_binned_vs_wspd_turbs'
        if bool_save_fig:
            fig.savefig(path2dir_fig_power + os.sep + figname + '.png',
                        bbox_inches='tight')
            plt.close('all')

    # plot power time series
    if bool_plot_P_norm_vs_t:

        title_add_str_list = ['ctrl_on', 'ctrl_off']

        resample_time_P = '1s'
        bool_resample_P = 0

        if resample_time_P == '1s':
            bool_resample_P = 0

        for ctrl_n in range(2):
            df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
            title_add_str = title_add_str_list[ctrl_n]

            figsize_cm = [96, 8]
            fig, ax = plt.subplots(figsize=cm2inch(figsize_cm))

            for turb_n, turb_key in enumerate(turb_keys_to_process):
                power_norm = df_filt_ctrl_dict[turb_key]['NormPower']
                time_ax_values = df_filt_ctrl_dict[turb_key].index.to_series()

                if bool_resample_P:
                    power_norm = power_norm.resample(resample_time_P).mean()
                    mask_ = ~power_norm.isna()
                    power_norm = power_norm[mask_]

                    time_ax_values = time_ax_values.resample(resample_time_P).last()
                    time_ax_values = np.arange(time_ax_values[mask_].shape[0])

                else:
                    time_ax_values = np.arange(time_ax_values.shape[0])

                # power_norm = df_filt_ctrl_dict[turb_key]['NormPower'].rolling(
                #     window=60, center=True).mean()
                ax.plot(
                    time_ax_values,
                    # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                    power_norm,
                    marker_list[turb_n],
                    fillstyle='none', mew=0.4,
                    label=turb_key
                )

                ax.tick_params(axis='x', labelrotation=90)
                ax.legend()
                # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
                ax.set_ylabel(r'$C_P$')
                ax.set_title(title_add_str)

            fig.tight_layout()
            figname = 'PowerNorm_' + title_add_str + '_' + resample_time_P
            if bool_save_fig:
                fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
                plt.close('all')

    # plot P norm versus u_upstream
    if bool_plot_P_norm_vs_u:

        # u_bins = np.array([5, 6, 7, 8, 9, 10, 11, 12])

        u_bin_min = 5.5
        u_bin_max = 12.5
        # u_bins = np.array([6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
        u_bins = np.arange(u_bin_min, u_bin_max+0.5, 1.0)

        title_add_str_list = ['ctrl_on', 'ctrl_off']

        resample_time_P = '1s'
        # bool_resample_P = 0

        # if resample_time_P == '1s':
        #     bool_resample_P = 0

        for ctrl_n in range(2):
            df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
            title_add_str = title_add_str_list[ctrl_n]

            figsize_cm = [8, 8]
            fig, ax = plt.subplots(figsize=cm2inch(figsize_cm))
            for pair_n in range(n_turb_pairs):
                turb_key_upstream = turb_keys_split_by_pair[pair_n][1]
                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                    u = df_filt_ctrl_dict[turb_key_upstream]['WSpeed']
                    power_norm = df_filt_ctrl_dict[turb_key]['NormPower']
                    # time_ax_values = df_filt_ctrl_dict[turb_key].index.to_series()

                    # if bool_resample_P:
                    #     power_norm = power_norm.resample(resample_time_P).mean()
                    #     mask_ = ~power_norm.isna()
                    #     power_norm = power_norm[mask_]
                    #
                    #     time_ax_values = time_ax_values.resample(resample_time_P).last()
                    #     time_ax_values = np.arange(time_ax_values[mask_].shape[0])
                    #
                    # else:
                    #     time_ax_values = np.arange(time_ax_values.shape[0])

                    # power_norm = df_filt_ctrl_dict[turb_key]['NormPower'].rolling(
                    #     window=60, center=True).mean()
                    ax.plot(
                        u,
                        # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                        power_norm,
                        marker_list[turb_n],
                        fillstyle='none', mew=0.4,
                        label=turb_key
                    )

            ax.set_xlim([u_bin_min, u_bin_max])
            # ax.set_ylim([cp_min, cp_max])
            ax.set_ylim([cp_min_plot, cp_max_plot])

            # ax.tick_params(axis='x', labelrotation=90)
            ax.legend()
            ax.grid()
            # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
            ax.set_ylabel(r'$C_P$')
            ax.set_xlabel(r'$u$ [m/s]')
            ax.set_title(title_add_str)

            fig.tight_layout()
            figname = 'PowerNorm_vs_u_' + title_add_str + '_' + resample_time_P
            if bool_save_fig:
                fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
                plt.close('all')

        # binning
        for ctrl_n in range(2):
            df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
            title_add_str = title_add_str_list[ctrl_n]

            figsize_cm = [8, 8]
            fig, ax = plt.subplots(figsize=cm2inch(figsize_cm))
            m_cnt = 0
            for pair_n in range(n_turb_pairs):
                turb_key_upstream = turb_keys_split_by_pair[pair_n][1]
                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):

                    u = df_filt_ctrl_dict[turb_key_upstream]['WSpeed']
                    power_norm = df_filt_ctrl_dict[turb_key]['NormPower']

                    bin_mask = pd.cut(u, u_bins)

                    var_binned_mean = power_norm.groupby(bin_mask, observed=False).mean()
                    var_binned_std = power_norm.groupby(bin_mask, observed=False).std()

                    if bool_use_std_of_mean:
                        var_binned_std = (
                                var_binned_std / np.sqrt(
                            power_norm.groupby(bin_mask, observed=False).count())
                        )

                    u_bins_center = u_bins[0:-1] + 0.5

                    start_idx = 0

                    ax.errorbar(
                        u_bins_center[start_idx:],
                        # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                        var_binned_mean.to_numpy()[start_idx:],
                        var_binned_std.to_numpy()[start_idx:],
                        fmt=dash_list[m_cnt] + marker_list[m_cnt],
                        capsize=5,
                        elinewidth=0.5,
                        fillstyle='none', mew=0.4,
                        label=turb_key
                    )

                    m_cnt += 1

            ax.set_xlim([u_bin_min, u_bin_max])
            # ax.set_ylim([cp_min, cp_max])
            ax.set_ylim([cp_min_plot, cp_max_plot])

            # ax.tick_params(axis='x', labelrotation=90)
            ax.legend()
            ax.grid()
            # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
            ax.set_ylabel(r'$C_P$')
            ax.set_xlabel(r'$u$ [m/s]')
            ax.set_title(title_add_str)

            fig.tight_layout()

            figname = 'PowerNorm_binned_vs_u_' + title_add_str + '_' + resample_time_P
            if bool_save_fig:
                fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
                plt.close('all')

        # plot each pair in a plot with binning
        title_add_str_list2 = ['pair 1', 'pair 2']
        MS = 5
        color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        # config.color_list

        figsize_cm = [16, 16]
        fig, axes = plt.subplots(ncols=2, figsize=cm2inch(figsize_cm))

        for pair_n in range(n_turb_pairs):

            ax = axes[pair_n]

            title_add_str = title_add_str_list2[pair_n]

            turb_key_upstream = turb_keys_split_by_pair[pair_n][idx_upstream]
            turb_key_downstream = turb_keys_split_by_pair[pair_n][idx_downstream]

            m_cnt = 0

            # u_bins = np.array([5, 6, 7, 8, 9, 10, 11, 12])

            for ctrl_n in range(2):

                df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
                ctrl_str = title_add_str_list[ctrl_n]

                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):

                    u = df_filt_ctrl_dict[turb_key_upstream]['WSpeed']
                    power_norm = df_filt_ctrl_dict[turb_key]['NormPower']

                    bin_mask = pd.cut(u, u_bins)

                    var_binned_mean = power_norm.groupby(bin_mask, observed=False).mean()
                    var_binned_std = power_norm.groupby(bin_mask, observed=False).std()

                    if bool_use_std_of_mean:
                        var_binned_std = (
                                var_binned_std / np.sqrt(
                            power_norm.groupby(bin_mask, observed=False).count())
                        )

                    u_bins_center = u_bins[0:-1] + 0.5

                    start_idx = 0

                    COLOR = color_list[m_cnt]

                    ax.errorbar(
                        u_bins_center[start_idx:],
                        # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                        var_binned_mean.to_numpy()[start_idx:],
                        var_binned_std.to_numpy()[start_idx:],
                        fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                        color=COLOR,
                        ms=MS,
                        capsize=0,
                        elinewidth=0.0,
                        fillstyle='none', mew=0.4,
                        label=(turb_key + '_' + ctrl_str)
                    )

                    ax.fill_between(
                        u_bins_center[start_idx:],
                        (var_binned_mean.to_numpy()[start_idx:]
                         - var_binned_std.to_numpy()[start_idx:]),
                        (var_binned_mean.to_numpy()[start_idx:]
                         + var_binned_std.to_numpy()[start_idx:]),
                        color=COLOR,
                        alpha=0.2,
                    )

                    m_cnt += 1

            ax.set_xlim([u_bin_min, u_bin_max])
            # ax.set_ylim([cp_min, cp_max])
            ax.set_ylim([cp_min_plot, cp_max_plot])

            # ax.tick_params(axis='x', labelrotation=90)
            ax.legend()
            ax.grid()
            # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
            ax.set_ylabel(r'$C_P$')
            ax.set_xlabel(r'$u$ [m/s]')
            ax.set_title(title_add_str)

        fig.tight_layout()

        figname = 'PowerNorm_binned_vs_u_' + resample_time_P
        if bool_save_fig:
            fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
            plt.close('all')

        # --- plot each pair total Cp in a plot with binning
        title_add_str_list2 = ['pair 1', 'pair 2']
        MS = 5
        color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        figsize_cm = [16, 16]
        fig, axes = plt.subplots(ncols=2, figsize=cm2inch(figsize_cm))

        u_bins_center = 0.5 * ( u_bins[1:] + u_bins[0:-1] )

        for pair_n in range(n_turb_pairs):

            ax = axes[pair_n]

            title_add_str = title_add_str_list2[pair_n]

            turb_key_upstream = turb_keys_split_by_pair[pair_n][idx_upstream]
            turb_key_downstream = turb_keys_split_by_pair[pair_n][idx_downstream]

            m_cnt = 0

            # u_bins = np.array([5, 6, 7, 8, 9, 10, 11, 12])

            for ctrl_n in range(2):

                df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
                ctrl_str = title_add_str_list[ctrl_n]

                # for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):

                u = df_filt_ctrl_dict[turb_key_upstream]['WSpeed']
                bin_mask = pd.cut(u, u_bins)

                power_norm_up = df_filt_ctrl_dict[turb_key_upstream]['NormPower']
                power_norm_down = df_filt_ctrl_dict[turb_key_downstream]['NormPower']

                power_norm = 0.5 * (power_norm_up + power_norm_down)

                var_binned_mean = power_norm.groupby(bin_mask, observed=False).mean()
                var_binned_std = power_norm.groupby(bin_mask, observed=False).std()

                if bool_use_std_of_mean:
                    var_binned_std = (
                            var_binned_std / np.sqrt(power_norm.groupby(
                        bin_mask, observed=False).count())
                    )



                start_idx = 0

                COLOR = color_list[m_cnt]

                ax.errorbar(
                    u_bins_center[start_idx:],
                    # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                    var_binned_mean.to_numpy()[start_idx:],
                    var_binned_std.to_numpy()[start_idx:],
                    fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                    color=COLOR,
                    ms=MS,
                    capsize=0,
                    elinewidth=0.0,
                    fillstyle='none', mew=0.4,
                    label=ctrl_str
                )

                ax.fill_between(
                    u_bins_center[start_idx:],
                    var_binned_mean.to_numpy()[start_idx:] - var_binned_std.to_numpy()[start_idx:],
                    var_binned_mean.to_numpy()[start_idx:] + var_binned_std.to_numpy()[start_idx:],
                    color=COLOR,
                    alpha=0.2,
                )

                m_cnt += 1

            ax.set_xlim([u_bin_min, u_bin_max])
            # ax.set_ylim([cp_min, cp_max])
            ax.set_ylim([cp_min_plot, cp_max_plot])

            # ax.tick_params(axis='x', labelrotation=90)
            ax.legend()
            ax.grid()
            # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
            ax.set_ylabel(r'$C_P$')
            ax.set_xlabel(r'$u$ [m/s]')
            ax.set_title(title_add_str)

        fig.tight_layout()

        figname = 'PowerNorm_binned_vs_u_tot_' + resample_time_P
        if bool_save_fig:
            fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
            plt.close('all')

    # --- bin wdir and wspd (upstream)
    # IN WORK
    if bool_plot_Cp_vs_wdir_binned_2D:
        u_bin_min = 5.5
        u_bin_max = 12.5
        # u_bins = np.array([6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
        u_bins = np.arange(u_bin_min, u_bin_max + 0.5, 1.0)
        u_bin_centers = 0.5 * (u_bins[0:-1] + u_bins[1:])

        # wdir_bin_min = 280
        # wdir_bin_max = 340
        # wdir_min = 288
        # wdir_max = 340

        wdir_bins = np.array([wdir_min - 1.0, 296, 308, 316, 328, wdir_max + 1.0])
        wdir_bin_centers = 0.5 * (wdir_bins[0:-1] + wdir_bins[1:])

        # plot separate plots for every turbine for ctrl off and on
        str_add_list = ['ctrl_on', 'ctrl_off']

        for ctrl_n in range(2):

            df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]

            for pair_n in range(n_turb_pairs):
                turb_key_upstream = turb_keys_split_by_pair[pair_n][idx_upstream]

                for turb_key in turb_keys_split_by_pair[pair_n]:

                    df_x = df_filt_ctrl_dict[turb_key]
                    df_x_up = df_filt_ctrl_dict[turb_key_upstream]

                    u = df_x_up['WSpeed']
                    bin_mask_u = pd.cut(u, u_bins)

                    wdir = df_x_up['WDir']
                    bin_mask_wdir = pd.cut(wdir, wdir_bins)

                    power_norm = df_x['NormPower']
                    var_binned_mean = power_norm.groupby([bin_mask_u, bin_mask_wdir],
                                                         observed=False).mean()
                    var_binned_std = power_norm.groupby([bin_mask_u, bin_mask_wdir],
                                                        observed=False).std()

                    if bool_use_std_of_mean:
                        var_binned_std = (
                                var_binned_std / np.sqrt(power_norm.groupby(
                            [bin_mask_u, bin_mask_wdir], observed=False).count())
                        )

                    # print('turb_key, var_binned_mean.index')
                    # print(turb_key, var_binned_mean.index)
                    # print('turb_key, var_binned_std.index')
                    # print(turb_key, var_binned_std.index)

                    # IN WORK
                    var_binned_mean_vs_wdir_per_wspd_list = []
                    var_binned_std_vs_wdir_per_wspd_list = []
                    for u_bin_n in range(len(u_bin_centers)):
                        int_x = var_binned_mean.index.get_level_values(0).unique()[u_bin_n]
                        var_binned_mean_vs_wdir = var_binned_mean.loc[[int_x]]
                        var_binned_mean_vs_wdir_per_wspd_list.append(var_binned_mean_vs_wdir)

                        int_x = var_binned_std.index.get_level_values(0).unique()[u_bin_n]
                        var_binned_std_vs_wdir = var_binned_std.loc[[int_x]]
                        var_binned_std_vs_wdir_per_wspd_list.append(var_binned_std_vs_wdir)

                    # plot Cp vs wdir for several wspd regimes

                    figsize_cm = [16, 22]
                    fig, axes = plt.subplots(
                        nrows=2, sharex=True,
                        figsize=cm2inch(figsize_cm),
                        height_ratios=[3, 1]
                    )

                    # plot Cp vs wdir
                    ax = axes[0]
                    start_idx = 0
                    m_cnt = 0
                    for u_bin_n in range(len(u_bin_centers)):

                        var_binned_mean_vs_wdir = var_binned_mean_vs_wdir_per_wspd_list[u_bin_n]
                        var_binned_std_vs_wdir = var_binned_std_vs_wdir_per_wspd_list[u_bin_n]

                        MS = 6
                        COLOR = config.color_list[m_cnt]
                        ax.errorbar(
                            wdir_bin_centers[start_idx:],
                            # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                            var_binned_mean_vs_wdir.to_numpy(),
                            var_binned_std_vs_wdir.to_numpy(),
                            # fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                            fmt='-' + marker_list[m_cnt],
                            color=COLOR,
                            ms=MS,
                            capsize=0,
                            elinewidth=0.0,
                            fillstyle='none', mew=0.4,
                            label=f'$u$={u_bin_centers[u_bin_n]:.1f} m/s'
                        )

                        ax.fill_between(
                            wdir_bin_centers[start_idx:],
                            (var_binned_mean_vs_wdir.to_numpy()[start_idx:]
                             - var_binned_std_vs_wdir.to_numpy()[start_idx:]),
                            (var_binned_mean_vs_wdir.to_numpy()[start_idx:]
                             + var_binned_std_vs_wdir.to_numpy()[start_idx:]),
                            color=COLOR,
                            alpha=0.2,
                        )

                        m_cnt += 1

                    ax.set_xlim([wdir_bins[0], wdir_bins[-1]])
                    ax.set_ylim([cp_min2, cp_max2])

                    # ax.tick_params(axis='x', labelrotation=90)
                    ax.legend()
                    ax.grid()
                    # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
                    ax.set_ylabel(r'$C_P$')
                    ax.set_xlabel(r'WDir [deg]')
                    ax.set_title(turb_key)

                    # plot yaw_mis from yaw_table
                    ax = axes[1]

                    calc_yaw_setpoint_pair_1, calc_yaw_setpoint_pair_2 = proc.gen_yaw_table_interp(
                        path2dir_yaw_table,
                        fname_yaw_table,
                    )

                    wdir_ = np.linspace(wdir_bins[0], wdir_bins[-1], 1000)
                    ax.plot(wdir_, calc_yaw_setpoint_pair_1(wdir_) - wdir_, label='T4')
                    ax.plot(wdir_, calc_yaw_setpoint_pair_2(wdir_) - wdir_, label='T6')
                    # ax.tick_params(axis='x', labelrotation=90)
                    ax.legend()
                    ax.grid()
                    # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
                    ax.set_xlabel(r'WDir [deg]')
                    ax.set_ylabel(r'Yaw misal. table [deg]')
                    # ax.set_title(title_add_str)

                    # save fig
                    fig.tight_layout()

                    # resample_time_P = '1s'
                    figname = 'Cp_binned_vs_wdir_per_wspd_' + str_add_list[ctrl_n] + '_' + turb_key
                    if bool_save_fig:
                        fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
                        plt.close('all')

        # plot big plot with 1 plot per turbine
        figsize_cm = [36, 20]
        fig, axes = plt.subplots(
            nrows=2, ncols=4, sharex=True,
            figsize=cm2inch(figsize_cm),
            height_ratios=[4, 1]
        )

        str_add_list = ['ctrl_on', 'ctrl_off']

        VLS = ':'
        VLW = 2
        turb_n = 0
        ALPHA_ERR = 0.1

        for pair_n in range(n_turb_pairs):
            turb_key_up = turb_keys_split_by_pair[pair_n][idx_upstream]
            # turb_key_down = turb_keys_split_by_pair[pair_n][idx_downstream]

            for turb_key in turb_keys_split_by_pair[pair_n]:

                ax = axes[0, turb_n]

                if turb_key in ['T3', 'T5']:
                    ax.vlines(wdir_bins, cp_min_down, cp_max_down,
                              colors='k', linestyles=VLS, linewidths=VLW,
                              alpha=0.5, zorder=0)
                else:
                    ax.vlines(wdir_bins, cp_min_up, cp_max_up,
                              colors='k', linestyles=VLS, linewidths=VLW,
                              alpha=0.5, zorder=0)

                for ctrl_n in range(2):
                    df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
                    str_add = str_add_list[ctrl_n]

                    df_x_up = df_filt_ctrl_dict[turb_key_up]
                    df_x = df_filt_ctrl_dict[turb_key]

                    u = df_x_up['WSpeed']
                    bin_mask_u = pd.cut(u, u_bins)

                    wdir = df_x_up['WDir']
                    bin_mask_wdir = pd.cut(wdir, wdir_bins)

                    # power_norm_up = df_x_up['NormPower']
                    # power_norm_down = df_x_down['NormPower']
                    # power_norm = 0.5 * (power_norm_up + power_norm_down)
                    power_norm = df_x['NormPower']

                    var_binned_mean = power_norm.groupby([bin_mask_u, bin_mask_wdir],
                                                         observed=False).mean()
                    var_binned_std = power_norm.groupby([bin_mask_u, bin_mask_wdir],
                                                        observed=False).std()

                    if bool_use_std_of_mean:
                        var_binned_std = (
                                var_binned_std / np.sqrt(power_norm.groupby(
                            [bin_mask_u, bin_mask_wdir], observed=False).count())
                        )

                    # print('turb_key, var_binned_mean.index')
                    # print(turb_key, var_binned_mean.index)
                    # print('turb_key, var_binned_std.index')
                    # print(turb_key, var_binned_std.index)

                    # IN WORK
                    var_binned_mean_vs_wdir_per_wspd_list = []
                    var_binned_std_vs_wdir_per_wspd_list = []
                    for u_bin_n in range(len(u_bin_centers)):
                        int_x = var_binned_mean.index.get_level_values(0).unique()[u_bin_n]
                        var_binned_mean_vs_wdir = var_binned_mean.loc[[int_x]]
                        var_binned_mean_vs_wdir_per_wspd_list.append(var_binned_mean_vs_wdir)

                        int_x = var_binned_std.index.get_level_values(0).unique()[u_bin_n]
                        var_binned_std_vs_wdir = var_binned_std.loc[[int_x]]
                        var_binned_std_vs_wdir_per_wspd_list.append(var_binned_std_vs_wdir)

                    # plot Cp vs wdir for several wspd regimes
                    # plot Cp vs wdir
                    # ax = axes[0]
                    start_idx = 0
                    m_cnt = 0

                    for u_bin_n in range(len(u_bin_centers)):

                        var_binned_mean_vs_wdir = var_binned_mean_vs_wdir_per_wspd_list[u_bin_n]
                        var_binned_std_vs_wdir = var_binned_std_vs_wdir_per_wspd_list[u_bin_n]

                        MS = 6
                        COLOR = config.color_list[m_cnt]
                        ax.errorbar(
                            wdir_bin_centers[start_idx:],
                            # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                            var_binned_mean_vs_wdir.to_numpy(),
                            var_binned_std_vs_wdir.to_numpy(),
                            # fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                            fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                            color=COLOR,
                            ms=MS,
                            capsize=0,
                            elinewidth=0.0,
                            fillstyle='none', mew=0.4,
                            label=f'$u$={u_bin_centers[u_bin_n]:.1f} m/s, {str_add}'
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

                if turb_key in ['T3', 'T5']:
                    ax.set_ylim([cp_min_down, cp_max_down])
                else:
                    ax.set_ylim([cp_min_up, cp_max_up])
                    ax.set_yticks(np.arange(cp_min_up, cp_max_up + 0.01, 0.05))

                # ax.tick_params(axis='x', labelrotation=90)
                # ax.set_xlabel(r'WDir [deg]')
                if turb_n == 0:
                    ax.set_ylabel(r'$C_P$')
                    fig.legend(
                        ncols=len(u_bins_center), loc='upper center',
                        bbox_to_anchor=(0.5, 1.05)
                    )

                ax.set_title(f'{turb_key}')

                # plot yaw_mis from yaw_table
                ax = axes[1, turb_n]

                calc_yaw_setpoint_pairs = proc.gen_yaw_table_interp(
                    path2dir_yaw_table,
                    fname_yaw_table,
                )

                calc_yaw_setpoint_pair = calc_yaw_setpoint_pairs[pair_n]

                wdir_ = np.linspace(wdir_bins[0] - 1, wdir_bins[-1] + 1, 1000)
                ax.plot(wdir_, calc_yaw_setpoint_pair(wdir_) - wdir_, label=turb_key_up)
                # ax.plot(wdir_, calc_yaw_setpoint_pair_2(wdir_) - wdir_, label='T6')
                ax.vlines(wdir_bins, -10, 10, colors='k',
                          linestyles=VLS, linewidths=VLW, alpha=0.5)
                # ax.tick_params(axis='x', labelrotation=90)
                ax.legend()
                ax.grid()
                # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
                ax.set_xlabel(r'WDir [deg]')
                if turb_n == 0:
                    ax.set_ylabel(r'Yaw mis. tab [deg]')
                # ax.set_title(title_add_str)

                turb_n += 1

        # save fig
        fig.tight_layout()

        # resample_time_P = '1s'
        figname = 'Cp_binned_vs_wdir_per_wspd_turbs'
        if bool_save_fig:
            fig.savefig(path2dir_fig_power + os.sep + figname + '.png',
                        bbox_inches='tight')
            plt.close('all')

        # plot big plot with 1 plot per pair
        figsize_cm = [32, 16]
        fig, axes = plt.subplots(
            nrows=2, ncols=2, sharex=True,
            figsize=cm2inch(figsize_cm),
            height_ratios=[4, 1]
        )

        str_add_list = ['ctrl_on', 'ctrl_off']

        cp_min2 = 0.2
        cp_max2 = 0.55
        VLS = ':'
        VLW = 2

        for pair_n in range(n_turb_pairs):
            turb_key_up = turb_keys_split_by_pair[pair_n][idx_upstream]
            turb_key_down = turb_keys_split_by_pair[pair_n][idx_downstream]

            ax = axes[0, pair_n]

            ax.vlines(wdir_bins, cp_min2, cp_max2,
                      colors='k', linestyles=VLS, linewidths=VLW,
                      alpha=0.5, zorder=0)

            for ctrl_n in range(2):
                df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
                str_add = str_add_list[ctrl_n]

                df_x_up = df_filt_ctrl_dict[turb_key_up]
                df_x_down = df_filt_ctrl_dict[turb_key_down]

                u = df_x_up['WSpeed']
                bin_mask_u = pd.cut(u, u_bins)

                wdir = df_x_up['WDir']
                bin_mask_wdir = pd.cut(wdir, wdir_bins)

                power_norm_up = df_x_up['NormPower']
                power_norm_down = df_x_down['NormPower']

                power_norm = 0.5 * (power_norm_up + power_norm_down)

                var_binned_mean = power_norm.groupby([bin_mask_u, bin_mask_wdir],
                                                     observed=False).mean()
                var_binned_std = power_norm.groupby([bin_mask_u, bin_mask_wdir],
                                                    observed=False).std()

                if bool_use_std_of_mean:
                    var_binned_std = (
                            var_binned_std / np.sqrt(power_norm.groupby(
                        [bin_mask_u, bin_mask_wdir], observed=False).count())
                    )

                # print('turb_key, var_binned_mean.index')
                # print(turb_key, var_binned_mean.index)
                # print('turb_key, var_binned_std.index')
                # print(turb_key, var_binned_std.index)

                # IN WORK
                var_binned_mean_vs_wdir_per_wspd_list = []
                var_binned_std_vs_wdir_per_wspd_list = []
                for u_bin_n in range(len(u_bin_centers)):
                    int_x = var_binned_mean.index.get_level_values(0).unique()[u_bin_n]
                    var_binned_mean_vs_wdir = var_binned_mean.loc[[int_x]]
                    var_binned_mean_vs_wdir_per_wspd_list.append(var_binned_mean_vs_wdir)

                    int_x = var_binned_std.index.get_level_values(0).unique()[u_bin_n]
                    var_binned_std_vs_wdir = var_binned_std.loc[[int_x]]
                    var_binned_std_vs_wdir_per_wspd_list.append(var_binned_std_vs_wdir)

                # plot Cp vs wdir for several wspd regimes
                # plot Cp vs wdir
                # ax = axes[0]
                start_idx = 0
                m_cnt = 0

                for u_bin_n in range(len(u_bin_centers)):

                    var_binned_mean_vs_wdir = var_binned_mean_vs_wdir_per_wspd_list[u_bin_n]
                    var_binned_std_vs_wdir = var_binned_std_vs_wdir_per_wspd_list[u_bin_n]

                    MS = 6
                    COLOR = config.color_list[m_cnt]
                    ax.errorbar(
                        wdir_bin_centers[start_idx:],
                        # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                        var_binned_mean_vs_wdir.to_numpy(),
                        var_binned_std_vs_wdir.to_numpy(),
                        # fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                        fmt=dash_list[ctrl_n] + marker_list[m_cnt],
                        color=COLOR,
                        ms=MS,
                        capsize=0,
                        elinewidth=0.0,
                        fillstyle='none', mew=0.4,
                        label=f'$u$={u_bin_centers[u_bin_n]:.1f} m/s, {str_add}'
                    )

                    ax.fill_between(
                        wdir_bin_centers[start_idx:],
                        (var_binned_mean_vs_wdir.to_numpy()[start_idx:]
                         - var_binned_std_vs_wdir.to_numpy()[start_idx:]),
                        (var_binned_mean_vs_wdir.to_numpy()[start_idx:]
                         + var_binned_std_vs_wdir.to_numpy()[start_idx:]),
                        color=COLOR,
                        alpha=0.2,
                    )

                    m_cnt += 1

            ax.set_xlim([wdir_bins[0]-1, wdir_bins[-1]+1])
            ax.set_ylim([cp_min2, cp_max2])

            # ax.tick_params(axis='x', labelrotation=90)
            if pair_n == 0:
                ax.set_ylabel(r'$C_P$')
            if pair_n == 0:
                ax.legend(ncols=6, loc='upper left', bbox_to_anchor=(0.0, 1.2))
            # ax.legend(ncols=2)
            # ax.grid()
            # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
            # ax.set_ylabel(r'$C_P$')
            ax.set_xlabel(r'WDir [deg]')
            ax.set_title(f'pair {pair_n+1:.0f}')

            # plot yaw_mis from yaw_table
            ax = axes[1, pair_n]

            calc_yaw_setpoint_pairs = proc.gen_yaw_table_interp(
                path2dir_yaw_table,
                fname_yaw_table,
            )

            calc_yaw_setpoint_pair = calc_yaw_setpoint_pairs[pair_n]

            wdir_ = np.linspace(wdir_bins[0] - 1, wdir_bins[-1] + 1, 1000)
            ax.plot(wdir_, calc_yaw_setpoint_pair(wdir_) - wdir_, label=turb_key_up)
            # ax.plot(wdir_, calc_yaw_setpoint_pair_2(wdir_) - wdir_, label='T6')
            ax.vlines(wdir_bins, -10, 10, colors='k',
                      linestyles=VLS, linewidths=VLW, alpha=0.5)
            # ax.tick_params(axis='x', labelrotation=90)
            ax.legend()
            ax.grid()
            # ax.set_ylabel(r'$P / u^3_\mathrm{upstream}$')
            ax.set_xlabel(r'WDir [deg]')
            if pair_n == 0:
                ax.set_ylabel(r'Yaw mis. tab [deg]')
            # ax.set_title(title_add_str)

        # save fig
        plt.subplots_adjust(
            left=0.07,
            right=0.97,
            bottom=0.1,
            top=0.87,
            wspace=0.1,
            hspace=0.2
        )
        # fig.tight_layout()

        # resample_time_P = '1s'
        figname = 'Cp_binned_vs_wdir_per_wspd_pairs'
        if bool_save_fig:
            fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
            plt.close('all')

    # plot Power curve
    if bool_plot_power_curve:

        u_bin_min = 5.5
        u_bin_max = 12.5
        # u_bins = np.array([6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
        u_bins = np.arange(u_bin_min, u_bin_max+0.5, 1.0)

        title_add_str_list = ['ctrl_on', 'ctrl_off']

        resample_time_P = '1s'
        bool_resample_P = 0

        if resample_time_P == '1s':
            bool_resample_P = 0

        for ctrl_n in range(2):
            df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
            title_add_str = title_add_str_list[ctrl_n]

            figsize_cm = [8, 8]
            fig, ax = plt.subplots(figsize=cm2inch(figsize_cm))
            for pair_n in range(n_turb_pairs):
                turb_key_upstream = turb_keys_split_by_pair[pair_n][1]
                # print('turb_key_upstream')
                # print(turb_key_upstream)
                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):
                    u = df_filt_ctrl_dict[turb_key_upstream]['WSpeed']
                    power = df_filt_ctrl_dict[turb_key]['Power']
                    # time_ax_values = df_filt_ctrl_dict[turb_key].index.to_series()

                    # if bool_resample_P:
                    #     power_norm = power_norm.resample(resample_time_P).mean()
                    #     mask_ = ~power_norm.isna()
                    #     power_norm = power_norm[mask_]
                    #
                    #     time_ax_values = time_ax_values.resample(resample_time_P).last()
                    #     time_ax_values = np.arange(time_ax_values[mask_].shape[0])
                    #
                    # else:
                    #     time_ax_values = np.arange(time_ax_values.shape[0])

                    # power_norm = df_filt_ctrl_dict[turb_key]['NormPower'].rolling(
                    #     window=60, center=True).mean()
                    ax.plot(
                        u,
                        # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                        power,
                        marker_list[turb_n],
                        fillstyle='none', mew=0.4,
                        label=turb_key
                    )

                    ax.set_ylim([0, 5000])

                    # ax.tick_params(axis='x', labelrotation=90)
                    ax.legend()
                    ax.set_ylabel(r'$P$ [kW]')
                    ax.set_xlabel(r'$u$ [m/s]')
                    ax.set_title(title_add_str)

            fig.tight_layout()
            figname = 'Power_curve_vs_u_up_' + title_add_str + '_' + resample_time_P
            if bool_save_fig:
                fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
                plt.close('all')

        # binning
        for ctrl_n in range(2):
            df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]
            title_add_str = title_add_str_list[ctrl_n]

            figsize_cm = [8, 8]
            fig, ax = plt.subplots(figsize=cm2inch(figsize_cm))
            m_cnt = 0
            for pair_n in range(n_turb_pairs):
                turb_key_upstream = turb_keys_split_by_pair[pair_n][1]
                for turb_n, turb_key in enumerate(turb_keys_split_by_pair[pair_n]):

                    u = df_filt_ctrl_dict[turb_key_upstream]['WSpeed']
                    power = df_filt_ctrl_dict[turb_key]['Power']

                    # u_bins = np.array([5, 6, 7, 8, 9, 10, 11, 12])

                    bin_mask = pd.cut(u, u_bins)

                    var_binned_mean = power.groupby(bin_mask, observed=False).mean()
                    var_binned_std = power.groupby(bin_mask, observed=False).std()

                    if bool_use_std_of_mean:
                        var_binned_std = (
                                var_binned_std / np.sqrt(power.groupby(bin_mask,
                                                                       observed=False).count())
                        )

                    u_bins_center = u_bins[0:-1] + 0.5

                    start_idx = 0

                    ax.errorbar(
                        u_bins_center[start_idx:],
                        # df_filt_ctrl_off_dict[turb_key]['NormPower'],
                        var_binned_mean.to_numpy()[start_idx:],
                        var_binned_std.to_numpy()[start_idx:],
                        fmt='-' + marker_list[m_cnt],
                        capsize=5,
                        elinewidth=0.5,
                        fillstyle='none', mew=0.4,
                        label=turb_key
                    )

                    ax.set_ylim([0, 5000])

                    # ax.tick_params(axis='x', labelrotation=90)
                    ax.legend()
                    ax.set_ylabel(r'$P$ [kW]')
                    ax.set_xlabel(r'$u$ [m/s]')
                    ax.set_title(title_add_str)

                    m_cnt += 1

            fig.tight_layout()

            figname = 'Power_curve_binned_' + title_add_str + '_' + resample_time_P
            if bool_save_fig:
                fig.savefig(path2dir_fig_power + os.sep + figname + '.png')
                plt.close('all')

    # compare norm power for both pairs between control on and off

    power_norm_ctrl_on_turb_mean_list = []
    power_norm_ctrl_on_turb_std_list = []

    power_norm_ctrl_off_turb_mean_list = []
    power_norm_ctrl_off_turb_std_list = []

    power_norm_ctrl_on_pair_mean_list = []
    power_norm_ctrl_on_pair_std_list = []

    power_norm_ctrl_off_pair_mean_list = []
    power_norm_ctrl_off_pair_std_list = []

    for pair_n in range(n_turb_pairs):
        turb_key_up = turb_keys_split_by_pair[pair_n][idx_upstream]
        turb_key_down = turb_keys_split_by_pair[pair_n][idx_downstream]

        for ctrl_n in range(2):
            u_min_mask = 7.5
            u_max_mask = 12.5
            # mask_u_on = df_filt_ctrl_on_dict[turb_key_up]['WSpeed'].between(
            # u_min_mask, u_max_mask)
            # mask_u_off = df_filt_ctrl_off_dict[turb_key_up]['WSpeed'].between(
            # u_min_mask, u_max_mask)

            df_filt_ctrl_dict = df_filt_ctrl_list[ctrl_n]

            mask_u = df_filt_ctrl_dict[turb_key_up]['WSpeed'].between(u_min_mask, u_max_mask)

            # power_norm_turb_ctrl_on = df_filt_ctrl_on_dict[turb_key]['NormPower'][mask_u_on]
            # power_norm_turb_ctrl_off = df_filt_ctrl_off_dict[turb_key]['NormPower'][mask_u_off]

            power_norm_turb_up = df_filt_ctrl_dict[turb_key_up]['NormPower'][mask_u]
            power_norm_turb_down = df_filt_ctrl_dict[turb_key_down]['NormPower'][mask_u]
            power_norm_pair = 0.5 * (power_norm_turb_up + power_norm_turb_down)

            print('ctrl_n, power_norm_pair.shape[0]')
            print(ctrl_n, power_norm_pair.shape[0])

            # power_norm_tot_ctrl_on = df_filt_ctrl_on_dict[turb_key_1]['NormPower'] \
            # \+ df_filt_ctrl_on_dict[turb_key_2]['NormPower']
            # power_norm_tot_ctrl_off = df_filt_ctrl_off_dict[turb_key_1]['NormPower'] \
            # + df_filt_ctrl_off_dict[turb_key_2]['NormPower']

            if ctrl_n == 0:
                power_norm_ctrl_on_turb_mean_list.append(power_norm_turb_down.mean())
                power_norm_ctrl_on_turb_std_list.append(power_norm_turb_down.std())
                power_norm_ctrl_on_turb_mean_list.append(power_norm_turb_up.mean())
                power_norm_ctrl_on_turb_std_list.append(power_norm_turb_up.std())

                power_norm_ctrl_on_pair_mean_list.append(power_norm_pair.mean())
                power_norm_ctrl_on_pair_std_list.append(power_norm_pair.std())

            if ctrl_n == 1:
                power_norm_ctrl_off_turb_mean_list.append(power_norm_turb_down.mean())
                power_norm_ctrl_off_turb_std_list.append(power_norm_turb_down.std())
                power_norm_ctrl_off_turb_mean_list.append(power_norm_turb_up.mean())
                power_norm_ctrl_off_turb_std_list.append(power_norm_turb_up.std())

                power_norm_ctrl_off_pair_mean_list.append(power_norm_pair.mean())
                power_norm_ctrl_off_pair_std_list.append(power_norm_pair.std())

        # power_norm_pair_ctrl_on = \
        #     df_filt_ctrl_on_dict[turb_key_1]['NormPower'][mask_u_on] \
        #     + df_filt_ctrl_on_dict[turb_key_2]['NormPower'][mask_u_on] \


## PLOT DATA


## SCRATCH

# # test resampling
# res_bin_closed = 'left'
# res_bin_label = 'right'
#
# test_np = np.arange(0, 86400)
# dti = pd.date_range('2020-01-01T00:00:00', '2020-01-01T23:59:59', freq='s')
# test_df = pd.DataFrame(test_np, index=dti)
# test_10s = test_df.resample(
#     '10s',
#     closed=res_bin_closed,
#     label=res_bin_label
# ).last()
#
# test_10s_first = test_df.resample(
#     '10s',
#     closed=res_bin_closed,
#     label=res_bin_label
# ).first()

# RESAMPLE DATA
# IN WORK: DOES NOT WORK YET FOR SEVERAL DATA INTERVALS
# if bool_resample_data:
#     for turb_key in turb_keys_to_process:
#         if resample_interval_s > resample_interval_s:
#             # downsample in case apply_interval > load_interval
#             df_list = []
#             for var_key in config.var_keys:
#                 if var_key == 'ControlSwitch' and turb_key not in ['T4', 'T6']:
#                     pass
#                 else:
#                     # need to resample the quantities separately, to apply proper angle averaging
#                     df_list.append(
#                         ana.downsample_dataframe_properly(
#                             df_dict[turb_key][var_key], var_key, resample_str_to_apply
#                         )
#                     )
#
#             df_dict[turb_key] = pd.concat(df_list, axis=1)
#
#     print(f'resampled data from {resample_str_to_load} to {resample_str_to_apply}')

# filter data old
# if bool_filter_data:
#
#     print('start filtering data')
#
#     # filter turbine pair one (T3 and T4)
#     filter_mask_pair_1_date = df_dict['T4'].index >= '2023-06-29 08:00:00'
#
#     filter_mask_pair_1 = df_dict['T4']['WDir'].between(wdir_min, wdir_max, inclusive='both')
#     # could use rolling window for filtering wdir due to high fluctuations (takes long time):
#     # filter_mask_pair_1 = df_dict['T4']['WDir'].rolling(window=10).apply(
#     #     ana.calc_mean_angle_deg).bfill().between(wdir_min, wdir_max, inclusive='both')
#
#     filter_mask_pair_1 = filter_mask_pair_1 & filter_mask_pair_1_date
#     filter_mask_pair_1 = filter_mask_pair_1 \
#                          & df_dict['T4']['Power'].between(power_min, power_max, inclusive='both')
#     filter_mask_pair_1 = filter_mask_pair_1 \
#                          & df_dict['T4']['WSpeed'].between(wspd_min, wspd_max, inclusive='both')
#     filter_mask_pair_1 = filter_mask_pair_1 \
#                          & df_dict['T4']['Errorcode'].between(errorcode_min, errorcode_max,
#                                                               inclusive='both')
#     filter_mask_pair_1 = filter_mask_pair_1 \
#                          & df_dict['T4']['Pitch'].between(pitch_min, pitch_max,
#                                                           inclusive='both')
#
#     filter_mask_pair_1 = filter_mask_pair_1 \
#                          & df_dict['T3']['Power'].between(power_min, power_max, inclusive='both')
#     filter_mask_pair_1 = filter_mask_pair_1 \
#                          & df_dict['T3']['Errorcode'].between(errorcode_min, errorcode_max,
#                                                               inclusive='both')
#     filter_mask_pair_1 = filter_mask_pair_1 \
#                          & df_dict['T3']['Pitch'].between(pitch_min, pitch_max,
#                                                           inclusive='both')
#
#     control_switch_mask_pair_1 = ~df_dict['T4']['ControlSwitch'].ffill().isna() & df_dict['T4']['ControlSwitch'].ffill().astype(bool)
#
#     # active controller on only
#     filter_mask_ctrl_on_pair_1 = filter_mask_pair_1 \
#                                  & control_switch_mask_pair_1
#
#     # active controller off only
#     filter_mask_ctrl_off_pair_1 = filter_mask_pair_1 \
#                                   & ~control_switch_mask_pair_1
#
#     # complete filter mask control on
#     grouping_mask_time_intervals_pair_1 = df_dict['T4'].index.to_series()[filter_mask_ctrl_on_pair_1].diff().dt.total_seconds().bfill() > max_gap_duration_s
#     grouping_mask_time_intervals_pair_1 = grouping_mask_time_intervals_pair_1.cumsum()
#     mask_time_intervals_pair_1 = \
#         df_dict['T4'].index.to_series()[filter_mask_ctrl_on_pair_1].groupby(grouping_mask_time_intervals_pair_1).transform('count') >= min_interval_duration_s
#     grouping_mask_time_intervals_pair_1 = grouping_mask_time_intervals_pair_1[mask_time_intervals_pair_1]
#     mask_1 = grouping_mask_time_intervals_pair_1.groupby(grouping_mask_time_intervals_pair_1).transform(
#         mask_first_n_rows, discard_time_at_beginning_s)
#     # grouping_mask_time_intervals_pair_1 = grouping_mask_time_intervals_pair_1[mask_1]
#     mask_time_intervals_pair_1 = mask_time_intervals_pair_1 & mask_1
#     filter_mask_ctrl_on_pair_1 = filter_mask_ctrl_on_pair_1 & mask_time_intervals_pair_1
#
#     # complete filter mask control off
#     grouping_mask_time_intervals_pair_1 = df_dict['T4'].index.to_series()[filter_mask_ctrl_off_pair_1].diff().dt.total_seconds().bfill() > max_gap_duration_s
#     grouping_mask_time_intervals_pair_1 = grouping_mask_time_intervals_pair_1.cumsum()
#     mask_time_intervals_pair_1 = \
#         df_dict['T4'].index.to_series()[filter_mask_ctrl_off_pair_1].groupby(grouping_mask_time_intervals_pair_1).transform('count') >= min_interval_duration_s
#     grouping_mask_time_intervals_pair_1 = grouping_mask_time_intervals_pair_1[mask_time_intervals_pair_1]
#     mask_1 = grouping_mask_time_intervals_pair_1.groupby(grouping_mask_time_intervals_pair_1).transform(
#         mask_first_n_rows, discard_time_at_beginning_s)
#     # grouping_mask_time_intervals_pair_1 = grouping_mask_time_intervals_pair_1[mask_1]
#     mask_time_intervals_pair_1 = mask_time_intervals_pair_1 & mask_1
#     filter_mask_ctrl_off_pair_1 = filter_mask_ctrl_off_pair_1 & mask_time_intervals_pair_1
#
#     # -------------------------------------------------------------------
#     # filter turbine pair two (T5 and T6)
#
#     # use rolling window for filtering wdir due to high fluctuations:
#     filter_mask_pair_2 = df_dict['T6']['WDir'].between(wdir_min, wdir_max, inclusive='both')
#     # filter_mask_pair_2 = df_dict['T6']['WDir'].rolling(window=10).apply(
#     #     ana.calc_mean_angle_deg).bfill().between(wdir_min, wdir_max, inclusive='both')
#
#     filter_mask_pair_2 = filter_mask_pair_2 \
#                          & df_dict['T6']['Power'].between(power_min, power_max, inclusive='both')
#     filter_mask_pair_2 = filter_mask_pair_2 \
#                          & df_dict['T6']['WSpeed'].between(wspd_min, wspd_max, inclusive='both')
#     filter_mask_pair_2 = filter_mask_pair_2 \
#                          & df_dict['T6']['Errorcode'].between(errorcode_min, errorcode_max,
#                                                               inclusive='both')
#     filter_mask_pair_2 = filter_mask_pair_2 \
#                          & df_dict['T6']['Pitch'].between(pitch_min, pitch_max,
#                                                           inclusive='both')
#
#     filter_mask_pair_2 = filter_mask_pair_2 \
#                          & df_dict['T5']['Power'].between(power_min, power_max, inclusive='both')
#     filter_mask_pair_2 = filter_mask_pair_2 \
#                          & df_dict['T5']['Errorcode'].between(errorcode_min, errorcode_max,
#                                                               inclusive='both')
#     filter_mask_pair_2 = filter_mask_pair_2 \
#                          & df_dict['T5']['Pitch'].between(pitch_min, pitch_max,
#                                                           inclusive='both')
#
#     #--
#     control_switch_mask_pair_2 = ~df_dict['T6']['ControlSwitch'].ffill().isna() & df_dict['T6']['ControlSwitch'].ffill().astype(bool)
#
#     # active controller on only
#     filter_mask_ctrl_on_pair_2 = filter_mask_pair_2 \
#                                  & control_switch_mask_pair_2
#
#     # active controller off only
#     filter_mask_ctrl_off_pair_2 = filter_mask_pair_2 \
#                                   & ~control_switch_mask_pair_2
#
#     # complete filter mask control on
#     grouping_mask_time_intervals_pair_2 = df_dict['T6'].index.to_series()[filter_mask_ctrl_on_pair_2].diff().dt.total_seconds().bfill() > max_gap_duration_s
#     grouping_mask_time_intervals_pair_2 = grouping_mask_time_intervals_pair_2.cumsum()
#     mask_time_intervals_pair_2 = \
#         df_dict['T6'].index.to_series()[filter_mask_ctrl_on_pair_2].groupby(grouping_mask_time_intervals_pair_2).transform('count') >= min_interval_duration_s
#     grouping_mask_time_intervals_pair_2 = grouping_mask_time_intervals_pair_2[mask_time_intervals_pair_2]
#     mask_2 = grouping_mask_time_intervals_pair_2.groupby(grouping_mask_time_intervals_pair_2).transform(
#         mask_first_n_rows, discard_time_at_beginning_s)
#     # grouping_mask_time_intervals_pair_2 = grouping_mask_time_intervals_pair_2[mask_2]
#     mask_time_intervals_pair_2 = mask_time_intervals_pair_2 & mask_2
#     filter_mask_ctrl_on_pair_2 = filter_mask_ctrl_on_pair_2 & mask_time_intervals_pair_2
#
#     # complete filter mask control off
#     grouping_mask_time_intervals_pair_2 = df_dict['T6'].index.to_series()[filter_mask_ctrl_off_pair_2].diff().dt.total_seconds().bfill() > max_gap_duration_s
#     grouping_mask_time_intervals_pair_2 = grouping_mask_time_intervals_pair_2.cumsum()
#     mask_time_intervals_pair_2 = \
#         df_dict['T6'].index.to_series()[filter_mask_ctrl_off_pair_2].groupby(grouping_mask_time_intervals_pair_2).transform('count') >= min_interval_duration_s
#     grouping_mask_time_intervals_pair_2 = grouping_mask_time_intervals_pair_2[mask_time_intervals_pair_2]
#     mask_2 = grouping_mask_time_intervals_pair_2.groupby(grouping_mask_time_intervals_pair_2).transform(
#         mask_first_n_rows, discard_time_at_beginning_s)
#     # grouping_mask_time_intervals_pair_2 = grouping_mask_time_intervals_pair_2[mask_2]
#     mask_time_intervals_pair_2 = mask_time_intervals_pair_2 & mask_2
#     filter_mask_ctrl_off_pair_2 = filter_mask_ctrl_off_pair_2 & mask_time_intervals_pair_2
#
#     # only active controller on
#     # filter_mask_pair_2= filter_mask_pair_2 \
#     #                      & df_dict['T6']['ControlSwitch'].ffill()
#
#     filter_mask_ctrl_on_list = [
#         filter_mask_ctrl_on_pair_1,
#         filter_mask_ctrl_on_pair_2,
#     ]
#
#     filter_mask_ctrl_off_list = [
#         filter_mask_ctrl_off_pair_1,
#         filter_mask_ctrl_off_pair_2,
#     ]
#
#     df_filt_ctrl_on_dict = {}
#     df_filt_ctrl_off_dict = {}
#
#     for pair_n in range(n_turb_pairs):
#         for turb_key in turb_keys_split_by_pair[pair_n]:
#             df_filt_ctrl_on_dict[turb_key] = df_dict[turb_key][filter_mask_ctrl_on_list[pair_n]]
#             df_filt_ctrl_off_dict[turb_key] = df_dict[turb_key][filter_mask_ctrl_off_list[pair_n]]
#
#             if bool_store_filtered_data:
#
#                 fname = 'filtered_ctrl_on_' + resample_str_to_apply + '_' + turb_key
#                 df_filt_ctrl_on_dict[turb_key].to_hdf(
#                     path2dir_filtered + os.sep + fname + '.h5',
#                     'scada', mode='w'
#                 )
#
#                 fname = 'filtered_ctrl_off_' + resample_str_to_apply + '_' + turb_key
#                 df_filt_ctrl_off_dict[turb_key].to_hdf(
#                     path2dir_filtered + os.sep + fname + '.h5',
#                     'scada', mode='w'
#                 )
#
#                 print('saved filtered data to hard drive')
#
#     print('finish filtering data')
#
#     # if bool_store_filtered_data:
#     #     filter_mask_pair_1.to_hdf(path2dir_filtered + os.sep + 'filter_mask_pair_1.h5',
#     #                               'mask', mode='w')
#     #     filter_mask_pair_2.to_hdf(path2dir_filtered + os.sep + 'filter_mask_pair_2.h5',
#     #                               'mask', mode='w')
