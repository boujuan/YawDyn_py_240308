## IMPORTS
import numpy as np
import pandas as pd

## CUSTOM FUNCTIONS
two_pi = 2 * np.pi


def deg2rad(angle):
    return np.pi / 180.0 * angle


def rad2deg(angle):
    return 180.0 / np.pi * angle


def calc_mean_angle(angle_arr):
    return np.arctan2( np.nanmean(np.sin(angle_arr)), np.nanmean(np.cos(angle_arr)) ) % two_pi


def calc_mean_angle_deg(angle_arr):
    return rad2deg( calc_mean_angle( deg2rad(angle_arr) ) )


def downsample_dataframe_properly(df, quant_key, resample_str):

    res_bin_closed = 'left'
    res_bin_label = 'right'

    if resample_str == '10s':

        if quant_key in [
            'Power',
            'PowerRef',
        ]:
            res = df.resample(
                resample_str,
                closed=res_bin_closed,
                label=res_bin_label
            ).mean()

        elif quant_key in [
            'WSpeed',
            'WDir',
        ]:
            res = df.resample(
                resample_str,
                closed=res_bin_closed,
                label=res_bin_label
            ).last()

        elif quant_key in [
            'Yaw',
            'Pitch',
        ]:
            res = df.resample(
                resample_str,
                closed=res_bin_closed,
                label=res_bin_label
            ).apply(calc_mean_angle_by_opt_pd_df)

            if quant_key == 'Pitch':
                res = (res + 180.) % 360. - 180.

        elif quant_key in [
            'Errorcode',
            'ControlSwitch'
        ]:
            df = df.dropna(how='any').astype(int)
            df_rs = df.resample(
                resample_str,
                closed=res_bin_closed,
                label=res_bin_label
            )
            mask_bad_rs = df_rs.nunique() > 1
            res = df_rs.last()
            res[mask_bad_rs] = -1

    else:
        if quant_key in [
            'Power',
            'WSpeed',
            'PowerRef',
            'Errorcode',
            'ControlSwitch'
        ]:
            res = df.resample(resample_str).mean()

        elif quant_key in [
            'WDir',
            'Yaw',
            'Pitch',
        ]:
            res = df.resample(resample_str).apply(calc_mean_angle_deg)

    return res


def calc_mean_angle_by_opt(angle):
    """
    for angle as numpy array
    :param angle:
    :return:
    """
    nx = angle.shape[0]
    phi0 = angle.mean()
    phi = (phi0 + 360. / nx * np.arange(nx)) % 360.

    err = (((angle - phi.reshape((-1, 1)) + 180.) % 360. - 180.)**2).sum(axis=1)

    idx_opt = np.argmin(err)

    phi_opt = phi[idx_opt]

    return phi_opt


def calc_mean_angle_by_opt_pd_df(angle):
    """
    for angle as pandas dataframe
    :param angle:
    :return:
    """
    nx = angle.shape[0]
    phi0 = angle.mean()
    phi = (phi0 + 360. / nx * np.arange(nx)) % 360.

    err = (((angle.values - phi.reshape((-1, 1)) + 180.) % 360. - 180.)**2).sum(axis=1)

    idx_opt = np.argmin(err)

    phi_opt = phi[idx_opt]

    return phi_opt

