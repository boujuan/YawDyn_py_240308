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
import pandas as pd
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

plt.close('all')
print('--- script started:', os.path.basename(__file__), '---')

##
FIG_WINDOW_START_POS = np.array((500, 0))
fig_window_pos = np.copy(FIG_WINDOW_START_POS)

##

# alpha = np.array([
    # 15, 345, 0, 30, 330,
    # 90, 270, 0, 180
# ])
# alpha = np.array([
#     30, 45, 60, 75, 90
# ])

alpha = np.arange(-160, 180)

alpha_rad = ana.deg2rad(alpha)
xa = np.cos(alpha_rad)
ya = np.sin(alpha_rad)

phi = ana.calc_mean_angle_by_opt(alpha)
phi_rad = ana.deg2rad(phi)
xp = np.cos(phi_rad)
yp = np.sin(phi_rad)

figsize = cm2inch(20, 20)
MS = 8

fig, ax = plt.subplots(figsize=figsize)
s = np.linspace(0, 2 * np.pi)
xc = np.cos(s)
yc = np.sin(s)

ax.plot(xc, yc, c='k')
ax.plot(xa, ya, 'o', ms=MS)
ax.plot(xp, yp, 'o', ms=MS, c='tab:orange', fillstyle='none', mew=2)

move_figure_window(fig, fig_window_pos, FIG_WINDOW_START_POS, MPL_BACKEND)

plt.show()
