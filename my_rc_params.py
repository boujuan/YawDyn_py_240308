def copy_dict_items(source_dict, dest_dict):
    for key_ in source_dict:
        dest_dict[key_] = source_dict[key_]

def generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS,
                           RASTER_DPI=None, SCREEN_DPI=None,
                           FONT_FAMILY=None,
                           FONT_SERIF=None,
                           FONT_SANS_SERIF=None,
                           MATHTEXT_FONTSET=None,
                           bool_use_latex=False):
    dict_ = {'lines.linewidth': LW,
             'lines.markersize': MS,
             'axes.titlesize': TTFS,
             'axes.labelsize': LFS,
             'legend.fontsize': LFS - 2,
             'xtick.labelsize': TKFS,
             'ytick.labelsize': TKFS,
             'font.size': LFS,
             # {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
             # center_baseline seems to be def, center is OK
             'xtick.alignment': 'center',
             'ytick.alignment': 'center',
             'text.usetex': bool_use_latex,
             }

    if RASTER_DPI is not None:
             dict_['savefig.dpi'] = RASTER_DPI
    if SCREEN_DPI is not None:
             dict_['figure.dpi'] = SCREEN_DPI
    # can be extended by taking the lists of mpl.rcParams and adding to the first position
    if FONT_FAMILY is not None:
        dict_['font.family'] = [FONT_FAMILY]
    if FONT_SERIF is not None:
        dict_['font.serif'] = [FONT_SERIF]
    if FONT_SANS_SERIF is not None:
        dict_['font.sans-serif'] = [FONT_SANS_SERIF]
    if MATHTEXT_FONTSET is not None:
        dict_['mathtext.fontset'] = MATHTEXT_FONTSET

    return dict_

# DO NOT SET mpl backend here, but set directly in script at top of file
# possible: 'TkAgg', 'Qt5Agg', 'pgf', 'pdf', 'inline'
# MPL_BACKEND = 'Qt5Agg'
# MPL_BACKEND = 'pgf'

# SET figure default properties
LW = 1.0 # linewidth
MS = 2 # markersize
TTFS = 8 # title fontsize
LFS = 10 # axes label fontsize (legend = label-2)
TKFS = 10 # tick label fontsize (usually the numbers on the axes)
RASTER_DPI = 300 # raster resolution for saving the figure as file (e.g. .png)
SCREEN_DPI = 100 # figure resolution for displaying on computer screen
FONT_FAMILY = 'serif'
FONT_SERIF = 'CMU Serif'
FONT_SANS_SERIF = 'CMU Sans Serif'
MATHTEXT_FONTSET = 'cm'
bool_use_latex = False
# bool_use_latex = True
bool_interactive = False

rc_params_dict = generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, RASTER_DPI, SCREEN_DPI,
                                        FONT_FAMILY, FONT_SERIF, FONT_SANS_SERIF,
                                        MATHTEXT_FONTSET, bool_use_latex)
rc_params_dict['interactive'] = bool_interactive

# dicts for special backends
pdf_dict = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Computer Modern Roman',
    'text.latex.preamble': "\n".join([  # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[]{siunitx}",
    ])
}

pgf_dict = {
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.serif': 'Computer Modern Roman',
    "pgf.preamble": "\n".join([  # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[]{siunitx}",
    ])
}

# if MPL_BACKEND == 'pdf':
#     copy_dict_items(pdf_dict, rc_params_dict)
# elif MPL_BACKEND == 'pgf':
#     copy_dict_items(pgf_dict, rc_params_dict)