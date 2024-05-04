"""
Documentation
HOW TO ADD FONTS TO MATPLOTLIB (NOT VIA LATEX, BUT USABLE FONTS WITH INTERNAL MPL RENDERER)
matplotlib = mpl

1. Install fonts (here e.g. in Windows)
- Download .ttf files you want to install
- Select all .ttf files to install and rightclick -> more options -> install for all users
- fonts can be found in Windows, sorted by family name under
C:\\Windows\\Fonts for all users or
C:\\Users\\USER\\AppData\Local\Microsoft\Windows\Fonts for local USER
it is recommended to install for all users
- check in the folder, if fonts were installed properly

2. Get mpl cache dir via python
mpl.get_cachedir()

3. delete all files in matplotlib cache directory
here lies the fontlist-.json file, wich defines font paths, properties and families for mpl
e.g. fontlist-v330.json

4. restart IPython console, if used

NOTES:
To find the matplotlib configuration file (to change global configuration settings)
mpl.matplotlib_fname()
[...]matplotlibrc

A font associated with a .ttf file contains several attributes.
These are for example listed in the mpl fontlist json file
Example:
      "fname": "C:\\Windows\\Fonts\\cmunrm.ttf",
      "name": "CMU Serif",
      "style": "normal",
      "variant": "normal",
      "weight": 500,
      "stretch": "normal",
      "size": "scalable",
      "__class__": "FontEntry"

Example 2:
      "fname": "C:\\Users\\bohre\\AppData\\Local\\Microsoft\\Windows\\Fonts\\cmunbi.ttf",
      "name": "CMU Serif",
      "style": "italic",
      "variant": "normal",
      "weight": 700,
      "stretch": "expanded",
      "size": "scalable",
      "__class__": "FontEntry"

The name defines the Family by which the font can be called in mpl

All fonts available on the system can be found via
mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

Fonts usable by mpl can be found via (incl path and properties)
mpl.font_manager.fontManager.ttflist
this basically displays what is contained in the fontlist.json file in mpl cache directory

Further commands:
mpl.font_manager.get_font_names()
displays (family) names of the font list only

There is also this snippet (not exactly sure what fontconfig_fonts() provides)
# import matplotlib.font_manager
# flist = matplotlib.font_manager.get_fontconfig_fonts()
# names = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]
# print (names)


----------------------------------------------------
get the rcparams file:
mpl.matplotlib_fname()

About backends and windows
mpl can be run with different backends
When importing mpl, a backend is set
By default, this list is checked one by one until a backend works for the system
["macosx", "qt5agg", "qt4agg", "gtk3agg", "tkagg", "wxagg"]

# mpl backend to use, possible values:
# interactive: MacOSX, Qt5Agg, Qt4Agg, GTK4Agg, Gtk3Agg, TkAgg, WxAgg
# non-interactive: pdf, pgf, ps, svg, Cairo, Agg
# special: inline
# NOTES: the default order by mpl is: MacOSX, QtAgg, GTK4Agg, Gtk3Agg, TkAgg, WxAgg, Agg
# 'TkAgg' is reliable under Windows
# 'MacOSX' is default under Mac
# 'QtAgg' windows are freezing when using pycharm and IPython console...
# 'GTK4Agg' sounds like Linux-stuff (?)
# 'inline' can be used for IPython or jupyter.
# in this case, %matplotlib inline will be called and MPL_BACKEND updated
# 'pdf' and 'pgf' to generate .pdf files. 'pgf' seems to do better job when using latex

On Windows, the backend is TkAgg for my applications so far
in this case:
fig = plt.figure() will generate a tkinter object (canvas)
i.e. the figure object has a canvas object with several attributes
fig.canvas    -> the figure canvas
fig.canvas.manager    -> the figure canvas manager
...
The window on the screen is managed by the window object:
fig.canvas.manager.window    -> e.g. tkinter.Tk object

Several useful commands for the window object (tkinter object!!)

Specify window location (WidthxHeight+x+y):
fig.canvas.manager.window.geometry('600x450+600+0')
For several monitors:
Windows creates one virtual monitor adding all pixels
You might need to select negatvie position, if main monitor is not the outer left one, e.g.
fig.canvas.manager.window.geometry('600x450+-800+0')

for other OS, the screen (monitor) number is specified here
fig.canvas.manager.window.winfo_screen()

bring to front AND keep it there (e.g. after moving to new screen):
fig.canvas.manager.window.attributes('-topmost', 1)
stop the forcing to keep it in front:
fig.canvas.manager.window.attributes('-topmost', 0)
or use this stop the forcing to keep it in front (maybe softer on the system??)
fig.canvas.manager.window.after_idle(fig.canvas.manager.window.attributes, '-topmost', False)

There is also fig.canvas.manager.window.lift(), but does not work on Windows

Bring focus to figure window:
This seems to very tricky on Windows!!
See, e.g. https://stackoverflow.com/questions/22751100/tkinter-main-window-focus
Does not work:
fig.canvas.manager.window.focus_force()
fig.canvas.manager.window.focus_set()
fig.canvas.manager.window.bind("<FocusIn>", fig.canvas.manager.window.focus_set())

This should work, but I dont want to use it
import win32gui
root = tk.Tk()

# Application specific setup ...

win32gui.SetForegroundWindow(root.winfo_id())
Disclaimer: Stealing focus is bad behaviour,
so only use it if you must and if it makes sense,
e.g. input box the user will instantly type into.

not that fig_win.geometry() (and also the winfo functions)
is updated, when it is called with values or when plt.show() is called, not before
when creating a figure, the fig_win.geometry() is set with default values, e.g. 200x200 size

----

plt.figure object:
fig.get_window_extent().size
The tk-window adds 42 pixels (upper margin) and 30 pixels (lower) to fig.get_window_extent().size
"""

#%% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

#%% CUSTOM MARKER LIST
marker_list = ['o', 's', 'd', 'x', '+', '*', '^', 'v', '<', '>']

#%% CUSTOM FUNCTIONS
def make_dir_if_not_exists(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

def cm2inch(*tupl):
    inch = 2.54
    if hasattr(tupl[0], "__len__"):
        return tuple(cm/inch for cm in tupl[0])
    else:
        return tuple(cm/inch for cm in tupl)

def move_figure_window(fig, pos, FIG_WINDOW_START_POS, MPL_BACKEND):
    """

    :param fig:
    :param pos: position [x, y], where to move the figure in pixels
    :param FIG_WINDOW_START_POS:
    :param MPL_BACKEND:
    :return:
    """
    max_x = 1920 # windows displayed on screen with resolution max_x x max_y pix
    max_x = FIG_WINDOW_START_POS[0] + max_x
    max_y = 1080
    max_y = FIG_WINDOW_START_POS[1] + max_y

    # width and height in pixels evaluated by plt.figure
    FW_, FH_ = (fig.get_window_extent().size).astype(int)

    if MPL_BACKEND == 'TkAgg':
        lower_margin_size = 42 # lower margin of the interactive figure window. for TkAgg is 42 ;)
        upper_margin_size = 30 # upper margin of the interactive figure window. for TkAgg is 30
        FH_ += upper_margin_size
        fig_window = fig.canvas.manager.window # e.g. a tkinter object for TkAgg backend
        # set figure window position
        fig_window.geometry(f'{FW_}x{FH_}+{pos[0]}+{pos[1]}')
        fig_window.update()
    if MPL_BACKEND in ['QtAgg', 'Qt5Agg']:
        lower_margin_size = 0 # lower margin of the interactive figure window. for TkAgg is 42 ;)
        upper_margin_size = 32 # upper margin of the interactive figure window. for TkAgg is 30
        FH_ += upper_margin_size
        fig_window = fig.canvas.manager.window # e.g. a tkinter object for TkAgg backend
        # set figure window position
        fig_window.move(pos[0], pos[1])

    # add movement for other backends here, e.g. QtAgg -> not working so far in pyCharm
    FH_ += lower_margin_size #
    if pos[0] + 2 * FW_ > max_x:
        if pos[1] + 2 * FH_ > max_y:
            pos = np.copy(FIG_WINDOW_START_POS)
        else:
            pos[0] = FIG_WINDOW_START_POS[0]
            pos[1] += FH_ + upper_margin_size
    else:
        pos[0] += FW_

    return pos

def delete_all_files_from_dir(path_dir):
    for filename in os.listdir(path_dir):
        file_path = os.path.join(path_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def close_all_figs():
    plt.close('all')

def png2gif(path_dir_pngs, path_dir_gif, name_gif, duration=300, loop=1):
    import glob
    from PIL import Image
    path_gif = path_dir_gif + os.sep + name_gif + '.gif'
    frames = [Image.open(image) for image in glob.glob(path_dir_pngs + os.sep + '*.png')]
    frame_one = frames[0]
    frame_one.save(path_gif, format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=loop)

def check_for_backend_gui(MPL_BACKEND):
    if MPL_BACKEND in ['MacOSX', 'QtAgg', 'Qt5Agg', 'Qt4Agg', 'GTK4Agg', 'GTK3Agg', 'TkAgg',
                           'wxAgg']:
        return True
    else:
        return False

# figure default properties
LW = 1.2 # linewidth
MS = 2 # markersize
TTFS = 10 # title fontsize
LFS = 10 # axes label fontsize (legend = label-2)
TKFS = 10 # tick label fontsize (usually the numbers on the axes)
RASTER_DPI = 600 # raster resolution for saving the figure as file (e.g. .png)
SCREEN_DPI = 100 # figure resolution for displaying on computer screen
FONT_FAMILY = 'serif'
# INFO: CHOOSE FONT (ORIGINAL: CMU Serif / CMU Sans Serif)
FONT_SERIF = 'CMU Serif'
FONT_SANS_SERIF = 'CMU Sans Serif'
MATHTEXT_FONTSET = 'cm'
bool_use_latex = False
# bool_use_latex = True

# position, where the first figure window should appear in pixels
# for multiple displays (on Windows), set beyond the size of main screen to reach other screens
# additional figure windows will be placed beside the first one
FIG_WINDOW_START_POS = np.array([0, 0]).astype(int)

my_default_plot_settings = [LW, MS, TTFS, LFS, TKFS, RASTER_DPI, SCREEN_DPI, FONT_FAMILY,
                            FONT_SERIF, FONT_SANS_SERIF, MATHTEXT_FONTSET, bool_use_latex,
                            FIG_WINDOW_START_POS]
