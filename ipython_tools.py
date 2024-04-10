# from IPython import get_ipython # not necessary, might lead to conflicts
# do not import matplotlib or pyplot here!
def check_ipython() -> bool:
    try:
        ip_ = get_ipython()
        return True
    except NameError:
        return False

def set_mpl_magic(MPL_BACKEND):
    bool_ipython = check_ipython()
    if bool_ipython and MPL_BACKEND=='TkAgg':
        try:
            get_ipython().run_line_magic('matplotlib', 'tk')
        except ModuleNotFoundError:
            pass
    elif bool_ipython and MPL_BACKEND in ['Qt5Agg', 'QtAgg']:
        try:
            get_ipython().run_line_magic('matplotlib', 'qt')
        except ModuleNotFoundError:
            pass
    return bool_ipython
