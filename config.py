import matplotlib as mpl
from plot_tools import marker_list

# possible turbine IDs
turb_ids = [
    3, 4, 5, 6,
]

# turbine keys (identifiers for dicts)
turb_keys = ['T' + str(i) for i in turb_ids]

# possible variable keys (identifiers for dicts)
var_keys = [
    'Power',
    'WSpeed',
    'WDir',
    'Yaw',
    'Pitch',
    'Errorcode',
    'PowerRef',
    'ControlSwitch'
]

# corresponds to var_keys
unit_list = [
    'kW',
    'm/s',
    'deg',
    'deg',
    'deg',
    '-',
    'kW',
    '-'
]

unit_dict = {}
for key_n, key in enumerate(var_keys):
    unit_dict[key] = unit_list[key_n]

# corresponds to var_keys
filenames_beginnings = [
    'VA_WindTurbineActivePowerOutput_Avg1s_N3_',
    'VA_WindSpeed_Avg10s_N3_',
    'VA_WindDirectionModulus_Avg10s_N3_',
    'VA_YawPositionModulus_N3_',
    'VA_PitchMinimumPositionOfAllBlades_N3_',
    'VA_OperationState_N3_',
    'VA_ActivePowerReferenceDemandCommand_Param1_N3_',
    'VA_YawWindTrailingFunctionActive_N3_'
]

filenames_dict = {}
for turb_n, turb_key in enumerate(turb_keys):
    filenames_dict[turb_key] = {}
    for quant_n, var_key in enumerate(var_keys):
        filenames_dict[turb_key][var_key] = \
            filenames_beginnings[quant_n] + str(turb_ids[turb_n]) + '.ASC'

# label_dict = {}
# for turb_key in turb_keys:
#     label_dict[turb_key] = {}
#     for var_key in var_keys:
#         label_dict[turb_key][var_key] = var_key + '_' + turb_key

# PLOT SETTINGS
color_list = mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color']

color_dict_by_turb = {
    'T3': color_list[0],
    'T4': color_list[1],
    'T5': color_list[2],
    'T6': color_list[3],
}

marker_dict_by_turb = {
    'T3': marker_list[0],
    'T4': marker_list[1],
    'T5': marker_list[2],
    'T6': marker_list[3],
}
