from typing import Any, Dict, List

from ert._c_wrappers.config import ConfigContent
from ert._c_wrappers.enkf.config_keys import ConfigKeys

# keywords that have one argument and
# the last occurrence of the keyword in the file
# is the value of which we use.
# ie. if the config contains:
#
# MAX_SUBMIT 2
# MAX_SUBMIT 3
# ert will use the value 3 for MAX_SUBMIT
SINGLE_OCCURRENCE_SINGLE_ARG_KEYS = [
    ConfigKeys.ALPHA_KEY,
    ConfigKeys.ANALYSIS_SELECT,
    ConfigKeys.CONFIG_DIRECTORY,
    ConfigKeys.DATAROOT,
    ConfigKeys.DATA_FILE,
    ConfigKeys.ECLBASE,
    ConfigKeys.ENSPATH,
    ConfigKeys.GEN_KW_EXPORT_NAME,
    ConfigKeys.GEN_KW_TAG_FORMAT,
    ConfigKeys.GRID,
    ConfigKeys.HISTORY_SOURCE,
    ConfigKeys.ITER_CASE,
    ConfigKeys.ITER_COUNT,
    ConfigKeys.ITER_RETRY_COUNT,
    ConfigKeys.JOBNAME,
    ConfigKeys.JOB_SCRIPT,
    ConfigKeys.LICENSE_PATH,
    ConfigKeys.MAX_RUNTIME,
    ConfigKeys.MAX_SUBMIT,
    ConfigKeys.MIN_REALIZATIONS,
    ConfigKeys.NUM_CPU,
    ConfigKeys.NUM_REALIZATIONS,
    ConfigKeys.OBS_CONFIG,
    ConfigKeys.QUEUE_SYSTEM,
    ConfigKeys.RANDOM_SEED,
    ConfigKeys.REFCASE,
    ConfigKeys.RERUN_KEY,
    ConfigKeys.RUNPATH,
    ConfigKeys.RUNPATH_FILE,
    ConfigKeys.STD_CUTOFF_KEY,
    ConfigKeys.STOP_LONG_RUNNING,
    ConfigKeys.TIME_MAP,
    ConfigKeys.UPDATE_LOG_PATH,
]

# Like SINGLE_OCCURRENCE_SINGLE_ARG_KEYS but
# the keyword can take more than one argument, ie.
# SCHEDULE_PREDICTION_FILE filename  <parameters:> <init_files:>
SINGLE_OCCURRENCE_MULTI_ARG_KEYS = [
    ConfigKeys.SCHEDULE_PREDICTION_FILE,
]

# Keywords that can occur more than once, but
# the number of arguments is always one.
MULTI_OCCURRENCE_SINGLE_ARG_KEYS = [
    ConfigKeys.WORKFLOW_JOB_DIRECTORY,
    ConfigKeys.INSTALL_JOB_DIRECTORY,
]

# Some keys have been split prematurely and must been
# joined back, ie.

# QUEUE_OPTION LOCAL opt the value of the option

# Should be interpreted as having three arguments:
# ["LOCAL", "opt", "the value of the option"]
JOIN_KEYS = [
    (ConfigKeys.QUEUE_OPTION, 2),
    (ConfigKeys.FORWARD_MODEL, 0),
]


def config_content_as_dict(
    user_config_content: "ConfigContent", site_config_content: "ConfigContent"
) -> Dict[str, List[Any]]:
    content_dict: Dict[str, List[Any]] = {}
    for key in set(list(user_config_content.keys()) + list(site_config_content.keys())):
        items = []
        if key in site_config_content:
            items.append(site_config_content[key])
        if key in user_config_content:
            items.append(user_config_content[key])
        for item in items:
            if key in SINGLE_OCCURRENCE_SINGLE_ARG_KEYS:
                content_dict[key] = item.getValue()
            elif key in SINGLE_OCCURRENCE_MULTI_ARG_KEYS:
                for node in item:
                    content_dict[key] = list(node)
            elif key in MULTI_OCCURRENCE_SINGLE_ARG_KEYS:
                if key not in content_dict:
                    content_dict[key] = []
                for node in item:
                    content_dict[key].append(list(node)[0])
            else:
                if key not in content_dict:
                    content_dict[key] = []
                for node in item:
                    values = list(node)
                    content_dict[key].append(values)

    for key, join_at in JOIN_KEYS:
        if key in content_dict:
            for occurrence in content_dict[key]:
                if len(occurrence) > join_at:
                    occurrence[join_at] = " ".join(occurrence[join_at:])
                    del occurrence[join_at + 1 :]

    # Add the defines if they exits
    defines = []
    if isinstance(site_config_content, ConfigContent):
        defines += [
            [key, val] for key, val in site_config_content.get_const_define_list()
        ]
    if isinstance(user_config_content, ConfigContent):
        defines += [
            [key, val] for key, val in user_config_content.get_const_define_list()
        ]
    if defines:
        content_dict[ConfigKeys.DEFINE_KEY] = defines

    return content_dict
