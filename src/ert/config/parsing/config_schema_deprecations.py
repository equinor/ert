from functools import partial
from typing import cast

from .deprecation_info import DeprecationInfo

REPLACE_WITH_GEN_KW = [
    "RELPERM",
    "MULTZ",
    "EQUIL",
    "GEN_PARAM",
    "MULTFLT",
]
JUST_REMOVE_KEYWORDS = [
    "ITER_CASE",
    "ITER_COUNT",
    "ITER_RETRY_COUNT",
    "UMASK",
    "LICENSE_PATH",
    "LOG_FILE",
    "LOG_LEVEL",
    "ENKF_RERUN",
    "GEN_KW_TAG_FORMAT",
    "ENKF_BOOTSTRAP",
    "ENKF_MODE",
    "ANALYSIS_SELECT",
    "RESULT_PATH",
    "DATA_ROOT",
]
RSH_KEYWORDS = ["RSH_HOST", "RSH_COMMAND", "MAX_RUNNING_RSH"]
USE_QUEUE_OPTION = [
    "LSF_QUEUE",
    "MAX_RUNNING_LSF",
    "MAX_RUNNING_LOCAL",
]

deprecated_keywords_list = [
    *[
        DeprecationInfo(
            keyword=kw,
            message=partial(
                lambda line, kw: f"Using {kw} with substitution strings "
                + "that are not of the form '<KEY>' is deprecated. "
                + f"Please change {line[0]} to "
                + f"<{line[0].replace('<', '').replace('>', '')}>",
                kw=kw,
            ),
            check=lambda line: not DeprecationInfo.is_angle_bracketed(str(line[0])),
        )
        for kw in ["DEFINE", "DATA_KW"]
    ],
    DeprecationInfo(
        keyword="QUEUE_OPTION",
        message=(
            "JOB_PREFIX as QUEUE_OPTION to the TORQUE system is deprecated. "
            "Please add your prefix to JOBNAME instead."
        ),
        check=lambda line: "JOB_PREFIX" in line,
    ),
    DeprecationInfo(
        keyword="ANALYSIS_SET_VAR",
        message=partial(
            lambda line: f"The {line[1]} keyword was removed in 2017 and has no "
            "effect. It has been used in the past to force the subspace "
            "dimension set using ENKF_NCOMP keyword. "
            "This can be safely removed.",
        ),
        check=lambda line: str(line[1]) == "ENKF_FORCE_NCOMP",
    ),
    DeprecationInfo(
        keyword="RUNPATH",
        message=lambda line: "RUNPATH keyword contains deprecated value "
        f"placeholders: %d, instead use: "
        f"{line[0].replace('%d', '<IENS>', 1).replace('%d', '<ITER>', 1)}",
        check=lambda line: any("%d" in str(v) for v in line),
    ),
    *[
        DeprecationInfo(
            keyword=kw,
            message=f"The {kw} keyword was replaced by the GEN_KW keyword. "
            "Please see https://ert.readthedocs.io/en/latest/"
            "reference/configuration/keywords.html#gen-kw "
            "to see how to migrate from MULTFLT to GEN_KW.",
        )
        for kw in REPLACE_WITH_GEN_KW
    ],
    *[
        DeprecationInfo(
            keyword=kw,
            message=f"The keyword {kw} no longer has any effect"
            f", and can be safely removed.",
        )
        for kw in JUST_REMOVE_KEYWORDS
    ],
    *[
        DeprecationInfo(
            keyword=kw,
            message=f"The {kw} was used for the deprecated and removed "
            "support for RSH queues. It no longer has any effect "
            "and can safely be removed from the config file.",
        )
        for kw in RSH_KEYWORDS
    ],
    *[
        DeprecationInfo(
            keyword=kw,
            message=f"The {kw} keyword has been removed. For most cases this option "
            "should be set in the site config, and as a regular user you can "
            "simply remove this from your config. If you need to set these "
            "options it is now done via the QUEUE_OPTION keyword.",
        )
        for kw in USE_QUEUE_OPTION
    ],
    DeprecationInfo(
        keyword="SCHEDULE_PREDICTION_FILE",
        message="The 'SCHEDULE_PREDICTION_FILE' config keyword "
        "has been removed and no longer has any effect",
    ),
    DeprecationInfo(
        keyword="HAVANA_FAULT",
        message="Direct interoperability with havana was removed from ert in 2009."
        " The behavior of HAVANA_FAULT can be reproduced using"
        " GEN_KW and FORWARD_MODEL.",
    ),
    DeprecationInfo(
        keyword="REFCASE_LIST",
        message="The REFCASE_LIST keyword was used to give a .DATA file "
        "to be used for plotting. The corresponding plotting functionality "
        "was removed in 2015, and this keyword can be safely removed from "
        "the config file.",
    ),
    DeprecationInfo(
        keyword="RFTPATH",
        message="The RFTPATH keyword was used to give a path to well observations "
        "to be used for plotting. The corresponding plotting functionality "
        "was removed in 2015, and the RFTPATH keyword can safely be removed "
        "from the config file.",
    ),
    DeprecationInfo(
        keyword="END_DATE",
        message="The END_DATE keyword was used to check that a the dates in a summary "
        "file would go beyond a certain date. This would only display a "
        "warning in case of problems. The keyword has since been deprecated, "
        "and can be safely removed from the config file.",
    ),
    DeprecationInfo(
        keyword="CASE_TABLE",
        message="The CASE_TABLE keyword was used with a deprecated sensitivity "
        "analysis feature to give descriptive names to cases. It no longer has "
        "any effect and can safely be removed from the config file.",
    ),
    DeprecationInfo(
        keyword="RERUN_START",
        message="The RERUN_START keyword was used for the deprecated run mode "
        "ENKF_ASSIMILATION which was removed in 2016. It does not have "
        "any effect on run modes currently supported by ERT, and can "
        "be safely removed.",
    ),
    DeprecationInfo(
        keyword="DELETE_RUNPATH",
        message="The DELETE_RUNPATH keyword would clear the runpath directories "
        "between runs. It was removed in 2017 and no longer has any effect.",
    ),
    DeprecationInfo(
        keyword="PLOT_SETTINGS",
        message="The keyword PLOT_SETTINGS was removed in 2019 and has no effect. "
        "It can safely be removed. It was used for controlling settings for outputting"
        " qc plots to disk and what plots were shown in the GUI. All plots can now"
        " be selected on request from the GUI, and there is also an alternative"
        " view accessible by using ert viz in the commandline.",
    ),
    DeprecationInfo(
        keyword="UPDATE_PATH",
        message="The UPDATE_PATH keyword has been removed and no longer has any effect."
        " It has been used in the past to set different python versions for the "
        "forward model. This should no longer be necessary."
        "If your setup is not longer working, do not hesitate to contact us.",
    ),
    DeprecationInfo(
        keyword="UPDATE_SETTINGS",
        message="The UPDATE_SETTINGS keyword has been removed and no longer has any "
        "effect. It has been used in the past to adjust control parameters "
        "for the Ensemble Smoother update algorithm. "
        "Please use ENKF_ALPHA and STD_CUTOFF keywords instead.",
    ),
    DeprecationInfo(
        keyword="QUEUE_OPTION",
        message="Memory requirements in LSF should now be set using REALIZATION_MEMORY "
        "and not through the LSF_RESOURCE option.",
        check=lambda line: "LSF_RESOURCE" in line
        and "mem=" in cast(list[str], line)[-1],
    ),
    DeprecationInfo(
        keyword="SIMULATION_JOB",
        message=lambda args: "The SIMULATION_JOB keyword has been removed and no "
        "longer has any effect. It has been used in the as a lightweight alternative "
        "of FORWARD_MODEL. Please use FORWARD_MODEL keyword instead:\n"
        f"FORWARD_MODEL {' '.join(args) if args else ''}",
    ),
    DeprecationInfo(
        keyword="CONFIG_DIRECTORY",
        message="The CONFIG_DIRECTORY keyword has been removed and no longer has any "
        "effect. It has been used in the past to specify the experiment "
        "configuration path if provided."
        "It has been replaced with the path to the directory containing the "
        "configuration file for the experiment.",
    ),
]
