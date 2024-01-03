import os
import warnings

if (_sched_str := os.environ.get("ERT_FEATURE_SCHEDULER")) not in ["ON", "OFF", None]:
    warnings.warn(
        "ERT_FEATURE_SCHEDULER environment variable "
        f"was given invalid value {_sched_str}, should be either ON or OFF."
        "Default value OFF is used.",
        stacklevel=1,
    )

if SCHEDULER_ENABLED := (_sched_str == "ON"):
    warnings.warn("Experimental feature SCHEDULER is enabled", stacklevel=1)
