from ert.simulator.batch_simulator_context import Status
from everest.simulator.simulator import Simulator

JOB_SUCCESS = "Success"
JOB_WAITING = "Waiting"
JOB_RUNNING = "Running"
JOB_FAILURE = "Failure"


DEFAULT_DATA_SUMMARY_KEYS = ("YEAR", "YEARS" "TCPU", "TCPUDAY", "MONTH", "DAY")


DEFAULT_FIELD_SUMMARY_KEYS = (
    "FOPR",
    "FOPT",
    "FOIR",
    "FOIT",
    "FWPR",
    "FWPT",
    "FWIR",
    "FWIT",
    "FGPR",
    "FGPT",
    "FGIR",
    "FGIT",
    "FVPR",
    "FVPT",
    "FVIR",
    "FVIT",
    "FWCT",
    "FGOR",
    "FOIP",
    "FOIPL",
    "FOIPG",
    "FWIP",
    "FGIP",
    "FGIPL",
    "FGIPG",
    "FPR",
    "FAQR",
    "FAQRG",
    "FAQT",
    "FAQTG",
    "FWGR",
)


DEFAULT_GROUP_SUMMARY_KEYS = (
    "GOPR",
    "GOPT",
    "GOIR",
    "GOIT",
    "GWPR",
    "GWPT",
    "GWIR",
    "GWIT",
    "GGPR",
    "GGPT",
    "GGIR",
    "GGIT",
    "GVPR",
    "GVPT",
    "GVIR",
    "GVIT",
    "GWCT",
    "GGOR",
    "GWGR",
)


_DEFAULT_WELL_SUMMARY_KEYS = (
    "WOPR",
    "WOPT",
    "WOIR",
    "WOIT",
    "WWPR",
    "WWPT",
    "WWIR",
    "WWIT",
    "WGPR",
    "WGPT",
    "WGIR",
    "WGIT",
    "WVPR",
    "WVPT",
    "WVIR",
    "WVIT",
    "WWCT",
    "WGOR",
    "WWGR",
    "WBHP",
    "WTHP",
    "WPI",
)

_EXCLUDED_TARGET_KEYS = "WGOR"

_DEFAULT_WELL_TARGET_SUMMARY_KEYS = (
    well_key + "T"
    for well_key in _DEFAULT_WELL_SUMMARY_KEYS
    if well_key.endswith("R") and well_key not in _EXCLUDED_TARGET_KEYS
)

DEFAULT_WELL_SUMMARY_KEYS = _DEFAULT_WELL_SUMMARY_KEYS + tuple(
    _DEFAULT_WELL_TARGET_SUMMARY_KEYS
)

__all__ = [
    "JOB_FAILURE",
    "JOB_RUNNING",
    "JOB_SUCCESS",
    "JOB_WAITING",
    "Simulator",
    "Status",
]
