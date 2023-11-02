import shutil
from typing import Any, List, Optional, Tuple, Union, no_type_check

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, validator

from .analysis_module import AnalysisMode
from .parsing import ConfigKeys, HistorySource, HookRuntime, QueueSystem

DEFAULT_ENSPATH = "storage"
DEFAULT_RUNPATH_FILE = ".ert_runpath_list"
DEFAULT_HISTORY_SOURCE = HistorySource.REFCASE_HISTORY
DEFAULT_RUNPATH = "simulations/realization-<IENS>/iter-<ITER>"
DEFAULT_GEN_KW_EXPORT_NAME = "parameters"
DEFAULT_JOB_SCRIPT = shutil.which("job_dispatch.py") or "job_dispatch.py"
DEFAULT_MAX_SUBMIT = 2
DEFAULT_QUEUE_SYSTEM = QueueSystem.LOCAL
DEFAULT_ENKF_ALPHA = 3.0
DEFAULT_STD_CUTOFF = 1e-6
DEFAULT_UPDATE_LOG_PATH = "update_log"
DEFAULT_ITER_COUNT = 4
DEFAULT_RETRY_COUNT = 4


class ErtConfigValues(BaseModel):
    num_realizations: PositiveInt = 1
    eclbase: Optional[str] = None
    run_template: List[Tuple[str, str]] = Field(default_factory=list)
    define: List[Tuple[str, str]] = Field(default_factory=list)
    stop_long_running: bool = False
    data_kw: List[Tuple[str, str]] = Field(default_factory=list)
    data_file: Optional[str] = None
    grid: Optional[str] = None
    job_script: str = DEFAULT_JOB_SCRIPT
    jobname: Optional[str] = None
    runpath: str = DEFAULT_RUNPATH
    time_map: Optional[str] = None
    obs_config: Optional[str] = None
    history_source: HistorySource = HistorySource.REFCASE_HISTORY
    gen_kw_export_name: str = DEFAULT_GEN_KW_EXPORT_NAME
    max_submit: PositiveInt = DEFAULT_MAX_SUBMIT
    num_cpu: Optional[PositiveInt] = None
    iter_case: Optional[str] = None
    iter_retry_count: PositiveInt = DEFAULT_RETRY_COUNT
    iter_count: PositiveInt = DEFAULT_ITER_COUNT
    min_realizations: str = "0"
    update_log_path: str = DEFAULT_UPDATE_LOG_PATH
    max_runtime: NonNegativeInt = 0
    enkf_alpha: float = DEFAULT_ENKF_ALPHA
    std_cutoff: float = DEFAULT_STD_CUTOFF
    queue_system: QueueSystem = DEFAULT_QUEUE_SYSTEM
    queue_option: List[
        Union[Tuple[QueueSystem, str], Tuple[QueueSystem, str, str]]
    ] = Field(default_factory=list)
    analysis_set_var: List[Tuple[str, str, Any]] = Field(default_factory=list)
    analysis_select: AnalysisMode = AnalysisMode.ENSEMBLE_SMOOTHER
    random_seed: Optional[int] = None
    field: List[Tuple[str, ...]] = Field(default_factory=list)
    gen_data: List[Tuple[str, ...]] = Field(default_factory=list)
    gen_kw: List[Tuple[str, ...]] = Field(default_factory=list)
    surface: List[Tuple[str, ...]] = Field(default_factory=list)
    summary: List[Tuple[str, ...]] = Field(default_factory=list)
    refcase: Optional[str] = None
    setenv: List[Tuple[str, str]] = Field(default_factory=list)
    runpath_file: str = DEFAULT_RUNPATH_FILE
    enspath: str = DEFAULT_ENSPATH
    simulation_job: List[List[str]] = Field(default_factory=list)
    forward_model: List[Union[Tuple[str], Tuple[str, List[Tuple[str, str]]]]] = Field(
        default_factory=list
    )
    install_job_directory: List[str] = Field(default_factory=list)
    install_job: List[Tuple[str, str]] = Field(default_factory=list)
    load_workflow_job: List[Tuple[str, ...]] = Field(default_factory=list)
    workflow_job_directory: List[str] = Field(default_factory=list)
    hook_workflow: List[Tuple[str, HookRuntime]] = Field(default_factory=list)
    load_workflow: List[Tuple[str, ...]] = Field(default_factory=list)
    config_directory: Optional[str] = None

    class Config:
        @staticmethod
        def alias_generator(x: str) -> str:
            return x.upper()

    @no_type_check
    @validator("stop_long_running")
    def convert_to_bool(cls, v):
        return bool(v)

    @no_type_check
    def to_config_dict(self):
        result = {
            ConfigKeys.FORWARD_MODEL: self.forward_model,
            ConfigKeys.SIMULATION_JOB: self.simulation_job,
            ConfigKeys.NUM_REALIZATIONS: self.num_realizations,
            ConfigKeys.RUNPATH_FILE: self.runpath_file,
            ConfigKeys.RUN_TEMPLATE: self.run_template,
            ConfigKeys.ENKF_ALPHA: self.enkf_alpha,
            ConfigKeys.ITER_CASE: self.iter_case,
            ConfigKeys.ITER_COUNT: self.iter_count,
            ConfigKeys.ITER_RETRY_COUNT: self.iter_retry_count,
            ConfigKeys.UPDATE_LOG_PATH: self.update_log_path,
            ConfigKeys.STD_CUTOFF: self.std_cutoff,
            ConfigKeys.MAX_RUNTIME: self.max_runtime,
            ConfigKeys.MIN_REALIZATIONS: self.min_realizations,
            ConfigKeys.DEFINE: self.define,
            ConfigKeys.STOP_LONG_RUNNING: self.stop_long_running,
            ConfigKeys.DATA_KW: self.data_kw,
            ConfigKeys.DATA_FILE: self.data_file,
            ConfigKeys.GRID: self.grid,
            ConfigKeys.JOB_SCRIPT: self.job_script,
            ConfigKeys.RUNPATH: self.runpath,
            ConfigKeys.ENSPATH: self.enspath,
            ConfigKeys.TIME_MAP: self.time_map,
            ConfigKeys.OBS_CONFIG: self.obs_config,
            ConfigKeys.HISTORY_SOURCE: self.history_source.value,
            ConfigKeys.REFCASE: self.refcase,
            ConfigKeys.GEN_KW_EXPORT_NAME: self.gen_kw_export_name,
            ConfigKeys.FIELD: self.field,
            ConfigKeys.GEN_DATA: self.gen_data,
            ConfigKeys.MAX_SUBMIT: self.max_submit,
            ConfigKeys.NUM_CPU: self.num_cpu,
            ConfigKeys.QUEUE_SYSTEM: self.queue_system,
            ConfigKeys.QUEUE_OPTION: self.queue_option,
            ConfigKeys.ANALYSIS_SET_VAR: self.analysis_set_var,
            ConfigKeys.ANALYSIS_SELECT: self.analysis_select.value,
            ConfigKeys.INSTALL_JOB: self.install_job,
            ConfigKeys.INSTALL_JOB_DIRECTORY: self.install_job_directory,
            ConfigKeys.RANDOM_SEED: self.random_seed,
            ConfigKeys.SETENV: self.setenv,
        }
        if self.eclbase is not None:
            result[ConfigKeys.ECLBASE] = self.eclbase
        if self.jobname is not None:
            result[ConfigKeys.JOBNAME] = self.jobname
        return result
