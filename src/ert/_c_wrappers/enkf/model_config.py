import os

from cwrap import BaseCClass
from ecl.summary import EclSum

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.time_map import TimeMap
from ert._c_wrappers.job_queue import ForwardModel
from ert._c_wrappers.sched import HistorySourceEnum
from ert._c_wrappers.util import PathFormat


class ModelConfig(BaseCClass):
    TYPE_NAME = "model_config"

    _alloc = ResPrototype(
        "void*  model_config_alloc(config_content, \
                                   char*, ext_joblist, \
                                   ecl_sum)",
        bind=False,
    )
    _alloc_full = ResPrototype(
        "void*  model_config_alloc_full(int, \
                                        int, \
                                        char*, \
                                        char*, \
                                        char*, \
                                        char*, \
                                        forward_model, \
                                        char*, \
                                        time_map, \
                                        char*, \
                                        history_source_enum, \
                                        ext_joblist, \
                                        ecl_sum)",
        bind=False,
    )
    _free = ResPrototype("void  model_config_free( model_config )")
    _get_forward_model = ResPrototype(
        "forward_model_ref model_config_get_forward_model(model_config)"
    )
    _get_max_internal_submit = ResPrototype(
        "int   model_config_get_max_internal_submit(model_config)"
    )
    _get_runpath_as_char = ResPrototype(
        "char* model_config_get_runpath_as_char(model_config)"
    )
    _select_runpath = ResPrototype(
        "bool  model_config_select_runpath(model_config, char*)"
    )
    _set_runpath = ResPrototype("void  model_config_set_runpath(model_config, char*)")
    _get_enspath = ResPrototype("char* model_config_get_enspath(model_config)")
    _get_history_source = ResPrototype(
        "history_source_enum model_config_get_history_source(model_config)"
    )
    _select_history = ResPrototype(
        "bool  model_config_select_history(model_config, history_source_enum, ecl_sum)"
    )
    _gen_kw_export_name = ResPrototype(
        "char* model_config_get_gen_kw_export_name(model_config)"
    )
    _runpath_requires_iterations = ResPrototype(
        "bool  model_config_runpath_requires_iter(model_config)"
    )
    _get_jobname_fmt = ResPrototype("char* model_config_get_jobname_fmt(model_config)")
    _get_runpath_fmt = ResPrototype(
        "path_fmt_ref model_config_get_runpath_fmt(model_config)"
    )
    _get_num_realizations = ResPrototype(
        "int model_config_get_num_realizations(model_config)"
    )
    _get_obs_config_file = ResPrototype(
        "char* model_config_get_obs_config_file(model_config)"
    )
    _get_data_root = ResPrototype("char* model_config_get_data_root(model_config)")
    _get_time_map = ResPrototype(
        "time_map_ref model_config_get_external_time_map(model_config)"
    )
    _get_last_history_restart = ResPrototype(
        "int model_config_get_last_history_restart(model_config)"
    )

    def __init__(
        self,
        data_root,
        joblist,
        refcase,
        config_content=None,
        config_dict=None,
        is_reference=False,
    ):
        if config_dict is not None and config_content is not None:
            raise ValueError(
                "Error: Unable to create ModelConfig with multiple config objects"
            )

        if config_dict is None:
            c_ptr = self._alloc(config_content, data_root, joblist, refcase)
        else:
            # MAX_RESAMPLE_KEY
            max_resample = config_dict.get(ConfigKeys.MAX_RESAMPLE, 1)

            # NUM_REALIZATIONS_KEY
            num_realizations = config_dict.get(ConfigKeys.NUM_REALIZATIONS)

            # RUNPATH_KEY
            run_path = config_dict.get(
                ConfigKeys.RUNPATH, "simulations/realization-<IENS>/iter-<ITER>"
            )
            if run_path is not None:
                run_path = os.path.realpath(run_path)

            # DATA_ROOT_KEY
            data_root_from_config = config_dict.get(ConfigKeys.DATAROOT)
            if data_root_from_config is not None:
                data_root = os.path.realpath(data_root_from_config)

            # ENSPATH_KEY
            ens_path = config_dict.get(ConfigKeys.ENSPATH, "storage")
            if ens_path is not None:
                ens_path = os.path.realpath(ens_path)

            # JOBNAME_KEY
            job_name = config_dict.get(ConfigKeys.JOBNAME)

            # FORWARD_MODEL_KEY
            forward_model = ForwardModel(ext_joblist=joblist)
            # SIMULATION_JOB_KEY
            for job_description in config_dict.get(ConfigKeys.FORWARD_MODEL, []):
                job = forward_model.add_job(job_description[ConfigKeys.NAME])
                job.set_private_args_as_string(job_description.get(ConfigKeys.ARGLIST))
                job.convertToCReference(None)

            # SIMULATION_JOB_KEY
            for job_description in config_dict.get(ConfigKeys.SIMULATION_JOB, []):
                job = forward_model.add_job(job_description[ConfigKeys.NAME])
                job.set_private_args_as_string(job_description.get(ConfigKeys.ARGLIST))
                job.convertToCReference(None)

            # OBS_CONFIG_KEY
            obs_config = config_dict.get(ConfigKeys.OBS_CONFIG)
            if obs_config is not None:
                obs_config = os.path.realpath(obs_config)

            # TIME_MAP_KEY
            time_map = None
            time_map_file = config_dict.get(ConfigKeys.TIME_MAP)
            if time_map_file is not None and not os.path.isfile(
                os.path.realpath(time_map_file)
            ):
                raise ValueError("Error: Time map is not a file")
            if time_map_file is not None:
                time_map = TimeMap()
                time_map.fload(filename=os.path.realpath(time_map_file))

            # GEN_KW_EXPORT_NAME_KEY
            gen_kw_export_name = config_dict.get(ConfigKeys.GEN_KW_EXPORT_NAME)

            # HISTORY_SOURCE_KEY
            history_source = config_dict.get(
                ConfigKeys.HISTORY_SOURCE, HistorySourceEnum.REFCASE_HISTORY
            )

            c_ptr = self._alloc_full(
                max_resample,
                num_realizations,
                run_path,
                data_root,
                ens_path,
                job_name,
                forward_model,
                obs_config,
                time_map,
                gen_kw_export_name,
                history_source,
                joblist,
                refcase,
            )

            # Fix ownership
            forward_model.convertToCReference(None)
            if time_map is not None:
                time_map.convertToCReference(None)

        if c_ptr is None:
            raise ValueError("Failed to construct ModelConfig instance.")

        super().__init__(c_ptr, is_reference=is_reference)

    def get_history_source(self) -> HistorySourceEnum:
        return self._get_history_source()

    def set_history_source(
        self, history_source: HistorySourceEnum, refcase: EclSum
    ) -> bool:
        assert isinstance(history_source, HistorySourceEnum)
        assert isinstance(refcase, EclSum)
        return self._select_history(history_source, refcase)

    def get_max_internal_submit(self) -> int:
        return self._get_max_internal_submit()

    def set_max_internal_submit(self, max_value):
        self._get_max_internal_submit(max_value)

    def getForwardModel(self) -> ForwardModel:
        return self._get_forward_model().setParent(self)

    def getRunpathAsString(self) -> str:
        return self._get_runpath_as_char()

    def selectRunpath(self, path_key) -> bool:
        return self._select_runpath(path_key)

    def setRunpath(self, path_format):
        self._set_runpath(path_format)

    def free(self):
        self._free()

    def getGenKWExportName(self) -> str:
        return self._gen_kw_export_name()

    def runpathRequiresIterations(self) -> bool:
        return self._runpath_requires_iterations()

    def getJobnameFormat(self) -> str:
        return self._get_jobname_fmt()

    @property
    def obs_config_file(self):
        return self._get_obs_config_file()

    def getEnspath(self) -> str:
        return self._get_enspath()

    def getRunpathFormat(self) -> PathFormat:
        return self._get_runpath_fmt()

    @property
    def num_realizations(self):
        return self._get_num_realizations()

    def data_root(self):
        return self._get_data_root()

    def get_time_map(self):
        return self._get_time_map()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        if self.data_root() != other.data_root():
            return False

        if self.num_realizations != other.num_realizations:
            return False

        if os.path.realpath(self.obs_config_file) != os.path.realpath(
            other.obs_config_file
        ):
            return False

        if os.path.realpath(self.getEnspath()) != os.path.realpath(other.getEnspath()):
            return False

        if self.getRunpathFormat() != other.getRunpathFormat():
            return False

        if self.getJobnameFormat() != other.getJobnameFormat():
            return False

        if os.path.realpath(self.getRunpathAsString()) != os.path.realpath(
            other.getRunpathAsString()
        ):
            return False

        if self.get_max_internal_submit() != other.get_max_internal_submit():
            return False

        if self.getGenKWExportName() != other.getGenKWExportName():
            return False

        if self.get_time_map() != other.get_time_map():
            return False

        if self.getForwardModel() != other.getForwardModel():
            return False

        return True

    @staticmethod
    def _get_history_src_enum(config_dict, config_content):
        hist_src_enum = None
        if config_dict and ConfigKeys.HISTORY_SOURCE in config_dict:
            hist_src_enum = config_dict.get(ConfigKeys.HISTORY_SOURCE)

        if config_content and config_content.hasKey(ConfigKeys.HISTORY_SOURCE):
            hist_src_str = config_content.getValue(ConfigKeys.HISTORY_SOURCE)
            hist_src_enum = HistorySourceEnum.from_string(hist_src_str)

        return hist_src_enum

    def get_last_history_restart(self):
        return self._get_last_history_restart()
