#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'model_config.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from cwrap import BaseCClass

from ecl.summary import EclSum
from ecl.util.util import StringList
from res import ResPrototype
from res.job_queue import ForwardModel, ExtJob, ExtJoblist
from res.sched import HistorySourceEnum
from res.util import PathFormat
from res.enkf import ConfigKeys
from res.enkf.util import TimeMap

import os


class ModelConfig(BaseCClass):
    TYPE_NAME = "model_config"

    _alloc = ResPrototype(
        "void*  model_config_alloc( config_content, \
                                                                            char*, ext_joblist, \
                                                                            int, ecl_sum)",
        bind=False,
    )
    _alloc_full = ResPrototype(
        "void*  model_config_alloc_full( int, \
                                                                                int, \
                                                                                char*, \
                                                                                char*, \
                                                                                char*, \
                                                                                char*, \
                                                                                forward_model, \
                                                                                char*, \
                                                                                time_map, \
                                                                                char*, \
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
    _set_max_internal_submit = ResPrototype(
        "void  model_config_set_max_internal_submit(model_config, int)"
    )
    _get_runpath_as_char = ResPrototype(
        "char* model_config_get_runpath_as_char(model_config)"
    )
    _select_runpath = ResPrototype(
        "bool  model_config_select_runpath(model_config, char*)"
    )
    _set_runpath = ResPrototype("void  model_config_set_runpath(model_config, char*)")
    _get_enspath = ResPrototype("char* model_config_get_enspath(model_config)")
    _get_history = ResPrototype("history_ref model_config_get_history(model_config)")
    _get_history_source = ResPrototype(
        "history_source_enum model_config_get_history_source(model_config)"
    )
    _select_history = ResPrototype(
        "bool  model_config_select_history(model_config, history_source_enum, ecl_sum)"
    )
    _has_history = ResPrototype("bool  model_config_has_history(model_config)")
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
    _set_data_root = ResPrototype(
        "void model_config_get_data_root(model_config, char*)"
    )
    _get_time_map = ResPrototype(
        "void model_config_get_external_time_map(model_config)"
    )

    def __init__(
        self,
        data_root,
        joblist,
        last_history_restart,
        refcase,
        config_content=None,
        config_dict=None,
        is_reference=False,
    ):
        if config_dict is not None and config_content is not None:
            raise ValueError(
                "Error: Unable to create ModelConfig with multiple config objects"
            )

        hist_src_enum = ModelConfig._get_history_src_enum(config_dict, config_content)
        if hist_src_enum == HistorySourceEnum.SCHEDULE:
            raise ValueError(
                "{} as {} is not supported".format(
                    HistorySourceEnum.SCHEDULE, ConfigKeys.HISTORY_SOURCE
                )
            )

        if config_dict is None:
            c_ptr = self._alloc(
                config_content, data_root, joblist, last_history_restart, refcase
            )
        else:
            # MAX_RESAMPLE_KEY
            max_resample = config_dict.get(ConfigKeys.MAX_RESAMPLE)

            # NUM_REALIZATIONS_KEY
            num_realizations = config_dict[ConfigKeys.NUM_REALIZATIONS]

            # RUNPATH_KEY
            run_path = config_dict.get(ConfigKeys.RUNPATH)
            if run_path is not None:
                run_path = os.path.realpath(run_path)

            # DATA_ROOT_KEY
            data_root_from_config = config_dict.get(ConfigKeys.DATAROOT)
            if data_root_from_config is not None:
                data_root = os.path.realpath(data_root_from_config)

            # ENSPATH_KEY
            ens_path = config_dict.get(ConfigKeys.ENSPATH)
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
            elif time_map_file is not None:
                time_map = TimeMap()
                time_map.fload(filename=os.path.realpath(time_map_file))

            # RFTPATH_KEY
            rft_path = config_dict.get(ConfigKeys.RFTPATH)
            if rft_path is not None:
                rft_path = os.path.realpath(rft_path)

            # GEN_KW_EXPORT_NAME_KEY
            gen_kw_export_name = config_dict.get(ConfigKeys.GEN_KW_EXPORT_NAME)

            # HISTORY_SOURCE_KEY
            history_source = config_dict.get(ConfigKeys.HISTORY_SOURCE)

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
                rft_path,
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

        super(ModelConfig, self).__init__(c_ptr, is_reference=is_reference)

    def hasHistory(self):
        return self._has_history()

    def get_history_source(self):
        """ @rtype: HistorySourceEnum """
        return self._get_history_source()

    def set_history_source(self, history_source, refcase):
        """
        @type history_source: HistorySourceEnum
        @type refcase: EclSum
        @rtype: bool
        """
        assert isinstance(history_source, HistorySourceEnum)
        assert isinstance(refcase, EclSum)
        return self._select_history(history_source, refcase)

    def get_max_internal_submit(self):
        """ @rtype: int """
        return self._get_max_internal_submit()

    def set_max_internal_submit(self, max_value):
        self._get_max_internal_submit(max_value)

    def getForwardModel(self):
        """ @rtype: ForwardModel """
        return self._get_forward_model().setParent(self)

    def getRunpathAsString(self):
        """ @rtype: str """
        return self._get_runpath_as_char()

    def selectRunpath(self, path_key):
        """ @rtype: bool """
        return self._select_runpath(path_key)

    def setRunpath(self, path_format):
        self._set_runpath(path_format)

    def free(self):
        self._free()

    def getGenKWExportName(self):
        """ @rtype: str """
        return self._gen_kw_export_name()

    def runpathRequiresIterations(self):
        """ @rtype: bool """
        return self._runpath_requires_iterations()

    def getJobnameFormat(self):
        """ @rtype: str """
        return self._get_jobname_fmt()

    @property
    def obs_config_file(self):
        return self._get_obs_config_file()

    def getEnspath(self):
        """ @rtype: str """
        return self._get_enspath()

    def getRunpathFormat(self):
        """ @rtype: PathFormat """
        return self._get_runpath_fmt()

    @property
    def num_realizations(self):
        return self._get_num_realizations()

    def data_root(self):
        return self._get_data_root()

    def _set_data_root(self, data_root):
        self._set_data_root(data_root)

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
