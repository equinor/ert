#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'enkf_state.py' is part of ERT - Ensemble based Reservoir Tool.
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
from typing import Optional

from cwrap import BaseCClass
from ecl.util.util import StringList

from res import ResPrototype
from res.enkf.enkf_fs import EnkfFs
from res.enkf.enums import EnkfInitModeEnum, EnkfVarType
from res.enkf.res_config import ResConfig
from res.enkf.run_arg import RunArg


class EnKFState(BaseCClass):
    TYPE_NAME = "enkf_state"
    _free = ResPrototype("void enkf_state_free( enkf_state )")
    _get_ens_config = ResPrototype(
        "ens_config_ref enkf_state_get_ensemble_config( enkf_state )"
    )
    _initialize = ResPrototype(
        "void enkf_state_initialize( enkf_state , enkf_fs , stringlist , enkf_init_mode_enum)"
    )
    _forward_model_OK = ResPrototype(
        "bool enkf_state_complete_forward_modelOK(res_config, run_arg)", bind=False
    )
    _forward_model_EXIT = ResPrototype(
        "bool enkf_state_complete_forward_model_EXIT_handler__(run_arg)", bind=False
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def free(self):
        self._free()

    def ensembleConfig(self):
        """@rtype: EnsembleConfig"""
        return self._get_ens_config()

    def initialize(
        self,
        fs: EnkfFs,
        param_list: Optional[StringList] = None,
        init_mode=EnkfInitModeEnum.INIT_CONDITIONAL,
    ):
        if param_list is None:
            ens_config = self.ensembleConfig()
            param_list = ens_config.getKeylistFromVarType(EnkfVarType.PARAMETER)
        self._initialize(fs, param_list, init_mode)

    @classmethod
    def forward_model_exit_callback(cls, args):
        if not isinstance(args[0], RunArg):
            raise ValueError("Expected RunArg as second argument")
        return cls._forward_model_EXIT(args[0])

    @classmethod
    def forward_model_ok_callback(cls, args):
        if not isinstance(args[1], ResConfig):
            raise ValueError("Expected ResConfig as second argument")
        if not isinstance(args[0], RunArg):
            raise ValueError("Expected RunArg as second argument")
        return cls._forward_model_OK(args[1], args[0])
