#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'local_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from dataclasses import dataclass, field
from typing import List, Union, Tuple

from deprecation import deprecated
from res._lib.local.row_scaling import RowScaling

from res.enkf.local_obsdata import LocalObsdata
from res.enkf.local_ministep import LocalMinistep


@dataclass
class LocalConfig:
    ministeps: List[LocalMinistep] = field(default_factory=list)

    @deprecated(
        "Create a new LocalConfig instead",
    )
    def clear(self):
        self.ministeps = []

    def clear_active(self):
        raise NotImplementedError("This functionality has been removed, use: clear")

    @deprecated("Use add_ministep instead")
    def createMinistep(self, mini_step_key, _=None):
        """@rtype: Ministep"""
        if mini_step_key in [ministep.name() for ministep in self.ministeps]:
            raise KeyError(f"Ministep: {mini_step_key} already in local_config")
        return self.add_ministep(mini_step_key)

    @deprecated(
        "Add observations to ministep directly",
    )
    def createObsdata(self, obsdata_key):
        """@rtype: Obsdata"""
        return LocalObsdata(obsdata_key)

    @deprecated(
        "Add observations to ministep directly",
    )
    def copyObsdata(self, _, new_key):
        return LocalObsdata(new_key)

    @deprecated(
        "Loop over configuration directly",
    )
    def getUpdatestep(self):
        """@rtype: UpdateStep"""
        return self

    @deprecated(
        "Add observations to ministep directly",
    )
    def getMinistep(self, key):
        """@rtype: Ministep"""
        for ministep in self.ministeps:
            if ministep.name() == key:
                return ministep
        raise KeyError(f"No ministep named: {key}")

    @deprecated(
        "Add observations to ministep directly",
    )
    def getObsdata(self, obsdata_key):
        """@rtype: Obsdata"""
        return obsdata_key

    @deprecated("Use add_ministep")
    def attachMinistep(self, mini_step):
        if mini_step.name not in [ministep.name for ministep in self.ministeps]:
            self.ministeps.append(mini_step)

    def add_ministep(
        self,
        name,
        observations: List[Union[str, Tuple[str, List[int]]]],
        parameters: List[
            Union[str, Tuple[str, List[int]], Tuple[str, RowScaling, List[int]]]
        ],
    ):
        ministep = LocalMinistep(name)
        for observation in observations:
            if isinstance(observation, str):
                observation = [observation]
            ministep.add_observation(*observation)
        for parameter in parameters:
            if isinstance(parameter, str):
                parameter = [parameter]
            if len(parameter) > 1 and isinstance(parameter[1], RowScaling):
                ministep.add_row_scaling_parameter(*parameter)
            else:
                ministep.add_parameter(*parameter)

        self.ministeps.append(ministep)

    def __iter__(self):
        yield from self.ministeps

    def __len__(self):
        return len(self.ministeps)

    def __getitem__(self, item):
        return self.ministeps[item]

    def context_validate(
        self, valid_observations: List[str], valid_parameters: List[str]
    ) -> bool:
        errors = []
        for ministep in self.ministeps:
            for observation in ministep.observations:
                if observation not in valid_observations:
                    errors.append(
                        f"Observation: {observation} not in valid observations"
                    )
            for parameter in list(ministep.parameters.keys()) + list(
                ministep.row_scaling_parameters.keys()
            ):
                if parameter not in valid_parameters:
                    errors.append(f"Parameter: {parameter} not in valid parameters")
        if errors:
            raise ValueError(
                f"Update configuration not valid, "
                f"valid observations: {valid_observations}, "
                f"valid parameters: {valid_parameters}, errors: {errors}"
            )
