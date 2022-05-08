from dataclasses import field, dataclass
from typing import Dict, List

from deprecation import deprecated

from res.enkf.row_scaling import RowScaling
from res.enkf.local_obsdata import LocalObsdata
from res._lib.update import RowScalingParameter, Parameter


@dataclass
class Observation:
    name: str
    active_index: List[int] = field(default_factory=list)

    def key(self):
        return self.name


@dataclass
class LocalMinistep:
    _name: str
    observations: Dict[str, Observation] = field(default_factory=dict)
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    row_scaling_parameters: Dict[str, RowScalingParameter] = field(default_factory=dict)

    @deprecated("Will be removed or renamed")
    def hasActiveData(self, key):
        return key in self.parameters or key in self.row_scaling_parameters

    @deprecated(
        "Will be removed, use: add_parameter",
    )
    def addActiveData(self, key):
        self.add_parameter(key)

    @deprecated("Will be removed")
    def getActiveList(self, key):
        """@rtype: ActiveList"""
        if key in self.parameters:
            return self.parameters[key].active_list
        elif key in self.row_scaling_parameters:
            return self.row_scaling_parameters[key].active_list
        else:
            raise KeyError(f'Local key "{key}" not recognized.')

    @deprecated("Will be removed")
    def numActiveData(self):
        return len(self.parameters) + len(self.row_scaling_parameters)

    @deprecated("Will be removed")
    def addNode(self, node):
        raise NotImplementedError("Adding node to ministep is not supported")

    @deprecated(
        "Will be removed, use: add_observation",
    )
    def attachObsset(self, obs_set):
        assert isinstance(obs_set, LocalObsdata)
        for node in obs_set:
            self.observations[node.key()] = Observation(
                node.key(), node.getActiveList().get_active_index_list()
            )

    @deprecated(
        "Will be removed, use: add_row_scaling and get the row_scaling property",
    )
    def get_or_create_row_scaling(self, name):
        return self.row_scaling(name)

    @deprecated(
        "Will be removed, use: add_row_scaling",
    )
    def row_scaling(self, parameter_name) -> RowScaling:
        if parameter_name in self.row_scaling_parameters:
            return self.row_scaling_parameters[parameter_name].row_scaling
        elif parameter_name in self.parameters:
            parameter = self.parameters[parameter_name]
            row_scaling = RowScaling()
            self.row_scaling_parameters[parameter_name] = RowScalingParameter(
                parameter.name, row_scaling, parameter.index_list
            )
            # We have converted a parameter -> row_scaling_parameter, so remove it from
            # parameters
            del self.parameters[parameter_name]
            return row_scaling
        raise KeyError(f"No such parameter: {parameter_name}")

    @deprecated("Use property: observations")
    def getLocalObsData(self):
        """@rtype: LocalObsdata"""
        return self.observations.values()

    @deprecated(
        "Will be turned into property: name",
    )
    def name(self):
        return self._name

    @deprecated(
        "Will be removed, and turned into property: name",
    )
    def getName(self):
        """@rtype: str"""
        return self.name()

    def observation_config(self):
        return [(key, value.active_index) for key, value in self.observations.items()]

    def parameter_config(self):
        return list(self.parameters.values())

    def row_scaling_config(self):
        return list(self.row_scaling_parameters.values())

    def add_observation(self, name, index_list=None):
        index_list = [] if index_list is None else index_list
        self.observations[name] = Observation(name, index_list)

    def add_parameter(self, name, index_list=None):
        index_list = [] if index_list is None else index_list
        self.parameters[name] = Parameter(name, index_list)

    def add_row_scaling_parameter(self, name, row_scaling: RowScaling, index_list=None):
        index_list = [] if index_list is None else index_list
        self.parameters[name] = RowScalingParameter(name, row_scaling, index_list)
