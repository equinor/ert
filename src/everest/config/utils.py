from copy import deepcopy
from typing import Any, Literal

from .control_config import ControlConfig
from .control_variable_config import ControlVariableGuessListConfig
from .sampler_config import SamplerConfig


class FlattenedControls:
    def __init__(self, controls: list[ControlConfig]) -> None:
        control_dicts = _get_control_dicts(controls)
        self.names = [control["name"] for control in control_dicts]
        self.types: list[Literal["real", "integer"]] = [
            control["control_type"] for control in control_dicts
        ]
        self.initial_guesses = [control["initial_guess"] for control in control_dicts]
        self.lower_bounds = [control["min"] for control in control_dicts]
        self.upper_bounds = [control["max"] for control in control_dicts]
        self.scaled_ranges = [control["scaled_range"] for control in control_dicts]
        self.enabled = [control["enabled"] for control in control_dicts]
        self.perturbation_magnitudes = [
            control["perturbation_magnitude"] for control in control_dicts
        ]
        self.perturbation_types = [
            control["perturbation_type"] for control in control_dicts
        ]
        self.samplers, self.sampler_indices = _get_samplers(controls)


def _get_control_dicts(controls: list[ControlConfig]) -> list[dict[str, Any]]:
    def _inject_defaults(control: ControlConfig, var_dict: dict[str, Any]) -> None:
        for key in [
            "type",
            "initial_guess",
            "control_type",
            "enabled",
            "min",
            "max",
            "perturbation_type",
            "perturbation_magnitude",
            "scaled_range",
        ]:
            if var_dict.get(key) is None:
                var_dict[key] = getattr(control, key)

    control_dicts: list[dict[str, Any]] = []
    for control in controls:
        for variable in control.variables:
            if isinstance(variable, ControlVariableGuessListConfig):
                for index, guess in enumerate(variable.initial_guess, start=1):
                    var_dict = deepcopy(variable.model_dump())
                    var_dict["name"] = (control.name, variable.name, index)
                    var_dict["initial_guess"] = guess
                    _inject_defaults(control, var_dict)
                    control_dicts.append(var_dict)
            else:
                var_dict = deepcopy(variable.model_dump())
                var_dict["name"] = (
                    (control.name, variable.name)
                    if variable.index is None
                    else (control.name, variable.name, variable.index)
                )
                _inject_defaults(control, var_dict)
                control_dicts.append(var_dict)
    return control_dicts


def _get_samplers(
    controls: list[ControlConfig],
) -> tuple[list[SamplerConfig | None], list[int]]:
    samplers: list[SamplerConfig | None] = []
    sampler_indices: list[int] = []
    for control in controls:
        control_sampler_idx: int | None = None
        for variable in control.variables:
            if variable.sampler is None:
                if control_sampler_idx is None:
                    samplers.append(control.sampler)
                    control_sampler_idx = len(samplers) - 1
                sampler_index = control_sampler_idx
            else:
                samplers.append(variable.sampler)
                sampler_index = len(samplers) - 1
            if isinstance(variable, ControlVariableGuessListConfig):
                sampler_indices.extend([sampler_index] * len(variable.initial_guess))
            else:
                sampler_indices.append(sampler_index)
    return samplers, sampler_indices


def flatten_controls(controls: list[ControlConfig]) -> FlattenedControls:
    return FlattenedControls(controls)
