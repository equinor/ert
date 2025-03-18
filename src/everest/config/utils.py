from collections.abc import Generator
from typing import Any, Literal

from .control_config import ControlConfig
from .control_variable_config import (
    ControlVariableConfig,
    ControlVariableGuessListConfig,
)
from .sampler_config import SamplerConfig


class FlattenedControls:
    def __init__(self, controls: list[ControlConfig]) -> None:
        self._controls = []
        self._samplers: list[SamplerConfig] = []

        for control in controls:
            control_sampler_idx = -1
            variables = []
            for variable in control.variables:
                match variable:
                    case ControlVariableConfig():
                        var_dict, control_sampler_idx = self._add_variable(
                            control, variable, control_sampler_idx
                        )
                        variables.append(var_dict)
                    case ControlVariableGuessListConfig():
                        var_dicts, control_sampler_idx = self._add_variable_guess_list(
                            control, variable, control_sampler_idx
                        )
                        variables.extend(var_dicts)
            self._inject_defaults(control, variables)
            self._controls.extend(variables)

        self.names = [control["name"] for control in self._controls]
        self.types: list[Literal["real", "integer"]] = [
            "real" if control["control_type"] is None else control["control_type"]
            for control in self._controls
        ]
        self.initial_guesses = [control["initial_guess"] for control in self._controls]
        self.lower_bounds = [control["min"] for control in self._controls]
        self.upper_bounds = [control["max"] for control in self._controls]
        self.scaled_ranges = [
            (0.0, 1.0) if control["scaled_range"] is None else control["scaled_range"]
            for control in self._controls
        ]
        self.enabled = [control["enabled"] for control in self._controls]
        self.perturbation_magnitudes = [
            control["perturbation_magnitude"] for control in self._controls
        ]
        self.perturbation_types = [
            control["perturbation_type"] for control in self._controls
        ]
        self.sampler_indices = [control["sampler_idx"] for control in self._controls]
        self.samplers = self._samplers

    def _add_variable(
        self,
        control: ControlConfig,
        variable: ControlVariableConfig,
        control_sampler_idx: int,
    ) -> tuple[dict[str, Any], int]:
        var_dict = {
            key: getattr(variable, key)
            for key in [
                "control_type",
                "enabled",
                "scaled_range",
                "min",
                "max",
                "perturbation_magnitude",
                "initial_guess",
            ]
        }
        var_dict["name"] = (
            (control.name, variable.name)
            if variable.index is None
            else (control.name, variable.name, variable.index)
        )
        if variable.sampler is not None:
            self._samplers.append(variable.sampler)
            var_dict["sampler_idx"] = len(self._samplers) - 1
        else:
            if control.sampler is not None and control_sampler_idx < 0:
                self._samplers.append(control.sampler)
                control_sampler_idx = len(self._samplers) - 1
            var_dict["sampler_idx"] = control_sampler_idx
        return var_dict, control_sampler_idx

    def _add_variable_guess_list(
        self,
        control: ControlConfig,
        variable: ControlVariableGuessListConfig,
        control_sampler_idx: int,
    ) -> tuple[Generator[dict[str, Any], None, None], int]:
        if control.sampler is not None and control_sampler_idx < 0:
            self._samplers.append(control.sampler)
            control_sampler_idx = len(self._samplers) - 1
        return (
            (
                {
                    "name": (control.name, variable.name, index + 1),
                    "initial_guess": guess,
                    "sampler_idx": control_sampler_idx,
                }
                for index, guess in enumerate(variable.initial_guess)
            ),
            control_sampler_idx,
        )

    @staticmethod
    def _inject_defaults(
        control: ControlConfig, variables: list[dict[str, Any]]
    ) -> None:
        for var_dict in variables:
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
