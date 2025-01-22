from collections.abc import Iterator

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
                if isinstance(variable, ControlVariableConfig):
                    var_dict = {
                        key: getattr(variable, key)
                        for key in [
                            "control_type",
                            "enabled",
                            "auto_scale",
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

                    variables.append(var_dict)
                elif isinstance(variable, ControlVariableGuessListConfig):
                    if control.sampler is not None and control_sampler_idx < 0:
                        self._samplers.append(control.sampler)
                        control_sampler_idx = len(self._samplers) - 1
                    variables.extend(
                        {
                            "name": (control.name, variable.name, index + 1),
                            "initial_guess": guess,
                            "sampler_idx": control_sampler_idx,
                        }
                        for index, guess in enumerate(variable.initial_guess)
                    )

            for var_dict in variables:
                for key in [
                    "type",
                    "initial_guess",
                    "control_type",
                    "enabled",
                    "auto_scale",
                    "min",
                    "max",
                    "perturbation_type",
                    "perturbation_magnitude",
                    "scaled_range",
                ]:
                    if var_dict.get(key) is None:
                        var_dict[key] = getattr(control, key)

            self._controls.extend(variables)

        self.names = [control["name"] for control in self._controls]
        self.types = [
            None if control["control_type"] is None else control["control_type"]
            for control in self._controls
        ]
        self.initial_guesses = [control["initial_guess"] for control in self._controls]
        self.lower_bounds = [control["min"] for control in self._controls]
        self.upper_bounds = [control["max"] for control in self._controls]
        self.auto_scales = [control["auto_scale"] for control in self._controls]
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


def control_tuples(
    controls: list[ControlConfig],
) -> Iterator[tuple[str, str, int] | tuple[str, str]]:
    for control in controls:
        for variable in control.variables:
            if isinstance(variable, ControlVariableGuessListConfig):
                for index in range(1, len(variable.initial_guess) + 1):
                    yield (control.name, variable.name, index)
            elif variable.index is not None:
                yield (control.name, variable.name, variable.index)
            else:
                yield (control.name, variable.name)
