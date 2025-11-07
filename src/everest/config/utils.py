from typing import Literal

from ert.config import ExtParamConfig, SamplerConfig


class FlattenedControls:
    def __init__(self, controls: list[ExtParamConfig]) -> None:
        self.names = [name for control in controls for name in control.input_keys]
        self.types: list[Literal["real", "integer"]] = [
            type_ for control in controls for type_ in control.control_types
        ]
        self.initial_guesses = [
            initial_guess
            for control in controls
            for initial_guess in control.initial_guesses
        ]
        self.lower_bounds = [min_ for control in controls for min_ in control.min]
        self.upper_bounds = [max_ for control in controls for max_ in control.max]
        self.scaled_ranges = [
            scaled_range
            for control in controls
            for scaled_range in control.scaled_ranges
        ]
        self.enabled = [enabled for control in controls for enabled in control.enabled]
        self.perturbation_magnitudes = [
            perturbation_magnitude
            for control in controls
            for perturbation_magnitude in control.perturbation_magnitudes
        ]
        self.perturbation_types = [
            perturbation_type
            for control in controls
            for perturbation_type in control.perturbation_types
        ]
        self.samplers, self.sampler_indices = _get_samplers(controls)


def _get_samplers(
    controls: list[ExtParamConfig],
) -> tuple[list[SamplerConfig | None], list[int]]:
    """
    Create a list of unique samplers, and a list mapping variable index
    to sampler index. I.e., this points each control variable to
    a sampler by index.
    """
    flattened_samplers = [
        sampler for control in controls for sampler in control.samplers
    ]
    unique_samplers: list[SamplerConfig | None] = []
    variable_to_unique_sampler_index: list[int] = []
    for sampler in flattened_samplers:
        try:
            unique_sampler_index = next(
                i for i, s in enumerate(unique_samplers) if s == sampler
            )
        except StopIteration:
            unique_sampler_index = len(unique_samplers)
            unique_samplers.append(sampler)

        variable_to_unique_sampler_index.append(unique_sampler_index)

    return unique_samplers, variable_to_unique_sampler_index
