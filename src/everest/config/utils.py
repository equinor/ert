from ert.config import EverestControl, SamplerConfig


def get_samplers(
    controls: list[EverestControl],
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
