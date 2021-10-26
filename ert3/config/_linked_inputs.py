from typing import Dict, NamedTuple

from ._stages_config import Step
from ._ensemble_config import EnsembleConfig, SourceNS


class LinkedInput(NamedTuple):
    """Provide a view of an input linked together from multiple configuration
    sources.
    """

    name: str
    source_mime: str
    source_namespace: SourceNS
    source_location: str
    source_is_directory: bool
    dest_location: str
    dest_mime: str
    dest_is_directory: bool


def link_inputs(
    ensemble_config: EnsembleConfig, step_config: Step
) -> Dict[SourceNS, Dict[str, LinkedInput]]:
    """Interpret multiple configurations to create one uniform input type broken
    down into all available namespaces.
    """
    inputs: Dict[SourceNS, Dict[str, LinkedInput]] = {
        SourceNS.storage: {},
        SourceNS.resources: {},
        SourceNS.stochastic: {},
    }
    for ensemble_input in ensemble_config.input:
        name = ensemble_input.record
        stage_is_directory = step_config.input[name].is_directory
        stage_mime = step_config.input[name].mime
        stage_location = step_config.input[name].location

        if stage_mime != ensemble_input.mime:
            print(
                f"Warning: Conflicting ensemble mime '{ensemble_input.mime}' and "
                + f"stage mime '{stage_mime}' for input '{name}'."
            )

        # fall back on stage is_directory
        ensemble_is_directory = (
            ensemble_input.is_directory
            if ensemble_input.is_directory is not None
            else stage_is_directory
        )

        input_ = LinkedInput(
            name=name,
            source_namespace=ensemble_input.source_namespace,
            source_mime=ensemble_input.mime,
            source_is_directory=ensemble_is_directory,
            source_location=ensemble_input.source_location,
            dest_mime=stage_mime,
            dest_location=stage_location,
            dest_is_directory=stage_is_directory,
        )
        inputs[input_.source_namespace][input_.name] = input_
    return inputs
