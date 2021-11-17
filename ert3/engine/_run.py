import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import ert
import ert3
from ert3.config import SourceNS
from ert_shared.async_utils import get_event_loop
from ert_shared.ensemble_evaluator.ensemble.builder import create_step_builder

from ._entity import TransmitterCoroutine
from ._sensitivity import (
    analyze_sensitivity,
    prepare_sensitivity,
    transmitter_map_sensitivity,
)

logger = logging.getLogger(__name__)


def _prepare_experiment(
    workspace_name: str,
    experiment_name: str,
    ensemble: ert3.config.EnsembleConfig,
    ensemble_size: int,
) -> None:
    if experiment_name in ert.storage.get_experiment_names(
        workspace_name=workspace_name
    ):
        raise ValueError(f"Experiment {experiment_name} has been carried out.")

    parameters = [elem.record for elem in ensemble.input]
    responses = [elem.record for elem in ensemble.output]
    ert.storage.init_experiment(
        experiment_name=experiment_name,
        parameters=parameters,
        ensemble_size=ensemble_size,
        responses=responses,
    )


def _gather_transmitter_maps(
    futures: List[TransmitterCoroutine],
) -> Dict[int, Dict[str, ert.data.RecordTransmitter]]:
    map_: Dict[int, Dict[str, ert.data.RecordTransmitter]] = {}
    res = get_event_loop().run_until_complete(asyncio.gather(*futures))
    for iens_to_trans_map in res:
        for iens, trans_map in iens_to_trans_map.items():
            if iens not in map_:
                map_[iens] = trans_map
            else:
                map_[iens].update(trans_map)
    return map_


def _transmitter_map_storage(
    inputs: Tuple[ert3.config.LinkedInput, ...],
    ensemble_size: int,
    records_url: str,
) -> List[TransmitterCoroutine]:
    futures: List[TransmitterCoroutine] = []
    for input_ in inputs:
        future = ert.storage.get_record_storage_transmitters(
            records_url=records_url,
            record_name=input_.name,
            record_source=input_.source_location,
            ensemble_size=ensemble_size,
        )
        futures.append(future)
    return futures


def _transmitter_map_resources(
    experiment_run_config: ert3.config.ExperimentRunConfig,
    inputs: Tuple[ert3.config.LinkedInput, ...],
    ensemble_size: int,
    experiment_name: str,
    workspace: ert3.workspace.Workspace,
) -> List[TransmitterCoroutine]:
    futures: List[TransmitterCoroutine] = []
    for input_ in inputs:
        collection_awaitable = workspace.load_resource(
            experiment_run_config, input_, ensemble_size=ensemble_size
        )
        future = ert.storage.transmit_awaitable_record_collection(
            record_awaitable=collection_awaitable,
            record_name=input_.name,
            workspace_name=workspace.name,
            experiment_name=experiment_name,
        )
        futures.append(future)

    return futures


def _transmitter_map_stochastic(
    inputs: Tuple[ert3.config.LinkedInput, ...],
    parameters_config: ert3.config.ParametersConfig,
    ensemble_size: int,
    experiment_name: str,
    workspace_name: str,
) -> List[TransmitterCoroutine]:
    futures: List[TransmitterCoroutine] = []
    for input_ in inputs:
        collection = ert3.engine.sample_record(
            parameters_config,
            input_.source_location,
            ensemble_size=ensemble_size,
        )
        future = ert.storage.transmit_record_collection(
            record_coll=collection,
            record_name=input_.name,
            workspace_name=workspace_name,
            experiment_name=experiment_name,
        )
        futures.append(future)
    return futures


def _get_storage_path(
    ensemble_config: ert3.config.EnsembleConfig,
    workspace: ert3.workspace.Workspace,
    experiment_name: str,
) -> str:
    if ensemble_config.storage_type == "ert_storage":
        return ert.storage.get_records_url(workspace.name, experiment_name)
    else:
        storage_tmp_dir = str(Path(tempfile.mkdtemp()) / experiment_name / "storage")
        logger.info(  # pylint: disable=logging-fstring-interpolation
            f"temporary storage location: {storage_tmp_dir}"
        )
        return storage_tmp_dir


# pylint: disable=too-many-arguments
def run(
    experiment_run_config: ert3.config.ExperimentRunConfig,
    workspace: ert3.workspace.Workspace,
    experiment_name: str,
) -> None:
    # This reassures mypy that the ensemble size is defined
    assert experiment_run_config.ensemble_config.size is not None
    ensemble_size = experiment_run_config.ensemble_config.size

    if experiment_run_config.experiment_config.type != "evaluation":
        raise ValueError("this entry point can only run 'evaluation' experiments")

    _prepare_experiment(
        workspace.name,
        experiment_name,
        experiment_run_config.ensemble_config,
        ensemble_size,
    )
    storage_path = _get_storage_path(
        experiment_run_config.ensemble_config, workspace, experiment_name
    )
    records_url = ert.storage.get_records_url(workspace.name)

    stage = experiment_run_config.get_stage()
    step_builder = (
        create_step_builder()
        .set_name(f"{stage.name}-only_step")
        .set_type("function" if isinstance(stage, ert3.config.Function) else "unix")
    )

    inputs = experiment_run_config.get_linked_inputs()

    storage_inputs = tuple(inputs[SourceNS.storage].values())
    resource_inputs = tuple(inputs[SourceNS.resources].values())
    stochastic_inputs = tuple(inputs[SourceNS.stochastic].values())
    transmitters = _gather_transmitter_maps(
        _transmitter_map_storage(storage_inputs, ensemble_size, records_url)
        + _transmitter_map_resources(
            experiment_run_config,
            resource_inputs,
            ensemble_size,
            experiment_name,
            workspace,
        )
        + _transmitter_map_stochastic(
            stochastic_inputs,
            experiment_run_config.parameters_config,
            ensemble_size,
            experiment_name,
            workspace.name,
        )
    )

    for records in (storage_inputs, resource_inputs, stochastic_inputs):
        ert3.evaluator.add_step_inputs(
            records,
            transmitters,
            step_builder,
        )

    ert3.evaluator.add_step_outputs(
        experiment_run_config.ensemble_config.storage_type,
        stage,
        storage_path,
        ensemble_size,
        step_builder,
    )

    if isinstance(stage, ert3.config.Unix):
        ert3.evaluator.add_commands(
            stage.transportable_commands,
            experiment_run_config.ensemble_config.storage_type,
            storage_path,
            step_builder,
        )

    ensemble = ert3.evaluator.build_ensemble(
        stage,
        experiment_run_config.ensemble_config.forward_model.driver,
        ensemble_size,
        step_builder,
    )
    ert3.evaluator.evaluate(ensemble)


# pylint: disable=too-many-arguments
def run_sensitivity_analysis(
    experiment_run_config: ert3.config.ExperimentRunConfig,
    workspace: ert3.workspace.Workspace,
    experiment_name: str,
) -> None:
    inputs = experiment_run_config.get_linked_inputs()
    storage_inputs = tuple(inputs[SourceNS.storage].values())
    resource_inputs = tuple(inputs[SourceNS.resources].values())
    stochastic_inputs = tuple(inputs[SourceNS.stochastic].values())
    sensitivity_input_records = prepare_sensitivity(
        stochastic_inputs,
        experiment_run_config.experiment_config,
        experiment_run_config.parameters_config,
    )
    ensemble_size = len(sensitivity_input_records)

    _prepare_experiment(
        workspace.name,
        experiment_name,
        experiment_run_config.ensemble_config,
        ensemble_size,
    )

    storage_path = _get_storage_path(
        experiment_run_config.ensemble_config, workspace, experiment_name
    )
    records_url = ert.storage.get_records_url(workspace.name)

    stage = experiment_run_config.get_stage()
    step_builder = (
        create_step_builder()
        .set_name(f"{stage.name}-only_step")
        .set_type("function" if isinstance(stage, ert3.config.Function) else "unix")
    )

    transmitters = _gather_transmitter_maps(
        _transmitter_map_storage(storage_inputs, ensemble_size, records_url)
        + _transmitter_map_resources(
            experiment_run_config,
            resource_inputs,
            ensemble_size,
            experiment_name,
            workspace,
        )
        + transmitter_map_sensitivity(
            stochastic_inputs,
            sensitivity_input_records,
            experiment_name,
            workspace,
        )
    )
    for records in (storage_inputs, resource_inputs, stochastic_inputs):
        ert3.evaluator.add_step_inputs(
            records,
            transmitters,
            step_builder,
        )

    ert3.evaluator.add_step_outputs(
        experiment_run_config.ensemble_config.storage_type,
        stage,
        storage_path,
        ensemble_size,
        step_builder,
    )

    if isinstance(stage, ert3.config.Unix):
        ert3.evaluator.add_commands(
            stage.transportable_commands,
            experiment_run_config.ensemble_config.storage_type,
            storage_path,
            step_builder,
        )

    ensemble = ert3.evaluator.build_ensemble(
        stage,
        experiment_run_config.ensemble_config.forward_model.driver,
        ensemble_size,
        step_builder,
    )

    output_transmitters = ert3.evaluator.evaluate(ensemble)
    analyze_sensitivity(
        stochastic_inputs,
        experiment_run_config.experiment_config,
        experiment_run_config.parameters_config,
        workspace,
        experiment_name,
        output_transmitters,
    )


def get_ensemble_size(
    experiment_run_config: ert3.config.ExperimentRunConfig,
) -> int:
    if experiment_run_config.experiment_config.type == "sensitivity":
        stochastic_inputs = tuple(
            experiment_run_config.get_linked_inputs()[SourceNS.stochastic].values()
        )
        return len(
            prepare_sensitivity(
                stochastic_inputs,
                experiment_run_config.experiment_config,
                experiment_run_config.parameters_config,
            )
        )
    else:
        assert experiment_run_config.ensemble_config.size is not None
        return experiment_run_config.ensemble_config.size
