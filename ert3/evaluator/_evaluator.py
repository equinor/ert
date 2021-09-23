import asyncio
import copy
import pathlib
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import cloudpickle
from pydantic import FilePath

import ert
import ert3
from ert3.config import EnsembleConfig, StagesConfig, Step
from ert.data import RecordTransmitter
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.ensemble.base import _Ensemble
from ert_shared.ensemble_evaluator.ensemble.builder import (
    create_ensemble_builder,
    create_file_io_builder,
    create_job_builder,
    create_realization_builder,
    create_step_builder,
)
from ert_shared.ensemble_evaluator.entity.identifiers import EVTYPE_EE_TERMINATED
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator

_EVTYPE_SNAPSHOT_STOPPED = "Stopped"
_EVTYPE_SNAPSHOT_FAILED = "Failed"


def _create_evaluator_tmp_dir(workspace_root: Path, evaluation_name: str) -> Path:
    return (
        pathlib.Path(workspace_root)
        / ert3._WORKSPACE_DATA_ROOT
        / "tmp"
        / evaluation_name
    )


def _prepare_input(
    storage_type: str,
    step_config: Step,
    parameters: Dict[int, Dict[str, ert.data.RecordTransmitter]],
    storage_path: str,
    ensemble_size: int,
) -> Dict[int, Dict[str, RecordTransmitter]]:
    transmitters = copy.deepcopy(parameters)
    if isinstance(step_config, ert3.config.Unix):
        transmitter: ert.data.RecordTransmitter
        command_futures = []
        for command in step_config.transportable_commands:
            if storage_type == "shared_disk":
                transmitter = ert.data.SharedDiskRecordTransmitter(
                    name=command.name,
                    storage_path=pathlib.Path(storage_path),
                )
            elif storage_type == "ert_storage":
                transmitter = ert.storage.StorageRecordTransmitter(
                    name=command.name, storage_url=storage_path
                )
            else:
                raise ValueError(f"Unsupported transmitter type: {storage_type}")
            command_futures.append(
                transmitter.transmit_file(
                    command.location, mime="application/octet-stream"
                )
            )
            for iens in range(ensemble_size):
                transmitters[iens][command.name] = transmitter
        asyncio.get_event_loop().run_until_complete(asyncio.gather(*command_futures))
    return transmitters


def _prepare_output(
    storage_type: str,
    step_config: Step,
    storage_path: str,
    ensemble_size: int,
) -> Dict[int, Dict[str, RecordTransmitter]]:
    transmitters: Dict[int, Dict[str, ert.data.RecordTransmitter]] = defaultdict(dict)

    for record_name in step_config.output.keys():
        for iens in range(0, ensemble_size):
            if storage_type == "shared_disk":
                transmitters[iens][record_name] = ert.data.SharedDiskRecordTransmitter(
                    name=record_name,
                    storage_path=pathlib.Path(storage_path),
                )
            elif storage_type == "ert_storage":
                transmitters[iens][record_name] = ert.storage.StorageRecordTransmitter(
                    name=record_name, storage_url=storage_path, iens=iens
                )
            else:
                raise ValueError(f"Unsupported transmitter type: {storage_type}")
    return dict(transmitters)


def _build_ensemble(
    storage_path: str,
    ensemble: EnsembleConfig,
    stages_config: StagesConfig,
    parameters: Dict[int, Dict[str, ert.data.RecordTransmitter]],
) -> _Ensemble:
    if ensemble.size is not None:
        ensemble_size = ensemble.size
    else:
        ensemble_size = len(parameters)

    stage = stages_config.step_from_key(ensemble.forward_model.stage)
    assert stage is not None
    commands = (
        stage.transportable_commands if isinstance(stage, ert3.config.Unix) else []
    )

    def command_location(name: str) -> FilePath:
        return next(
            (cmd.location for cmd in commands if cmd.name == name), pathlib.Path(name)
        )

    step_builder = (
        create_step_builder()
        .set_name(f"{stage.name}-only_step")
        .set_type("function" if isinstance(stage, ert3.config.Function) else "unix")
    )

    for output in stage.output.values():
        step_builder.add_output(
            create_file_io_builder()
            .set_name(output.record)
            .set_path(Path(output.location))
            .set_mime(output.mime)
        )

    for input_ in stage.input.values():
        step_builder.add_input(
            create_file_io_builder()
            .set_name(input_.record)
            .set_path(Path(input_.location))
            .set_mime(input_.mime)
        )

    for cmd in commands:
        step_builder.add_input(
            create_file_io_builder()
            .set_name(cmd.name)
            .set_path(command_location(cmd.name))
            .set_mime(cmd.mime)
            .set_executable()
        )

    if isinstance(stage, ert3.config.Function):
        step_builder.add_job(
            create_job_builder()
            .set_name(stage.function.__name__)
            .set_executable(cloudpickle.dumps(stage.function))
        )
    if isinstance(stage, ert3.config.Unix):
        for script in stage.script:
            name, *args = script.split()
            step_builder.add_job(
                create_job_builder()
                .set_name(name)
                .set_executable(command_location(name))
                .set_args(tuple(args))
            )

    builder = (
        create_ensemble_builder()
        .set_ensemble_size(ensemble_size)
        .set_max_running(10000)
        .set_max_retries(0)
        .set_executor(ensemble.forward_model.driver)
        .set_forward_model(
            create_realization_builder().active(True).add_step(step_builder)
        )
    )

    inputs = _prepare_input(
        ensemble.storage_type, stage, parameters, storage_path, ensemble_size
    )
    outputs = _prepare_output(ensemble.storage_type, stage, storage_path, ensemble_size)
    builder.set_inputs(inputs).set_outputs(outputs)

    return builder.build()


def _run(
    ensemble_evaluator: EnsembleEvaluator,
) -> Dict[int, Dict[str, RecordTransmitter]]:
    result: Dict[int, Dict[str, RecordTransmitter]] = {}
    with ensemble_evaluator.run() as monitor:
        for event in monitor.track():
            if isinstance(event.data, dict) and event.data.get("status") in [
                _EVTYPE_SNAPSHOT_STOPPED,
                _EVTYPE_SNAPSHOT_FAILED,
            ]:
                monitor.signal_done()
            if event["type"] == EVTYPE_EE_TERMINATED and isinstance(event.data, bytes):
                result = pickle.loads(event.data)

    return result


def evaluate(
    storage_path: str,
    parameters: Dict[int, Dict[str, ert.data.RecordTransmitter]],
    ensemble_config: EnsembleConfig,
    stages_config: StagesConfig,
) -> Dict[int, Dict[str, RecordTransmitter]]:

    ensemble = _build_ensemble(
        storage_path,
        ensemble_config,
        stages_config,
        parameters,
    )

    config = EvaluatorServerConfig()
    ee = EnsembleEvaluator(ensemble=ensemble, config=config, iter_=0)
    result = _run(ee)
    return result


def cleanup(workspace_root: Path, evaluation_name: str) -> None:
    tmp_dir = _create_evaluator_tmp_dir(workspace_root, evaluation_name)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
