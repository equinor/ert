import json
import os
import pathlib
import shutil

from prefect.engine import result

from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.prefect_ensemble.prefect_ensemble import (
    PrefectEnsemble,
    storage_driver_factory,
)

from collections import defaultdict
from graphlib import TopologicalSorter

import ert3
from ert3.config._stages_config import StagesConfig
import typing


_EVTYPE_SNAPSHOT_STOPPED = "Stopped"
_EVTYPE_SNAPSHOT_FAILED = "Failed"


def _create_evaluator_tmp_dir(workspace_root, evaluation_name):
    return (
        pathlib.Path(workspace_root)
        / ert3._WORKSPACE_DATA_ROOT
        / "tmp"
        / evaluation_name
    )


def _assert_single_stage_forward_model(stages_config, ensemble):
    # The current implementation only support one stage as the forward model
    # and hence we fail if multiple are provided
    assert len(ensemble.forward_model.stages) == 1


def _prepare_input(
    ee_config,
    step_config: ert3.config._stages_config.Step,
    inputs: "ert3.data.MultiEnsembleRecord",
    evaluation_tmp_dir,
    ensemble_size,
) -> typing.Dict[str, typing.Dict[str, typing.List["ert3.data.RecordTransmitter"]]]:
    tmp_input_folder = evaluation_tmp_dir / "prep_input_files"
    os.makedirs(tmp_input_folder)
    ee_storage = storage_driver_factory(ee_config["storage"], tmp_input_folder)
    transmitters = defaultdict(dict)

    for input_ in step_config.input:
        for iens, record in enumerate(inputs.ensemble_records[input_.record].records):
            transmitter = ert3.data.PrefectStorageRecordTransmitter(
                input_.record, ee_storage
            )
            transmitter.transmit(record.data)
            transmitters[iens][input_.record] = transmitter
    for command in step_config.transportable_commands:
        for iens in range(0, ensemble_size):
            transmitter = ert3.data.PrefectStorageRecordTransmitter(
                command.name, ee_storage
            )
            with open(command.location, "rb") as f:
                transmitter.transmit([f.read()], mime=command.mime)
            transmitters[iens][command.name] = transmitter
    return dict(transmitters)


def _prepare_output(
    ee_config,
    step_config: ert3.config._stages_config.Step,
    evaluation_tmp_dir: pathlib.Path,
    ensemble_size: int,
) -> typing.Dict[str, typing.Dict[str, typing.List["ert3.data.RecordTransmitter"]]]:
    # TODO: ensemble_size should rather be a list of ensemble ids
    tmp_input_folder = evaluation_tmp_dir / "output_files"
    os.makedirs(tmp_input_folder)
    ee_storage = storage_driver_factory(ee_config["storage"], tmp_input_folder)
    transmitters = defaultdict(dict)
    for output in step_config.output:
        for iens in range(0, ensemble_size):
            transmitters[iens][
                output.record
            ] = ert3.data.PrefectStorageRecordTransmitter(output.record, ee_storage)
    return dict(transmitters)


def _build_ee_config(
    evaluation_tmp_dir,
    ensemble,
    stages_config: StagesConfig,
    input_records: "ert3.data.MultiEnsembleRecord",
    dispatch_uri: str,
):
    _assert_single_stage_forward_model(stages_config, ensemble)

    if ensemble.size != None:
        ensemble_size = ensemble.size
    else:
        ensemble_size = input_records.ensemble_size

    stage_name = ensemble.forward_model.stages[0]
    stage = stages_config.step_from_key(stage_name)
    commands = stage.transportable_commands
    output_locations = [out.location for out in stage.output]
    jobs = []

    def command_location(name):
        return next(
            cmd.location for cmd in stage.transportable_commands if cmd.name == name
        )

    if stage.function:
        jobs.append(
            {
                "name": stage.function.__name__,
                "executable": stage.function,
                "output": output_locations[0],
            }
        )

    for script in stage.script:
        name, *args = script.split()
        jobs.append(
            {
                "name": name,
                "executable": command_location(name),
                "args": tuple(args),
            }
        )

    stages = [
        {
            "name": stage.name,
            "steps": [
                {
                    "name": stage.name + "-only_step",
                    "inputs": [
                        {
                            "record": input_.record,
                            "location": input_.location,
                            "mime": input_.mime,
                        }
                        for input_ in stage.input
                    ]
                    + [
                        {
                            "record": cmd.name,
                            "location": command_location(cmd.name),
                            "mime": cmd.mime,
                            "is_executable": True,
                        }
                        for cmd in commands
                    ],
                    "outputs": [
                        {
                            "record": output.record,
                            "location": output.location,
                            "mime": output.mime,
                        }
                        for output in stage.output
                    ],
                    "jobs": jobs,
                    "type": "function" if stage.function else "unix",
                }
            ],
        }
    ]

    ee_config = {
        "stages": stages,
        "realizations": ensemble_size,
        "max_running": 10000,
        "max_retries": 0,
        "run_path": evaluation_tmp_dir / "my_output",
        "executor": ensemble.forward_model.driver,
        "storage": {
            "type": "shared_disk",
            "storage_path": evaluation_tmp_dir / ".my_storage",
        },
        "dispatch_uri": dispatch_uri,
    }

    ee_config["inputs"] = _prepare_input(
        ee_config, stage, input_records, evaluation_tmp_dir, ensemble_size
    )
    ee_config["outputs"] = _prepare_output(
        ee_config, stage, evaluation_tmp_dir, ensemble_size
    )

    return ee_config


def _run(ensemble_evaluator):
    result = None
    with ensemble_evaluator.run() as monitor:
        for event in monitor.track():
            if event.data is not None and event.data.get("status") in [
                _EVTYPE_SNAPSHOT_STOPPED,
                _EVTYPE_SNAPSHOT_FAILED,
            ]:
                if event.data.get("status") == _EVTYPE_SNAPSHOT_STOPPED:
                    result = monitor.get_result()
                monitor.signal_done()
    return result


def _prepare_responses(raw_responses):
    data_results = []
    for iens in sorted(raw_responses.keys(), key=int):
        real_data = {}
        for record, transmitter in raw_responses[iens].items():
            real_data[record] = transmitter.load()
        data_results.append(real_data)

    responses = {response_name: [] for response_name in data_results[0]}
    for realization in data_results:
        assert responses.keys() == realization.keys()
        for key in realization:
            responses[key].append(realization[key])

    for key in responses:
        responses[key] = ert3.data.EnsembleRecord(records=responses[key])

    return ert3.data.MultiEnsembleRecord(ensemble_records=responses)


def evaluate(
    workspace_root, evaluation_name, input_records, ensemble_config, stages_config
):
    evaluation_tmp_dir = _create_evaluator_tmp_dir(workspace_root, evaluation_name)

    config = EvaluatorServerConfig()
    ee_config = _build_ee_config(
        evaluation_tmp_dir,
        ensemble_config,
        stages_config,
        input_records,
        config.dispatch_uri,
    )
    ensemble = PrefectEnsemble(ee_config)

    ee = EnsembleEvaluator(ensemble=ensemble, config=config, iter_=0)
    result = _run(ee)

    responses = _prepare_responses(result)

    return responses


def cleanup(workspace_root, evaluation_name):
    tmp_dir = _create_evaluator_tmp_dir(workspace_root, evaluation_name)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
