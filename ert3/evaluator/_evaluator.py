import json
import os
import pathlib

from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.prefect_ensemble.prefect_ensemble import (
    PrefectEnsemble,
    storage_driver_factory,
)

import ert3

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


def _prepare_input(ee_config, stages_config, inputs, evaluation_tmp_dir):
    tmp_input_folder = evaluation_tmp_dir / "prep_input_files"
    os.makedirs(tmp_input_folder)
    ee_storage = storage_driver_factory(ee_config["storage"], tmp_input_folder)
    record2location = {input.record: input.location for input in stages_config[0].input}
    input_files = {iens: () for iens in range(ee_config["realizations"])}
    for record_name in inputs.record_names:
        for iens, record in enumerate(inputs.ensemble_records[record_name].records):
            filename = record2location[record_name]
            with open(tmp_input_folder / filename, "w") as f:
                json.dump(record.data, f)
            input_files[iens] += (ee_storage.store(filename, iens),)
    return input_files


def _build_ee_config(evaluation_tmp_dir, ensemble, stages_config, input_records):
    _assert_single_stage_forward_model(stages_config, ensemble)

    if ensemble.size != None:
        ensemble_size = ensemble.size
    else:
        ensemble_size = input_records.ensemble_size

    stage_name = ensemble.forward_model.stages[0]
    step_name = stage_name + "-only_step"
    stage = stages_config.step_from_key(stage_name)

    command_scripts = [cmd.location for cmd in stage.transportable_commands]
    output_files = [out.location for out in stage.output]

    cmd_name2script = {cmd.name: cmd.location for cmd in stage.transportable_commands}
    jobs = [
        {
            "name": cmd[0],
            "executable": cmd_name2script[cmd[0]],
            "args": tuple(cmd[1:]),
        }
        for cmd in [elem.split() for elem in stage.script]
    ]

    ee_config = {
        "stages": [
            {
                "name": stage_name,
                "steps": [
                    {
                        "name": step_name,
                        "resources": command_scripts,
                        "parameter": [],
                        "inputs": [],
                        "outputs": output_files,
                        "jobs": jobs,
                    }
                ],
            }
        ],
        "realizations": ensemble_size,
        "max_running": 10000,
        "max_retries": 1,
        "run_path": evaluation_tmp_dir / "my_output",
        "executor": ensemble.forward_model.driver,
        "storage": {
            "type": "shared_disk",
            "storage_path": evaluation_tmp_dir / ".my_storage",
        },
    }

    ee_config["input_files"] = _prepare_input(
        ee_config, stages_config, input_records, evaluation_tmp_dir
    )

    return ee_config


def _fetch_results(ee_config, ensemble, stages_config):
    _assert_single_stage_forward_model(stages_config, ensemble)

    results = []
    ee_storage = storage_driver_factory(ee_config["storage"], ee_config["run_path"])
    for iens in range(ee_config["realizations"]):
        spath = pathlib.Path(ee_storage.get_storage_path(iens))
        realization_results = {}

        stage_name = ensemble.forward_model.stages[0]
        stage = stages_config.step_from_key(stage_name)
        for output_elem in stage.output:
            with open(spath / output_elem.location) as f:
                realization_results[output_elem.record] = json.load(f)
        results.append(realization_results)
    return results


def _run(ensemble_evaluator):
    with ensemble_evaluator.run() as monitor:
        for event in monitor.track():
            if event.data is not None and event.data.get("status") in [
                _EVTYPE_SNAPSHOT_STOPPED,
                _EVTYPE_SNAPSHOT_FAILED,
            ]:
                monitor.signal_done()


def _prepare_responses(raw_responses):
    responses = {response_name: [] for response_name in raw_responses[0]}
    for realization in raw_responses:
        assert responses.keys() == realization.keys()
        for key in realization:
            responses[key].append(ert3.data.Record(data=realization[key]))

    for key in responses:
        responses[key] = ert3.data.EnsembleRecord(records=responses[key])

    return ert3.data.MultiEnsembleRecord(ensemble_records=responses)


def evaluate(
    workspace_root, evaluation_name, input_records, ensemble_config, stages_config
):
    evaluation_tmp_dir = _create_evaluator_tmp_dir(workspace_root, evaluation_name)

    ee_config = _build_ee_config(
        evaluation_tmp_dir, ensemble_config, stages_config, input_records
    )
    ensemble = PrefectEnsemble(ee_config)

    config = EvaluatorServerConfig()
    ee = EnsembleEvaluator(ensemble=ensemble, config=config, iter_=0)
    _run(ee)

    results = _fetch_results(ee_config, ensemble_config, stages_config)
    responses = _prepare_responses(results)

    return responses
