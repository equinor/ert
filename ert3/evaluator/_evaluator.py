import json
import os
import pathlib

from ert_shared.ensemble_evaluator import config as evaluator_config
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.prefect_ensemble.prefect_ensemble import (
    PrefectEnsemble,
    storage_driver_factory,
)

import ert3


def _create_evaluator_tmp_dir(workspace_root, evaluation_name):
    return (
        pathlib.Path(workspace_root)
        / ert3._WORKSPACE_DATA_ROOT
        / "tmp"
        / evaluation_name
    )


def _prepare_input(ee_config, ensemble, stages_config, inputs, evaluation_tmp_dir):
    tmp_input_folder = evaluation_tmp_dir / "prep_input_files"
    os.makedirs(tmp_input_folder)
    ee_storage = storage_driver_factory(ee_config["storage"], tmp_input_folder)
    record2location = {}
    for stage_name in ensemble.forward_model.stages:
        stage = stages_config.step_from_key(stage_name)
        record2location.update({input.record: input.location for input in stage.input})
    input_files = {iens: () for iens in range(ee_config["realizations"])}
    for iens, realization_inputs in enumerate(inputs):
        for name, value in realization_inputs.items():
            filename = record2location[name]
            with open(tmp_input_folder / filename, "w") as f:
                json.dump(value, f)
            input_files[iens] += (ee_storage.store(filename, iens),)
    return input_files


def _build_ee_config(evaluation_tmp_dir, ensemble, stages_config, input_records):
    stage_names = ensemble.forward_model.stages

    ee_config = {
        "stages": stages_config.get_prefect_stages(stage_names),
        "realizations": ensemble.size,
        "max_running": 10000,
        "max_retries": 0,
        "run_path": evaluation_tmp_dir / "my_output",
        "executor": ensemble.forward_model.driver,
        "storage": {
            "type": "shared_disk",
            "storage_path": evaluation_tmp_dir / ".my_storage",
        },
    }

    ee_config["input_files"] = _prepare_input(
        ee_config, ensemble, stages_config, input_records, evaluation_tmp_dir
    )

    return ee_config


def _fetch_results(ee_config, ensemble, stages_config):

    results = []
    ee_storage = storage_driver_factory(ee_config["storage"], ee_config["run_path"])
    for iens in range(ee_config["realizations"]):
        spath = pathlib.Path(ee_storage.get_storage_path(iens))
        realization_results = {}
        for stage_name in ensemble.forward_model.stages:
            stage = stages_config.step_from_key(stage_name)
            for output_elem in stage.output:
                with open(spath / output_elem.location) as f:
                    realization_results[output_elem.record] = json.load(f)
        results.append(realization_results)
    return results


def _run(ensemble_evaluator):
    monitor = ensemble_evaluator.run()
    for event in monitor.track():
        if event.data is not None and event.data.get("status") == "Stopped":
            monitor.signal_done()


def evaluate(
    workspace_root, evaluation_name, input_records, ensemble_config, stages_config
):
    evaluation_tmp_dir = _create_evaluator_tmp_dir(workspace_root, evaluation_name)

    ee_config = _build_ee_config(
        evaluation_tmp_dir, ensemble_config, stages_config, input_records
    )
    ensemble = PrefectEnsemble(ee_config)

    config = evaluator_config.load_config()
    ee = EnsembleEvaluator(ensemble=ensemble, config=config)
    _run(ee)

    results = _fetch_results(ee_config, ensemble_config, stages_config)
    return results
