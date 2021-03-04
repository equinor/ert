import ert3

import json
from pathlib import Path


def _prepare_export(workspace_root, experiment_name, parameters, responses):

    data = None

    for record_name, ensemble_record in responses.items():
        if data is None:
            data = [{"input": {}, "output": {}} for _ in ensemble_record.records]
        for realization, record in zip(data, ensemble_record.records):
            realization["output"][record_name] = record.data

    for record_name, ensemble_record in parameters.items():
        if data is None:
            data = [{"input": {}, "output": {}} for _ in ensemble_record.records]
        for realization, record in zip(data, ensemble_record.records):
            realization["input"][record_name] = record.data

    return data


def export(workspace_root, experiment_name):
    experiment_root = (
        Path(workspace_root) / ert3.workspace.EXPERIMENTS_BASE / experiment_name
    )
    ert3.workspace.assert_experiment_exists(workspace_root, experiment_name)

    if not ert3.workspace.experiment_have_run(workspace_root, experiment_name):
        raise ValueError("Cannot export experiment that has not been carried out")

    parameters = ert3.storage.get_experiment_parameters(
        workspace=workspace_root, experiment_name=experiment_name
    )
    responses = ert3.storage.get_ensemble_records(
        workspace=workspace_root, experiment_name=experiment_name
    )

    data = _prepare_export(workspace_root, experiment_name, parameters, responses)
    with open(experiment_root / "data.json", "w") as f:
        json.dump(data, f)
