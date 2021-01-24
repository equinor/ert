import ert3

import json
from pathlib import Path


def export(workspace_root, experiment_name):
    experiment_root = Path(workspace_root) / experiment_name
    ert3.workspace.assert_experiment_exists(workspace_root, experiment_name)

    if not ert3.workspace.experiment_have_run(workspace_root, experiment_name):
        raise ValueError("Cannot export experiment that has not been carried out")

    input_data = ert3.storage.get_input_data(workspace_root, experiment_name)
    output_data = ert3.storage.get_output_data(workspace_root, experiment_name)
    with open(experiment_root / "data.json", "w") as f:
        json.dump(_reformat_input_output(input_data, output_data), f)


def _reformat_input_output(input_data, output_data):
    return [
        {"input": _input_data, "output": _output_data}
        for _input_data, _output_data in zip(input_data, output_data)
    ]
