import ert3

import json
from pathlib import Path


def export(experiment):
    if not experiment.have_run:
        raise ValueError("Cannot export experiment that has not been carried out")

    input_data = ert3.storage.get_input_data(experiment)
    output_data = ert3.storage.get_output_data(experiment)
    with open(experiment.location / "data.json", "w") as f:
        json.dump(_reformat_input_output(input_data, output_data), f)


def _reformat_input_output(input_data, output_data):
    return [
        {"input": _input_data, "output": _output_data}
        for _input_data, _output_data in zip(input_data, output_data)
    ]
