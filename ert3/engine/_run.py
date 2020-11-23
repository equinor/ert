import ert3

import random


def _generate_coefficients():
    return [
        {
            "coefficients": {
                "a": random.gauss(0, 1),
                "b": random.gauss(0, 1),
                "c": random.gauss(0, 1),
            }
        }
        for _ in range(1000)
    ]


def run(workspace_root, experiment_name):
    experiment_root = Path(workspace_root) / experiment_name
    ert3.workspace.assert_experiment_exists(workspace_root, experiment_name)

    if ert3.workspace.experiment_have_run(workspace_root, experiment_name):
        raise ValueError(f"Experiment {experiment_name} have been carried out.")

    ert3.storage.init_experiment(workspace_root, experiment_name)

    coefficients = _generate_coefficients()
    ert3.storage.add_input_data(workspace_root, experiment_name, coefficients)

    response = ert3.evaluator.evaluate(coefficients)
    ert3.storage.add_output_data(workspace_root, experiment_name, response)
