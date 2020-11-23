import ert3

from pathlib import Path
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


def run(experiment):
    if experiment.have_run:
        raise ValueError(f"Experiment {experiment.name} have been carried out.")

    coefficients = _generate_coefficients()

    ert3.storage.init_experiment(experiment)
    ert3.storage.add_input_data(experiment, coefficients)

    response = ert3.evaluator.evaluate(coefficients)
    ert3.storage.add_output_data(experiment, response)
