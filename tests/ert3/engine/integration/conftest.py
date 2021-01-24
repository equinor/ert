import json
import yaml
import pytest
import numpy as np


def load_experiment_config(workspace, ensemble_config, stages_config):
    config = {}
    config["ensemble"] = ensemble_config
    config["stages"] = stages_config
    with open(workspace / "parameters.yml") as f:
        config["parameters"] = yaml.safe_load(f)
    return config


def assert_distribution(
    workspace, ensemble_config, stages_config, distribution, coefficients
):
    indices = ("a", "b", "c")

    for real_coefficient in coefficients:
        assert sorted(indices) == sorted(real_coefficient.keys())
        for val in real_coefficient.values():
            assert isinstance(val, float)

    samples = {idx: [] for idx in indices}
    for sample in coefficients:
        for key in indices:
            samples[key].append(sample[key])

    config = load_experiment_config(workspace, ensemble_config, stages_config)
    parameter = None
    for p in config["parameters"]:
        if p["distribution"]["type"] == distribution:
            parameter = p
            break

    assert parameter is not None

    input_data = parameter["distribution"]["input"]

    for variable in parameter["variables"]:
        values = samples[variable]

        if distribution == "gaussian":
            assert input_data["mean"] == pytest.approx(
                sum(values) / len(values), abs=0.1
            )
            assert input_data["std"] == pytest.approx(np.std(values), abs=0.1)

        elif distribution == "uniform":
            assert input_data["lower_bound"] == pytest.approx(min(values), abs=0.1)
            assert input_data["upper_bound"] == pytest.approx(max(values), abs=0.1)
            mean = (input_data["lower_bound"] + input_data["upper_bound"]) / 2
            assert mean == pytest.approx(sum(values) / len(values), abs=0.1)

        else:
            raise ValueError(f"Unknown distribution {distribution}")
