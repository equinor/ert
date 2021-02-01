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


def assert_ensemble_size(config, export_data):
    assert len(export_data) == config["ensemble"].size


def assert_input_records(config, export_data):
    input_records = {}
    for input_data in config["ensemble"].input:
        record = input_data.record
        source = input_data.source

        for p in config["parameters"]:
            if p["type"] + "." + p["name"] == source:
                parameter = p
                break

        input_records[record] = parameter["variables"]

    for realisation in export_data:
        assert sorted(input_records.keys()) == sorted(realisation["input"].keys())
        for record_name in input_records.keys():
            input_variables = sorted(input_records[record_name])
            realisation_variables = sorted(realisation["input"][record_name].keys())
            assert input_variables == realisation_variables


def assert_output_records(config, export_data):
    output_records = []
    for forward_stage in config["ensemble"].forward_model.stages:
        for stage in config["stages"]:
            if stage.name == forward_stage:
                output_records += [output_data.record for output_data in stage.output]
    for realisation in export_data:
        assert sorted(output_records) == sorted(realisation["output"].keys())


def assert_poly_output(export_data):
    for realisation in export_data:
        coeff = realisation["input"]["coefficients"]
        poly_out = realisation["output"]["polynomial_output"]

        assert 10 == len(poly_out)
        for x, y in zip(range(10), poly_out):
            assert coeff["a"] * x ** 2 + coeff["b"] * x + coeff["c"] == pytest.approx(y)


def assert_export(workspace, experiment_name, ensemble_config, stages_config):
    with open(workspace / experiment_name / "data.json") as f:
        export_data = json.load(f)

    config = load_experiment_config(workspace, ensemble_config, stages_config)
    assert_ensemble_size(config, export_data)
    assert_input_records(config, export_data)
    assert_output_records(config, export_data)

    # Note: This test assumes the forward model in the setup indeed
    # evaluates a * x^2 + b * x + c. If not, this will fail miserably!
    assert_poly_output(export_data)


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


def assert_sensitivity_oat_export(
    workspace, experiment_name, ensemble_config, stages_config
):
    with open(workspace / experiment_name / "data.json") as f:
        export_data = json.load(f)

    num_input_coeffs = 3
    assert 2 * num_input_coeffs == len(export_data)

    config = load_experiment_config(workspace, ensemble_config, stages_config)
    assert_input_records(config, export_data)
    assert_output_records(config, export_data)

    # Note: This test assumes the forward model in the setup indeed
    # evaluates a * x^2 + b * x + c. If not, this will fail miserably!
    assert_poly_output(export_data)
