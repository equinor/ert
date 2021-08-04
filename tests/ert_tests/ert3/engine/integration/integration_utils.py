import numpy as np
import pytest


def load_experiment_config(ensemble_config, stages_config, parameters_config):
    config = {}
    config["ensemble"] = ensemble_config
    config["stages"] = stages_config
    config["parameters"] = parameters_config
    return config


def assert_ensemble_size(config, export_data):
    assert len(export_data) == config["ensemble"].size


def assert_input_records(config, export_data):
    input_records = {}
    for input_data in config["ensemble"].input:
        record = input_data.record
        source = input_data.source

        for p in config["parameters"]:
            if p.type + "." + p.name == source:
                parameter = p
                break

        assert parameter.variables or parameter.size
        has_variables = parameter.variables is not None
        input_records[record] = parameter.variables if has_variables else parameter.size

    for realisation in export_data:
        assert sorted(input_records.keys()) == sorted(realisation["input"].keys())
        for record_name in input_records.keys():
            has_variables = not isinstance(input_records[record_name], int)
            if has_variables:
                input_variables = sorted(input_records[record_name])
                realisation_variables = sorted(realisation["input"][record_name].keys())
                assert input_variables == realisation_variables
            else:
                input_size = input_records[record_name]
                realisation_size = len(realisation["input"][record_name])
                assert input_size == realisation_size


def assert_output_records(config, export_data):
    output_records = []
    for stage in config["stages"]:
        if stage.name == config["ensemble"].forward_model.stage:
            output_records += [output_data.record for output_data in stage.output]
    for realisation in export_data:
        assert sorted(output_records) == sorted(realisation["output"].keys())


def assert_poly_output(export_data):
    for realisation in export_data:
        coeff = realisation["input"]["coefficients"]
        poly_out = realisation["output"]["polynomial_output"]

        assert len(poly_out) == 10
        xs = (
            map(sum, zip(realisation["input"]["x_uncertainties"], range(10)))
            if "x_uncertainties" in realisation["input"]
            else range(10)
        )

        for x, y in zip(xs, poly_out):
            assert coeff["a"] * x ** 2 + coeff["b"] * x + coeff["c"] == pytest.approx(y)


def assert_export(export_data, ensemble_config, stages_config, parameters_config):
    config = load_experiment_config(ensemble_config, stages_config, parameters_config)
    assert_ensemble_size(config, export_data)
    assert_input_records(config, export_data)
    assert_output_records(config, export_data)

    # Note: This test assumes the forward model in the setup indeed
    # evaluates a * x^2 + b * x + c. If not, this will fail miserably!
    assert_poly_output(export_data)


def assert_distribution(
    ensemble_config, stages_config, parameters_config, distribution, coefficients
):
    indices = ("a", "b", "c")

    for real_coefficient in coefficients.records:
        assert sorted(indices) == sorted(real_coefficient.index)
        for idx in real_coefficient.index:
            assert isinstance(real_coefficient.data[idx], float)

    samples = {idx: [] for idx in indices}
    for sample in coefficients.records:
        for key in indices:
            samples[key].append(sample.data[key])

    config = load_experiment_config(ensemble_config, stages_config, parameters_config)
    parameter = None
    for p in config["parameters"]:
        if p.distribution.type == distribution:
            parameter = p
            break

    assert parameter is not None

    input_data = parameter.distribution.input

    for variable in parameter.variables:
        values = samples[variable]

        if distribution == "gaussian":
            assert input_data.mean == pytest.approx(sum(values) / len(values), abs=0.1)
            assert input_data.std == pytest.approx(np.std(values), abs=0.1)

        elif distribution == "uniform":
            assert input_data.lower_bound == pytest.approx(min(values), abs=0.1)
            assert input_data.upper_bound == pytest.approx(max(values), abs=0.1)
            mean = (input_data.lower_bound + input_data.upper_bound) / 2
            assert mean == pytest.approx(sum(values) / len(values), abs=0.1)

        else:
            raise ValueError(f"Unknown distribution {distribution}")


def assert_sensitivity_export(
    export_data, ensemble_config, stages_config, parameters_config, algorithm
):
    if algorithm == "one-at-a-time":
        num_input_coeffs = 3
        export_data_size = 2 * num_input_coeffs
    elif algorithm == "fast":
        num_input_coeffs = 3
        sample_size = 10
        export_data_size = sample_size * num_input_coeffs
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    assert export_data_size == len(export_data)

    config = load_experiment_config(ensemble_config, stages_config, parameters_config)
    assert_input_records(config, export_data)
    assert_output_records(config, export_data)

    # Note: This test assumes the forward model in the setup indeed
    # evaluates a * x^2 + b * x + c. If not, this will fail miserably!
    assert_poly_output(export_data)
