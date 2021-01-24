import ert3

import json
import yaml


def load_record(workspace, record_name, record_file):
    record = json.load(record_file)
    record_file.close()

    ert3.storage.add_variables(
        workspace,
        record_name,
        record,
    )


def sample_record(workspace, parameter_group_name, record_name, ensemble_size):
    with open(workspace / "parameters.yml") as f:
        parameters = yaml.safe_load(f)

    parameter_group = None
    for ps in parameters:
        if ps["name"] == parameter_group_name:
            parameter_group = ps

    if parameter_group is None:
        raise ValueError(f"No parameter group found named: {parameter_group_name}")

    dist_config = parameter_group["distribution"]
    if dist_config["type"] == "gaussian":
        distribution = ert3.stats.Gaussian(
            dist_config["input"]["mean"],
            dist_config["input"]["std"],
            index=parameter_group["variables"],
        )
    elif dist_config["type"] == "uniform":
        distribution = ert3.stats.Uniform(
            dist_config["input"]["lower_bound"],
            dist_config["input"]["upper_bound"],
            index=parameter_group["variables"],
        )
    else:
        raise ValueError("Unknown distribution type: {}".format(dist_config["type"]))

    ert3.storage.add_variables(
        workspace,
        record_name,
        [distribution.sample() for _ in range(ensemble_size)],
    )
