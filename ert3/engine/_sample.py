import ert3

import yaml


def sample(workspace, parameter_group_name, sample_name, ensemble_size):
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
    else:
        raise ValueError("Unknown distribution type: {}".format(dist_config["type"]))

    ert3.storage.add_variables(
        workspace,
        sample_name,
        [distribution.sample() for _ in range(ensemble_size)],
    )
