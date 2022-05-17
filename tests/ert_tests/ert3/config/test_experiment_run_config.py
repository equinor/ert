import copy

import pytest

import ert3
import ert


def test_experiment_run_config_validate(workspace, ensemble, stages_config):
    ert3.config._experiment_run_config.ExperimentRunConfig(
        ert3.config.ExperimentConfig(type="evaluation"),
        stages_config,
        ensemble,
        ert3.config.ParametersConfig.parse_obj([]),
    )


def test_experiment_run_config_validate_ensemble_size(
    workspace, ensemble, stages_config, base_ensemble_dict, plugin_registry
):
    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["size"] = None
    ensemble_config = ert3.config.create_ensemble_config(
        plugin_registry=plugin_registry
    )
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="An ensemble size must be specified.",
    ):
        ert3.config.ExperimentRunConfig(
            ert3.config.ExperimentConfig(type="evaluation"),
            stages_config,
            ensemble_config.parse_obj(ensemble_dict),
            ert3.config.ParametersConfig.parse_obj([]),
        )

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="No ensemble size should be specified for a sensitivity analysis.",
    ):
        ert3.config.ExperimentRunConfig(
            ert3.config.ExperimentConfig(type="sensitivity", algorithm="one-at-a-time"),
            stages_config,
            ensemble,
            ert3.config.ParametersConfig.parse_obj([]),
        )

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["size"] = None
    ert3.config.ExperimentRunConfig(
        ert3.config.ExperimentConfig(type="sensitivity", algorithm="one-at-a-time"),
        stages_config,
        ensemble_config.parse_obj(ensemble_dict),
        ert3.config.ParametersConfig.parse_obj([]),
    )


def test_experiment_run_config_validate_stage(
    workspace, ensemble, double_stages_config, base_ensemble_dict, plugin_registry
):
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=(
            "Ensemble and stage inputs do not match.\n"
            "Missing record names in ensemble input: {'other_coefficients'}."
        ),
    ):
        ert3.config.ExperimentRunConfig(
            ert3.config.ExperimentConfig(type="evaluation"),
            double_stages_config,
            ensemble,
            ert3.config.ParametersConfig.parse_obj([]),
        )

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["forward_model"]["stage"] = "foo"
    ensemble_config = ert3.config.create_ensemble_config(
        plugin_registry=plugin_registry
    )
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=(
            "Invalid stage in forward model: 'foo'. "
            "Must be one of: 'evaluate_polynomial'"
        ),
    ):
        ert3.config.ExperimentRunConfig(
            ert3.config.ExperimentConfig(type="evaluation"),
            double_stages_config,
            ensemble_config.parse_obj(ensemble_dict),
            ert3.config.ParametersConfig.parse_obj([]),
        )


def test_experiment_run_config_validate_stage_missing_stage_record(
    stages_config, base_ensemble_dict, plugin_registry
):

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["input"].append(
        {"source": "stochastic.other_coefficients", "name": "other_coefficients"}
    )
    ensemble_config = ert3.config.create_ensemble_config(
        plugin_registry=plugin_registry
    )
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=(
            "Ensemble and stage inputs do not match.\n"
            "Missing record names in stage input: {'other_coefficients'}."
        ),
    ):
        ert3.config.ExperimentRunConfig(
            ert3.config.ExperimentConfig(type="evaluation"),
            stages_config,
            ensemble_config.parse_obj(ensemble_dict),
            ert3.config.ParametersConfig.parse_obj([]),
        )


def test_conflicting_transformation_type(
    stages_config, base_ensemble_dict, plugin_registry
):
    ensemble_dict = copy.deepcopy(base_ensemble_dict)

    # the source will now have a copy transformation attached, which in turn makes
    # the transformation likely to produce a blob, causing a conflict with the stage
    # configuration which expects numerical data
    ensemble_dict["input"][0]["source"] = "resources.coefficients"

    ensemble_config = ert3.config.load_ensemble_config(
        ensemble_dict, plugin_registry=plugin_registry
    )
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="transformation mismatch for input coefficients: source prefers binary "
        + "data and stage prefers numerical data",
    ):
        ert3.config.ExperimentRunConfig(
            ert3.config.ExperimentConfig(type="evaluation"),
            stages_config,
            ensemble_config,
            ert3.config.load_parameters_config([]),
        ).get_linked_inputs()
