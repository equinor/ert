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
    workspace, ensemble, stages_config, base_ensemble_dict
):
    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["size"] = None
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="An ensemble size must be specified.",
    ):
        ert3.config.ExperimentRunConfig(
            ert3.config.ExperimentConfig(type="evaluation"),
            stages_config,
            ert3.config.EnsembleConfig.parse_obj(ensemble_dict),
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
        ert3.config.EnsembleConfig.parse_obj(ensemble_dict),
        ert3.config.ParametersConfig.parse_obj([]),
    )


def test_experiment_run_config_validate_stage(
    workspace, ensemble, double_stages_config, base_ensemble_dict
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
            ert3.config.EnsembleConfig.parse_obj(ensemble_dict),
            ert3.config.ParametersConfig.parse_obj([]),
        )


def test_experiment_run_config_validate_stage_missing_stage_record(
    stages_config, base_ensemble_dict
):

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["input"].append(
        {"source": "stochastic.other_coefficients", "record": "other_coefficients"}
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
            ert3.config.EnsembleConfig.parse_obj(ensemble_dict),
            ert3.config.ParametersConfig.parse_obj([]),
        )
