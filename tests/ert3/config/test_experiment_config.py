import pydantic
import pytest

import ert3


def test_valid_evaluation():
    raw_config = {"type": "evaluation"}
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "evaluation"
    assert experiment_config.algorithm == None


def test_valid_sensitivity():
    raw_config = {"type": "sensitivity", "algorithm": "one-at-a-time"}
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "sensitivity"
    assert experiment_config.algorithm == "one-at-a-time"


def test_unknown_experiment_type():
    raw_config = {"type": "unknown_experiment_type"}
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match="unexpected value; permitted: 'evaluation', 'sensitivity' \(",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_evaluation_and_algorithm():
    raw_config = {"type": "evaluation", "algorithm": "one-at-a-time"}
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match="Did not expect algorithm for evaluation experiment",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_sensitivity_and_no_algorithm():
    raw_config = {"type": "sensitivity"}
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match="Expected an algorithm for sensitivity experiments",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_unkown_sensitivity_algorithm():
    raw_config = {"type": "sensitivity", "algorithm": "unknown_algorithm"}
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match="unexpected value; permitted: 'one-at-a-time' \(",
    ):
        experiment_config = ert3.config.load_experiment_config(raw_config)
