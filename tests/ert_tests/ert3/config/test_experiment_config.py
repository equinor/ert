import pytest

import ert3


def test_valid_evaluation():
    raw_config = {"type": "evaluation"}
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "evaluation"
    assert experiment_config.algorithm == None
    assert experiment_config.tail == None


def test_valid_sensitivity_no_tail():
    raw_config = {"type": "sensitivity", "algorithm": "one-at-a-time"}
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "sensitivity"
    assert experiment_config.algorithm == "one-at-a-time"
    assert experiment_config.tail == None


@pytest.mark.parametrize(
    ("algorithm", "tail"),
    (
        ("one-at-a-time", 0.99),
        ("one-at-a-time", None),
    ),
)
def test_valid_sensitivity_tail(algorithm, tail):
    raw_config = {"type": "sensitivity", "algorithm": algorithm, "tail": tail}
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "sensitivity"
    assert experiment_config.algorithm == "one-at-a-time"
    assert experiment_config.tail == 0.99 or experiment_config.tail == None


def test_unknown_experiment_type():
    raw_config = {"type": "unknown_experiment_type"}
    with pytest.raises(
        ert3.exceptions.ConfigValidationError,
        match=r"unexpected value; permitted: 'evaluation', 'sensitivity' \(",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_evaluation_and_algorithm():
    raw_config = {"type": "evaluation", "algorithm": "one-at-a-time"}
    with pytest.raises(
        ert3.exceptions.ConfigValidationError,
        match="Did not expect algorithm for evaluation experiment",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_evaluation_and_tail():
    raw_config = {"type": "evaluation", "tail": "0.99"}
    with pytest.raises(
        ert3.exceptions.ConfigValidationError,
        match="Did not expect tail for evaluation experiment",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_sensitivity_and_no_algorithm():
    raw_config = {"type": "sensitivity"}
    with pytest.raises(
        ert3.exceptions.ConfigValidationError,
        match="Expected an algorithm for sensitivity experiments",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_unkown_sensitivity_algorithm():
    raw_config = {"type": "sensitivity", "algorithm": "unknown_algorithm"}
    with pytest.raises(
        ert3.exceptions.ConfigValidationError,
        match=r"unexpected value; permitted: 'one-at-a-time' \(",
    ):
        ert3.config.load_experiment_config(raw_config)


@pytest.mark.parametrize(
    ("tail", "err_msg"),
    ((-0.5, "Tail cannot be <= 0"), (1.5, "Tail cannot be >= 1")),
)
def test_invalid_tail(tail, err_msg):
    raw_config = {"type": "sensitivity", "algorithm": "one-at-a-time", "tail": tail}
    with pytest.raises(ert3.exceptions.ConfigValidationError, match=err_msg):
        ert3.config.load_experiment_config(raw_config)


def test_unknown_field():
    raw_config = {"type": "evaluation", "unknown": "field"}
    with pytest.raises(
        ert3.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_immutable_field():
    raw_config = {"type": "evaluation"}
    experiment_config = ert3.config.load_experiment_config(raw_config)

    with pytest.raises(TypeError, match="immutable and does not support"):
        experiment_config.type = "toggle"
