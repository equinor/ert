import pytest

import ert
import ert3


def test_valid_evaluation():
    raw_config = {"type": "evaluation"}
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "evaluation"
    assert experiment_config.algorithm == None
    assert experiment_config.tail == None
    assert experiment_config.harmonics == None
    assert experiment_config.sample_size == None


def test_valid_sensitivity_no_tail():
    raw_config = {"type": "sensitivity", "algorithm": "one-at-a-time"}
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "sensitivity"
    assert experiment_config.algorithm == "one-at-a-time"
    assert experiment_config.tail == None
    assert experiment_config.harmonics == None
    assert experiment_config.sample_size == None


@pytest.mark.parametrize(
    ("tail"),
    ((0.99), (None)),
)
def test_valid_sensitivity_tail(tail):
    raw_config = {"type": "sensitivity", "algorithm": "one-at-a-time", "tail": tail}
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "sensitivity"
    assert experiment_config.algorithm == "one-at-a-time"
    assert experiment_config.tail == tail


def test_valid_sensitivity_fast():
    raw_config = {
        "type": "sensitivity",
        "algorithm": "fast",
        "harmonics": 4,
        "sample_size": 1000,
    }
    experiment_config = ert3.config.load_experiment_config(raw_config)
    assert experiment_config.type == "sensitivity"
    assert experiment_config.algorithm == "fast"
    assert experiment_config.harmonics == 4
    assert experiment_config.sample_size == 1000


def test_unknown_experiment_type():
    raw_config = {"type": "unknown_experiment_type"}
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=r"unexpected value; permitted: 'evaluation', 'sensitivity' \(",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_evaluation_and_algorithm():
    raw_config = {"type": "evaluation", "algorithm": "one-at-a-time"}
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Did not expect algorithm for evaluation experiment",
    ):
        ert3.config.load_experiment_config(raw_config)


@pytest.mark.parametrize(
    ("raw_config", "err_msg"),
    (
        (
            {"type": "evaluation", "tail": "0.99"},
            "Did not expect tail for evaluation experiment",
        ),
        (
            {"type": "evaluation", "harmonics": "4"},
            "Did not expect harmonics for evaluation experiment",
        ),
        (
            {"type": "evaluation", "sample_size": "1000"},
            "Did not expect sample_size for evaluation experiment",
        ),
    ),
)
def test_evaluation_and_algorithm_parameters(raw_config, err_msg):
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=err_msg,
    ):
        ert3.config.load_experiment_config(raw_config)


def test_sensitivity_and_no_algorithm():
    raw_config = {"type": "sensitivity"}
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Expected an algorithm for sensitivity experiments",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_unkown_sensitivity_algorithm():
    raw_config = {"type": "sensitivity", "algorithm": "unknown_algorithm"}
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=r"unexpected value; permitted: 'one-at-a-time', 'fast' \(",
    ):
        ert3.config.load_experiment_config(raw_config)


@pytest.mark.parametrize(
    ("raw_config", "err_msg"),
    (
        (
            {"type": "sensitivity", "algorithm": "one-at-a-time", "harmonics": 4},
            "Did not expect harmonics for one-at-a-time algorithm",
        ),
        (
            {"type": "sensitivity", "algorithm": "one-at-a-time", "sample_size": 1000},
            "Did not expect sample_size for one-at-a-time algorithm",
        ),
    ),
)
def test_invalid_oat_parameters(raw_config, err_msg):
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=err_msg,
    ):
        ert3.config.load_experiment_config(raw_config)


def test_fast_and_tail():
    raw_config = {"type": "sensitivity", "algorithm": "fast", "tail": 0.99}
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Did not expect tail for fast algorithm",
    ):
        ert3.config.load_experiment_config(raw_config)


@pytest.mark.parametrize(
    ("raw_config", "err_msg"),
    (
        (
            {"type": "sensitivity", "algorithm": "fast", "harmonics": 4},
            "Expected sample_size for fast algorithm",
        ),
        (
            {"type": "sensitivity", "algorithm": "fast", "sample_size": 1000},
            "Expected harmonics for fast algorithm",
        ),
        (
            {
                "type": "sensitivity",
                "algorithm": "fast",
                "harmonics": 4,
                "sample_size": 10,
            },
            "Sample_size must be >= 4 \\* harmonics\\^2 \\+ 1",
        ),
    ),
)
def test_invalid_fast_parameters(raw_config, err_msg):
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=err_msg,
    ):
        ert3.config.load_experiment_config(raw_config)


@pytest.mark.parametrize(
    ("tail", "err_msg"),
    ((-0.5, "Tail cannot be <= 0"), (1.5, "Tail cannot be >= 1")),
)
def test_invalid_tail(tail, err_msg):
    raw_config = {"type": "sensitivity", "algorithm": "one-at-a-time", "tail": tail}
    with pytest.raises(ert.exceptions.ConfigValidationError, match=err_msg):
        ert3.config.load_experiment_config(raw_config)


def test_invalid_harmonics():
    raw_config = {"type": "sensitivity", "algorithm": "fast", "harmonics": -4}
    with pytest.raises(
        ert.exceptions.ConfigValidationError, match="Harmonics cannot be <= 0"
    ):
        ert3.config.load_experiment_config(raw_config)


def test_invalid_sample_size():
    raw_config = {"type": "sensitivity", "algorithm": "fast", "sample_size": -1000}
    with pytest.raises(
        ert.exceptions.ConfigValidationError, match="Sample_size cannot be <= 0"
    ):
        ert3.config.load_experiment_config(raw_config)


def test_unknown_field():
    raw_config = {"type": "evaluation", "unknown": "field"}
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_experiment_config(raw_config)


def test_immutable_field():
    raw_config = {"type": "evaluation"}
    experiment_config = ert3.config.load_experiment_config(raw_config)

    with pytest.raises(TypeError, match="immutable and does not support"):
        experiment_config.type = "toggle"
