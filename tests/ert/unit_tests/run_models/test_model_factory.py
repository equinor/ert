import queue
from argparse import Namespace
from unittest.mock import MagicMock, patch
from uuid import uuid1

import pytest

import ert
from ert.config import (
    AnalysisConfig,
    ConfigValidationError,
    ConfigWarning,
    ErtConfig,
    ModelConfig,
    ObservationSettings,
)
from ert.mode_definitions import (
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    EVALUATE_ENSEMBLE_MODE,
)
from ert.run_models import (
    EnsembleExperiment,
    EnsembleSmoother,
    MultipleDataAssimilation,
    SingleTestRun,
    create_model,
    model_factory,
)
from ert.run_models.model_factory import (
    _setup_ensemble_information_filter,
    _setup_ensemble_smoother,
    _setup_multiple_data_assimilation,
)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ENSEMBLE_SMOOTHER_MODE),
        pytest.param(ES_MDA_MODE),
    ],
)
def test_that_the_model_warns_when_active_realizations_less_min_realizations(
    mode,
    change_to_tmpdir,
):
    """
    Verify that the run model checks that active realizations is equal or higher than
    NUM_REALIZATIONS when running an experiment.
    A warning is issued when NUM_REALIZATIONS is higher than active_realizations.
    """
    with pytest.warns(
        ConfigWarning,
        match=(
            "MIN_REALIZATIONS was set to the current "
            r"number of active realizations \(5\)"
        ),
    ):
        _ = model_factory.create_model(
            ErtConfig.from_file_contents(
                """\
                NUM_REALIZATIONS 100
                MIN_REALIZATIONS 10
                """
            ),
            Namespace(
                mode=mode,
                realizations="0-4",
                target_ensemble="target",
                experiment_name="experiment",
                num_iterations=1,
                restart_run=False,
                prior_ensemble_id="",
                weights="2,3",
            ),
            queue.SimpleQueue(),
        )


def test_iterative_ensemble_format_is_set_by_target_ensemble():
    assert (
        model_factory._iterative_ensemble_format(
            Namespace(current_ensemble="current", target_ensemble="target_%d")
        )
        == "target_%d"
    )


def test_iterative_ensemble_format_defaults_to_current_when_no_target_ensemble_is_given():  # noqa: E501
    assert (
        model_factory._iterative_ensemble_format(
            Namespace(current_ensemble="current", target_ensemble=None)
        )
        == "current_%d"
    )


def test_ensemble_format_is_default_when_neither_current_or_target_is_given():
    assert (
        model_factory._iterative_ensemble_format(Namespace(target_ensemble=None))
        == "default_%d"
    )


def test_default_realizations():
    ensemble_size = 100
    assert (
        model_factory._realizations(
            Namespace(realizations=None), ensemble_size
        ).tolist()
        == [True] * ensemble_size
    )


def test_custom_realizations():
    ensemble_size = 100
    args = Namespace(realizations="0-4,7,8")
    active_mask = [False] * ensemble_size
    active_mask[0:5] = [True] * 5
    active_mask[7:9] = [True] * 2
    assert model_factory._realizations(args, ensemble_size).tolist() == active_mask


def test_setup_single_test_run(tmp_path):
    model = model_factory._setup_single_test_run(
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 100\nENSPATH {tmp_path}"),
        Namespace(
            current_ensemble="current-ensemble",
            target_ensemble=None,
            random_seed=None,
            experiment_name=None,
        ),
        queue.SimpleQueue(),
    )
    assert isinstance(model, SingleTestRun)
    assert model._storage.path == tmp_path


def test_setup_single_test_run_with_ensemble(tmp_path):
    model = model_factory._setup_single_test_run(
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 100\nENSPATH {tmp_path}"),
        Namespace(
            current_ensemble="current-ensemble",
            target_ensemble=None,
            random_seed=None,
            experiment_name=None,
        ),
        queue.SimpleQueue(),
    )
    assert isinstance(model, SingleTestRun)
    assert model._storage.path == tmp_path


def test_setup_ensemble_experiment(tmp_path):
    model = model_factory._setup_ensemble_experiment(
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 100\nENSPATH {tmp_path}"),
        Namespace(
            realizations=None,
            iter_num=1,
            current_ensemble="default",
            target_ensemble=None,
            experiment_name="ensemble_experiment",
        ),
        queue.SimpleQueue(),
    )
    assert isinstance(model, EnsembleExperiment)

    assert model.active_realizations == [True] * 100


@pytest.mark.filterwarnings("ignore:MIN_REALIZATIONS")
def test_setup_ensemble_smoother(tmp_path):
    model = model_factory._setup_ensemble_smoother(
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 100\nENSPATH {tmp_path}"),
        Namespace(
            realizations="0-4,7,8",
            current_ensemble="default",
            target_ensemble="test_case",
            experiment_name="just_smoothing",
        ),
        ObservationSettings(),
        queue.SimpleQueue(),
    )
    assert isinstance(model, EnsembleSmoother)
    assert (
        model.active_realizations
        == [True] * 5 + [False] * 2 + [True] * 2 + [False] * 91
    )


@pytest.mark.filterwarnings("ignore:MIN_REALIZATIONS")
def test_setup_multiple_data_assimilation(tmp_path):
    model = model_factory._setup_multiple_data_assimilation(
        ErtConfig.from_file_contents(f"NUM_REALIZATIONS 100\nENSPATH {tmp_path}"),
        Namespace(
            realizations="0-4,8",
            weights="6,4,2",
            target_ensemble="test_case_%d",
            restart_run=False,
            prior_ensemble_id="b272fe09-83ac-4744-b667-9a0a5415420b",
            experiment_name="My-experiment",
            starting_iteration=0,
        ),
        ObservationSettings(),
        queue.SimpleQueue(),
    )
    assert isinstance(model, MultipleDataAssimilation)
    assert model.weights == "6,4,2"
    assert model._parsed_weights == MultipleDataAssimilation.parse_weights("6,4,2")
    assert (
        model.active_realizations
        == [True] * 5 + [False] * 3 + [True] * 1 + [False] * 91
    )
    assert model.target_ensemble == "test_case_%d"
    assert model.prior_ensemble_id == "b272fe09-83ac-4744-b667-9a0a5415420b"
    assert model.restart_run is False


@pytest.mark.parametrize(
    "restart_from_iteration, expected_path",
    [
        [
            0,
            [
                "realization-0/iter-1",
                "realization-0/iter-2",
                "realization-0/iter-3",
                "realization-1/iter-1",
                "realization-1/iter-2",
                "realization-1/iter-3",
            ],
        ],
        [
            1,
            [
                "realization-0/iter-2",
                "realization-0/iter-3",
                "realization-1/iter-2",
                "realization-1/iter-3",
            ],
        ],
        [2, ["realization-0/iter-3", "realization-1/iter-3"]],
        [3, []],
    ],
)
def test_multiple_data_assimilation_restart_paths(
    tmp_path, monkeypatch, restart_from_iteration, expected_path
):
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        realizations="0,1",
        weights="6,4,2",
        target_ensemble="restart_case_%d",
        restart_run=True,
        prior_ensemble_id=str(uuid1()),
        experiment_name="just_assimilatin",
    )

    monkeypatch.setattr(
        ert.run_models.run_model.RunModel,
        "validate_successful_realizations_count",
        MagicMock(),
    )
    ensemble_mock = MagicMock()
    ensemble_mock.iteration = restart_from_iteration
    config = ErtConfig(runpath_config=ModelConfig(num_realizations=2))

    with patch(
        "ert.run_models.run_model.Storage.get_ensemble", return_value=ensemble_mock
    ):
        model = model_factory._setup_multiple_data_assimilation(
            config, args, MagicMock(spec=ObservationSettings), MagicMock()
        )
    base_path = tmp_path / "simulations"
    expected_path = [str(base_path / expected) for expected in expected_path]
    assert set(model.paths) == set(expected_path)


@pytest.mark.parametrize(
    "analysis_mode",
    [
        model_factory._setup_multiple_data_assimilation,
        model_factory._setup_ensemble_smoother,
    ],
)
def test_num_realizations_specified_incorrectly_raises(analysis_mode):
    config = ErtConfig(runpath_config=ModelConfig(num_realizations=1))
    args = Namespace(
        realizations="0",
        weights="6,4,2",
        target_ensemble="restart_case_%d",
        restart_run=True,
        prior_ensemble_id=str(uuid1()),
        experiment_name=None,
    )

    with pytest.raises(
        ConfigValidationError,
        match="Number of active realizations must be at least 2 for an update step",
    ):
        analysis_mode(config, args, MagicMock(), MagicMock())


@pytest.mark.parametrize(
    "ensemble_iteration, expected_path",
    [
        [0, ["realization-0/iter-0"]],
        [1, ["realization-0/iter-1"]],
        [2, ["realization-0/iter-2"]],
        [100, ["realization-0/iter-100"]],
    ],
)
def test_evaluate_ensemble_paths(
    tmp_path, monkeypatch, ensemble_iteration, expected_path
):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        ert.run_models.run_model.RunModel,
        "validate_successful_realizations_count",
        MagicMock(),
    )
    ensemble_mock = MagicMock()
    ensemble_mock.iteration = ensemble_iteration
    config = ErtConfig(
        random_seed=1234,
        runpath_config=ModelConfig(num_realizations=1),
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
    )

    with patch(
        "ert.run_models.run_model.Storage.get_ensemble", return_value=ensemble_mock
    ):
        model = create_model(
            config,
            Namespace(
                ensemble_id=str(uuid1(0)),
                mode=EVALUATE_ENSEMBLE_MODE,
                realizations=None,
            ),
            MagicMock(),
        )

    base_path = tmp_path / "simulations"
    expected_path = [str(base_path / expected) for expected in expected_path]
    assert set(model.paths) == set(expected_path)


@pytest.mark.parametrize(
    "experiment_setup_method",
    [
        _setup_multiple_data_assimilation,
        _setup_ensemble_information_filter,
        _setup_ensemble_smoother,
    ],
)
def test_that_setting_up_experiment_with_update_step_raises_config_validation_error_given_less_than_two_active_realizations(  # noqa: E501
    experiment_setup_method,
):
    """This test tests that specifying a single realization to run in an update
    experiment is not allowed.
    Though confusing, the active realizations are derived from args.realizations (the
    ones specified in the gui) and all active realization in the 'config', meaning we
    are referring to two different active realizations."""
    args = MagicMock(realizations="0", restart_run=False, prior_ensemble_id="")
    config = MagicMock()
    config.active_realizations = [True] * 10
    config.analysis_config = MagicMock(minimum_required_realizations=1)

    with pytest.raises(
        ConfigValidationError,
        match="Number of active realizations must be at least 2 for an update step",
    ):
        experiment_setup_method(config, args, MagicMock(), MagicMock())
