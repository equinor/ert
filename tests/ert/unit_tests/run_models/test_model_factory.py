import queue
from argparse import Namespace
from unittest.mock import MagicMock, patch
from uuid import uuid1

import pytest
from pydantic import ValidationError

import ert
from ert.config import (
    AnalysisConfig,
    ConfigValidationError,
    ConfigWarning,
    EnsembleConfig,
    ErtConfig,
    GenKwConfig,
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
    _resolve_parameter_configs,
    _setup_ensemble_information_filter,
    _setup_ensemble_smoother,
    _setup_multiple_data_assimilation,
)


def _gen_kw_config(name: str = "COEFFS") -> GenKwConfig:
    return GenKwConfig(name=name, distribution={"name": "normal", "mean": 0, "std": 1})


def _gen_kw_config_text(tmp_path, kw_name: str = "COEFFS") -> str:
    """Writes a GEN_KW prior file to tmp_path and returns the corresponding
    GEN_KW config line.
    """
    prior_file = tmp_path / "prior.txt"
    prior_file.write_text(f"{kw_name} NORMAL 0 1", encoding="utf-8")
    return f"GEN_KW KW_NAME {prior_file}"


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ENSEMBLE_SMOOTHER_MODE),
        pytest.param(ES_MDA_MODE),
    ],
)
def test_that_the_model_warns_when_active_realizations_less_min_realizations(
    mode,
    tmp_path,
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
                f"""\
                NUM_REALIZATIONS 100
                MIN_REALIZATIONS 10
                {_gen_kw_config_text(tmp_path)}
                """
            ),
            Namespace(
                mode=mode,
                realizations="0-4",
                target_ensemble="target",
                experiment_name="experiment",
                num_iterations=1,
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


def test_iterative_ensemble_format_defaults_to_current_when_no_target_ensemble_is_given():  # ruff: ignore[line-too-long]
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


@pytest.mark.parametrize("realizations", ["0", "0-2", "1-2"])
def test_that_single_test_setup_requires_and_runs_only_realization_zero(
    tmp_path, realizations
):
    config = ErtConfig.from_file_contents(f"NUM_REALIZATIONS 3\nENSPATH {tmp_path}")
    args = Namespace(
        realizations=realizations,
        current_ensemble="ensemble",
        experiment_name="experiment",
    )
    if realizations == "1-2":
        with pytest.raises(ValidationError, match="first realization is inactive"):
            model_factory._setup_single_test_run(config, args, queue.SimpleQueue())
    else:
        model = model_factory._setup_single_test_run(config, args, queue.SimpleQueue())
        assert model.active_realizations == [True]


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
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 100\nENSPATH {tmp_path}\n{_gen_kw_config_text(tmp_path)}"
        ),
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
def test_that_setup_multiple_data_assimilation_uses_the_arguments_from_the_cli(
    tmp_path,
):
    model = model_factory._setup_multiple_data_assimilation(
        ErtConfig.from_file_contents(
            f"NUM_REALIZATIONS 100\nENSPATH {tmp_path}\n{_gen_kw_config_text(tmp_path)}"
        ),
        Namespace(
            realizations="0-4,8",
            weights="6,4,2",
            target_ensemble="test_case_%d",
            prior_ensemble_id=None,
            experiment_name="My-experiment",
            starting_iteration=0,
        ),
        ObservationSettings(),
        queue.SimpleQueue(),
    )
    assert isinstance(model, MultipleDataAssimilation)
    assert model.analysis_settings.weights == "6,4,2"
    assert model._parsed_weights == MultipleDataAssimilation.parse_weights("6,4,2")
    assert (
        model.active_realizations
        == [True] * 5 + [False] * 3 + [True] * 1 + [False] * 91
    )
    assert model.target_ensemble == "test_case_%d"
    assert model.prior_ensemble_id is None


@pytest.mark.filterwarnings("ignore:MIN_REALIZATIONS")
def test_that_setup_multiple_data_assimilation_uses_config_weights_when_cli_omits_them(
    tmp_path,
):
    model = model_factory._setup_multiple_data_assimilation(
        ErtConfig.from_file_contents(
            f"""
            NUM_REALIZATIONS 100
            ENSPATH {tmp_path}
            ANALYSIS_SET_VAR STD_ENKF WEIGHTS 8, 4, 2, 1
            {_gen_kw_config_text(tmp_path)}
            """
        ),
        Namespace(
            realizations="0-4,8",
            target_ensemble="test_case_%d",
            weights=None,
            prior_ensemble_id=None,
            experiment_name="My-experiment",
            starting_iteration=0,
        ),
        ObservationSettings(),
        queue.SimpleQueue(),
    )
    assert model.analysis_settings.weights == "8, 4, 2, 1"


@pytest.mark.parametrize(
    ("restart_from_iteration", "expected_path"),
    [
        (
            0,
            [
                "realization-0/iter-1",
                "realization-0/iter-2",
                "realization-0/iter-3",
                "realization-1/iter-1",
                "realization-1/iter-2",
                "realization-1/iter-3",
            ],
        ),
        (
            1,
            [
                "realization-0/iter-2",
                "realization-0/iter-3",
                "realization-1/iter-2",
                "realization-1/iter-3",
            ],
        ),
        (2, ["realization-0/iter-3", "realization-1/iter-3"]),
        (3, []),
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
            config, args, ObservationSettings(), queue.SimpleQueue()
        )
    base_path = tmp_path / "simulations"
    expected_path = [str(base_path / expected) for expected in expected_path]
    assert set(model.paths) == set(expected_path)


@pytest.mark.parametrize(
    "analysis_mode",
    [
        model_factory._setup_multiple_data_assimilation,
        model_factory._setup_ensemble_smoother,
        model_factory._setup_ensemble_information_filter,
        model_factory._setup_manual_update,
        model_factory._setup_manual_update_enif,
    ],
)
def test_that_update_setup_rejects_one_active_realization(analysis_mode):
    parameter = _gen_kw_config()
    config = ErtConfig(
        runpath_config=ModelConfig(num_realizations=1),
        ensemble_config=EnsembleConfig(parameter_configs={parameter.name: parameter}),
    )
    args = Namespace(
        realizations="0",
        weights="6,4,2",
        target_ensemble="restart_case_%d",
        prior_ensemble_id=str(uuid1()),
        experiment_name="experiment",
        ensemble_id=str(uuid1()),
    )

    with pytest.raises(
        ValidationError,
        match="Number of active realizations must be at least 2 for an update step",
    ):
        analysis_mode(config, args, ObservationSettings(), queue.SimpleQueue())


@pytest.mark.parametrize(
    ("ensemble_iteration", "expected_path"),
    [
        (0, ["realization-0/iter-0"]),
        (1, ["realization-0/iter-1"]),
        (2, ["realization-0/iter-2"]),
        (100, ["realization-0/iter-100"]),
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
            queue.SimpleQueue(),
        )

    base_path = tmp_path / "simulations"
    expected_path = [str(base_path / expected) for expected in expected_path]
    assert set(model.paths) == set(expected_path)


@pytest.mark.parametrize("has_parameters", [False, True], ids=["empty", "all-disabled"])
def test_that_prior_ensemble_allows_current_config_without_updatable_parameters(
    has_parameters: bool,
):
    parameter = _gen_kw_config()
    parameter.update_strategy = None
    config = ErtConfig(
        ensemble_config=EnsembleConfig(
            parameter_configs={parameter.name: parameter} if has_parameters else {}
        )
    )

    parameter_configs, design_matrix_dict = _resolve_parameter_configs(
        config, prior_ensemble=str(uuid1())
    )

    assert parameter_configs == config.ensemble_config.parameter_configuration
    assert design_matrix_dict is None


@pytest.mark.filterwarnings("ignore:MIN_REALIZATIONS")
@pytest.mark.parametrize(
    "experiment_setup_method",
    [
        _setup_multiple_data_assimilation,
        _setup_ensemble_smoother,
        _setup_ensemble_information_filter,
    ],
)
def test_that_setting_up_experiment_with_update_step_raises_config_validation_error_given_no_parameters_configured(  # ruff: ignore[line-too-long]
    experiment_setup_method, tmp_path
):
    config = ErtConfig.from_file_contents(f"NUM_REALIZATIONS 100\nENSPATH {tmp_path}")
    args = Namespace(
        realizations="0-4",
        weights="2,3",
        target_ensemble="test_case_%d",
        prior_ensemble_id=None,
        experiment_name="experiment",
        starting_iteration=0,
    )

    with pytest.raises(
        ConfigValidationError,
        match="No parameters to update as no GEN_KW, FIELD or SURFACE "
        "parameters are configured!",
    ):
        experiment_setup_method(
            config, args, ObservationSettings(), queue.SimpleQueue()
        )
