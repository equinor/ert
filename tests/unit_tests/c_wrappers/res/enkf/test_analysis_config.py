from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import AnalysisConfig, ConfigKeys, ErtConfig
from ert.parsing import ConfigValidationError


@pytest.fixture
def analysis_config(use_tmpdir):
    with open("analysis_config", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 10
        NUM_REALIZATIONS 10

        ANALYSIS_COPY     STD_ENKF    ENKF_HIGH_TRUNCATION
        ANALYSIS_SET_VAR  STD_ENKF     ENKF_NCOMP    2
        ANALYSIS_SET_VAR  ENKF_HIGH_TRUNCATION  ENKF_TRUNCATION 0.99
        ANALYSIS_SELECT   ENKF_HIGH_TRUNCATION

        QUEUE_SYSTEM LOCAL
        QUEUE_OPTION LOCAL MAX_RUNNING 50
        """
            )
        )
    return ErtConfig.from_file("analysis_config").analysis_config


def test_keywords_for_monitoring_simulation_runtime(minimum_case):
    analysis_config = minimum_case.analysisConfig()
    # Unless the MIN_REALIZATIONS is set in config, one is required to
    # have "all" realizations.
    assert not analysis_config.have_enough_realisations(5)
    assert analysis_config.have_enough_realisations(10)

    assert analysis_config.get_max_runtime() == 42

    analysis_config.set_max_runtime(50)
    assert analysis_config.get_max_runtime() == 50

    analysis_config.set_stop_long_running(True)
    assert analysis_config.get_stop_long_running()


def test_analysis_config_constructor(analysis_config):
    assert analysis_config == AnalysisConfig.from_dict(
        config_dict={
            ConfigKeys.NUM_REALIZATIONS: 10,
            ConfigKeys.ALPHA_KEY: 3,
            ConfigKeys.UPDATE_LOG_PATH: "update_log",
            ConfigKeys.STD_CUTOFF_KEY: 1e-6,
            ConfigKeys.STOP_LONG_RUNNING: False,
            ConfigKeys.MAX_RUNTIME: 0,
            ConfigKeys.MIN_REALIZATIONS: 10,
            ConfigKeys.ANALYSIS_COPY: [
                (
                    "STD_ENKF",
                    "ENKF_HIGH_TRUNCATION",
                )
            ],
            ConfigKeys.ANALYSIS_SET_VAR: [
                (
                    "STD_ENKF",
                    "ENKF_NCOMP",
                    2,
                ),
                (
                    "ENKF_HIGH_TRUNCATION",
                    "ENKF_TRUNCATION",
                    0.99,
                ),
            ],
            ConfigKeys.ANALYSIS_SELECT: "ENKF_HIGH_TRUNCATION",
        }
    )


@pytest.mark.parametrize(
    "num_realization, min_realizations, expected_min_real",
    [
        (80, "10%", 8),
        (5, "2%", 1),
        (8, "50%", 4),
        (8, "100%", 8),
        (8, "100%", 8),
        (8, "10%", 1),
        (80, 0, 80),
        (80, 0, 80),
        (900, None, 900),
        (900, 10, 10),
        (80, 50, 50),
        (80, 80, 80),
        (80, 100, 80),
    ],
)
def test_analysis_config_min_realizations(
    num_realization, min_realizations, expected_min_real
):
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: num_realization,
    }
    if min_realizations is not None:
        config_dict[ConfigKeys.MIN_REALIZATIONS] = min_realizations

    analysis_config = AnalysisConfig.from_dict(config_dict)

    assert analysis_config.minimum_required_realizations == expected_min_real


def test_invalid_min_realization():
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 1,
        ConfigKeys.MIN_REALIZATIONS: "1s",
    }

    with pytest.raises(
        ConfigValidationError, match="MIN_REALIZATIONS value is not integer"
    ):
        AnalysisConfig.from_dict(config_dict)


def test_analysis_config_stop_long_running():
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 10,
    }
    analysis_config = AnalysisConfig.from_dict(config_dict)
    assert not analysis_config.get_stop_long_running()
    analysis_config.set_stop_long_running(True)
    assert analysis_config.get_stop_long_running()


def test_analysis_config_alpha():
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 10,
    }
    analysis_config = AnalysisConfig.from_dict(config_dict)
    assert analysis_config.get_enkf_alpha() == 3.0
    analysis_config.set_enkf_alpha(42.0)
    assert analysis_config.get_enkf_alpha() == 42.0

    config_dict[ConfigKeys.ALPHA_KEY] = 24
    new_analysis_config = AnalysisConfig.from_dict(config_dict)
    assert new_analysis_config.get_enkf_alpha() == 24.0


def test_analysis_config_std_cutoff():
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 10,
    }
    analysis_config = AnalysisConfig.from_dict(config_dict)
    assert analysis_config.get_std_cutoff() == 1e-06
    analysis_config.set_std_cutoff(42.0)
    assert analysis_config.get_std_cutoff() == 42.0

    config_dict[ConfigKeys.STD_CUTOFF_KEY] = 24
    new_analysis_config = AnalysisConfig.from_dict(config_dict)
    assert new_analysis_config.get_std_cutoff() == 24.0


def test_analysis_config_iter_config():
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 10,
    }
    analysis_config = AnalysisConfig.from_dict(config_dict)

    assert analysis_config.case_format is None
    assert analysis_config.case_format_is_set() is False

    expected_case_format = "case_%d"
    analysis_config.set_case_format(expected_case_format)
    assert analysis_config.case_format_is_set() is True
    assert analysis_config.case_format == expected_case_format

    assert analysis_config.num_iterations == 4
    analysis_config.set_num_iterations(42)
    assert analysis_config.num_iterations == 42

    assert analysis_config.num_retries_per_iter == 4


def test_analysis_config_iter_config_dict_initialisation():
    expected_case_format = "case_%d"
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 10,
        ConfigKeys.ITER_CASE: expected_case_format,
        ConfigKeys.ITER_COUNT: 42,
        ConfigKeys.ITER_RETRY_COUNT: 24,
    }
    analysis_config = AnalysisConfig.from_dict(config_dict)

    assert analysis_config.case_format_is_set() is True
    assert analysis_config.case_format == expected_case_format
    assert analysis_config.num_iterations == 42
    assert analysis_config.num_retries_per_iter == 24


def test_analysis_config_modules():
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 10,
    }
    analysis_config = AnalysisConfig.from_dict(config_dict)
    default_modules = analysis_config.get_module_list()
    assert len(default_modules) == 2
    assert "IES_ENKF" in default_modules
    assert "STD_ENKF" in default_modules

    assert analysis_config.active_module_name() == "STD_ENKF"
    assert analysis_config.get_active_module().name == "STD_ENKF"

    assert analysis_config.select_module("IES_ENKF")

    assert analysis_config.get_active_module().name == "IES_ENKF"
    assert analysis_config.active_module_name() == "IES_ENKF"

    es_module = analysis_config.get_module("STD_ENKF")
    assert es_module.name == "STD_ENKF"
    with pytest.raises(
        ConfigValidationError, match="Analysis module UNKNOWN not found!"
    ):
        analysis_config.get_module("UNKNOWN")
    assert not analysis_config.select_module("UNKNOWN")


def test_analysis_config_iter_config_default_initialisation():
    config = AnalysisConfig()
    expected_case_format = "case_%d"
    config.set_case_format(expected_case_format)
    assert config.case_format_is_set() is True
    assert config.case_format == expected_case_format
    config.set_num_iterations(42)
    assert config.num_iterations == 42
    assert config.num_retries_per_iter == 4

    new_config = AnalysisConfig()
    assert new_config.num_iterations == 4
    assert new_config.case_format is None
    assert new_config.case_format_is_set() is False


def test_analysis_config_wrong_argument_type():
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 10,
        ConfigKeys.ANALYSIS_SET_VAR: [["STD_ENKF", "IES_INVERSION", "FOO"]],
    }

    with pytest.raises(
        ConfigValidationError, match="Variable 'IES_INVERSION' with value 'FOO'"
    ):
        _ = AnalysisConfig.from_dict(config_dict)


def test_analysis_config_wrong_unknown_argument():
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 10,
        ConfigKeys.ANALYSIS_SET_VAR: [["STD_ENKF", "BAR", "1"]],
    }

    with pytest.raises(
        ConfigValidationError, match="Variable 'BAR' not found in 'STD_ENKF' analysis"
    ):
        _ = AnalysisConfig.from_dict(config_dict)
