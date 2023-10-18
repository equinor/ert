import pytest

from ert.config import (
    AnalysisMode,
    AnalysisModule,
    ConfigValidationError,
)
from ert.config.analysis_module import (
    DEFAULT_ENKF_TRUNCATION,
    DEFAULT_IES_DEC_STEPLENGTH,
    DEFAULT_IES_INVERSION,
    DEFAULT_IES_MAX_STEPLENGTH,
    DEFAULT_IES_MIN_STEPLENGTH,
    DEFAULT_LOCALIZATION,
    DEFAULT_LOCALIZATION_CORRELATION_THRESHOLD,
    get_mode_variables,
)


def test_analysis_module_default_values():
    ies_am = AnalysisModule.iterated_ens_smoother_module()

    # Check default values in dict
    assert ies_am.variable_value_dict() == {
        "IES_MAX_STEPLENGTH": DEFAULT_IES_MAX_STEPLENGTH,
        "IES_MIN_STEPLENGTH": DEFAULT_IES_MIN_STEPLENGTH,
        "IES_DEC_STEPLENGTH": DEFAULT_IES_DEC_STEPLENGTH,
        "IES_INVERSION": DEFAULT_IES_INVERSION,
        "ENKF_TRUNCATION": DEFAULT_ENKF_TRUNCATION,
        "LOCALIZATION": DEFAULT_LOCALIZATION,
        "LOCALIZATION_CORRELATION_THRESHOLD": DEFAULT_LOCALIZATION_CORRELATION_THRESHOLD,  # noqa
    }

    es_am = AnalysisModule.ens_smoother_module()
    assert es_am.variable_value_dict() == {
        "IES_INVERSION": DEFAULT_IES_INVERSION,
        "ENKF_TRUNCATION": DEFAULT_ENKF_TRUNCATION,
        "LOCALIZATION": DEFAULT_LOCALIZATION,
        "LOCALIZATION_CORRELATION_THRESHOLD": DEFAULT_LOCALIZATION_CORRELATION_THRESHOLD,  # noqa
    }


def test_analysis_module_set_get_values():
    ies_am = AnalysisModule.iterated_ens_smoother_module()
    ies_am.set_var("ENKF_TRUNCATION", "0.1")
    assert ies_am.get_variable_value("ENKF_TRUNCATION") == 0.1
    assert ies_am.get_truncation() == 0.1

    ies_am.set_var("IES_INVERSION", "2")
    assert ies_am.get_variable_value("IES_INVERSION") == 2

    ies_am.set_var("IES_MAX_STEPLENGTH", "0.9")
    assert ies_am.get_variable_value("IES_MAX_STEPLENGTH") == 0.9

    ies_am.set_var("IES_MIN_STEPLENGTH", 0.33)
    assert ies_am.get_variable_value("IES_MIN_STEPLENGTH") == 0.33
    ies_am.set_var("IES_DEC_STEPLENGTH", 2.1)
    assert ies_am.get_variable_value("IES_DEC_STEPLENGTH") == 2.1

    # Special keys
    expected_inversion = {
        "EXACT": 0,
        "SUBSPACE_EXACT_R": 1,
        "SUBSPACE_EE_R": 2,
        "SUBSPACE_RE": 3,
    }
    for key, value in expected_inversion.items():
        ies_am.set_var("INVERSION", key)
        assert ies_am.get_variable_value("IES_INVERSION") == value

    ies_am.set_var("ENKF_NCOMP", "0.2")
    assert ies_am.get_truncation() == 0.2
    ies_am.set_var("ENKF_SUBSPACE_DIMENSION", "0.3")
    assert ies_am.get_truncation() == 0.3


def test_set_get_var_errors():
    mod = AnalysisModule.ens_smoother_module()

    with pytest.raises(ConfigValidationError):
        mod.set_var("ENKF_TRUNCATION", "super1.0")

    with pytest.raises(ConfigValidationError):
        mod.set_var("NO-NOT_THIS_KEY", 100)

    with pytest.raises(ConfigValidationError):
        mod.get_variable_value("NO-NOT_THIS_KEY")


def test_set_get_var_out_of_bounds():
    ies_variables = get_mode_variables(AnalysisMode.ITERATED_ENSEMBLE_SMOOTHER)
    ies = AnalysisModule.iterated_ens_smoother_module()
    enkf_trunc_max = ies_variables["ENKF_TRUNCATION"]["max"]
    enkf_trunc_min = ies_variables["ENKF_TRUNCATION"]["min"]
    ies.set_var("ENKF_TRUNCATION", enkf_trunc_max + 1)
    assert ies.get_truncation() == enkf_trunc_max

    ies.set_var("ENKF_TRUNCATION", enkf_trunc_min - 1)
    assert ies.get_truncation() == enkf_trunc_min

    ies.set_var("ENKF_NCOMP", enkf_trunc_max + 1)
    assert ies.get_truncation() == enkf_trunc_max

    ies.set_var("ENKF_NCOMP", enkf_trunc_min - 1)
    assert ies.get_truncation() == enkf_trunc_min

    ies.set_var("ENKF_SUBSPACE_DIMENSION", enkf_trunc_max + 1)
    assert ies.get_truncation() == enkf_trunc_max
    ies.set_var("ENKF_SUBSPACE_DIMENSION", enkf_trunc_min - 1)
    assert ies.get_truncation() == enkf_trunc_min

    ies.set_var("IES_INVERSION", 5)
    assert ies_variables["IES_INVERSION"]["max"] == 3
    assert ies.inversion == ies_variables["IES_INVERSION"]["max"]

    ies.set_var("IES_INVERSION", -1)
    assert ies_variables["IES_INVERSION"]["min"] == 0
    assert ies.inversion == ies_variables["IES_INVERSION"]["min"]
