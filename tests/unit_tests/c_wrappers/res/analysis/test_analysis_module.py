import pytest

from ert._c_wrappers.analysis import AnalysisModule


def test_analysis_module():
    am = AnalysisModule.iterated_ens_smother_module()

    am.set_var("ENKF_TRUNCATION", "1.0")

    assert am.name == "IES_ENKF"

    assert isinstance(am.get_variable_value("ENKF_TRUNCATION"), float)
    assert am.get_variable_value("ENKF_TRUNCATION") == 1.0
    with pytest.raises(ValueError):
        am.set_var("ENKF_TRUNCATION", "super1.0")


def test_set_get_var():
    mod = AnalysisModule.ens_smother_module()

    with pytest.raises(KeyError):
        mod.set_var("NO-NOT_THIS_KEY", 100)

    with pytest.raises(KeyError):
        mod.get_variable_value("NO-NOT_THIS_KEY")


def test_analysis_module_copy():
    import copy

    am = AnalysisModule.iterated_ens_smother_module()
    print(am.name)
    print(am.get_variable_value("ENKF_TRUNCATION"))
    print(am.get_variable_value("IES_MAX_STEPLENGTH"))
    print(am.get_variable_value("IES_MIN_STEPLENGTH"))
    print(am.get_variable_value("IES_DEC_STEPLENGTH"))
    # am.setVar("IES_DEC_STEPLENGTH", 1)
    print(am.get_variable_value("IES_DEC_STEPLENGTH"))
    print("IES_INVERSION")
    t = am.inversion
    print("t")
    print(t)
    # print(am.get_variable_value("ENKF_SUBSPACE_DIMENSION"))
    am.set_var("INVERSION", "SUBSPACE_RE")
