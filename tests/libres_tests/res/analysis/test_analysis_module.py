import pytest

from ert._c_wrappers.analysis import AnalysisModule


def test_analysis_module():
    am = AnalysisModule(2)

    assert am.setVar("ENKF_TRUNCATION", "1.0")

    assert am.name() == "IES_ENKF"

    assert am.hasVar("IES_INVERSION")

    assert isinstance(am.getDouble("ENKF_TRUNCATION"), float)


def test_set_get_var():
    mod = AnalysisModule(1)

    with pytest.raises(KeyError):
        mod.setVar("NO-NOT_THIS_KEY", 100)

    with pytest.raises(KeyError):
        mod.getInt("NO-NOT_THIS_KEY")
