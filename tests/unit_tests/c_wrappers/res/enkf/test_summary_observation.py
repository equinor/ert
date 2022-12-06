from ert._c_wrappers.enkf import SummaryObservation


def test_create():
    sum_obs = SummaryObservation("WWCT:OP_X", "WWCT:OP_X", 0.25, 0.12)

    assert sum_obs.getValue() == 0.25
    assert sum_obs.getStandardDeviation() == 0.12
    assert sum_obs.getStdScaling() == 1.0
