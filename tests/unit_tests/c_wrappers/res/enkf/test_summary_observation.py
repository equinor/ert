from ert._c_wrappers.enkf import ActiveList, SummaryObservation


def test_create():
    sum_obs = SummaryObservation("WWCT:OP_X", "WWCT:OP_X", 0.25, 0.12)

    assert sum_obs.getValue() == 0.25
    assert sum_obs.getStandardDeviation() == 0.12
    assert sum_obs.getStdScaling() == 1.0


def test_std_scaling():
    sum_obs = SummaryObservation("WWCT:OP_X", "WWCT:OP_X", 0.25, 0.12)

    active_list = ActiveList()
    sum_obs.updateStdScaling(0.50, active_list)
    sum_obs.updateStdScaling(0.125, active_list)
    assert sum_obs.getStdScaling() == 0.125
