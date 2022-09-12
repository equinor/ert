import pytest

from ert._c_wrappers.enkf.export import MisfitCollector


def test_misfit_collector(snake_oil_case):
    ert = snake_oil_case
    data = MisfitCollector.loadAllMisfitData(ert, "default_0")

    assert pytest.approx(data["MISFIT:FOPR"][0]) == 738.735586
    assert pytest.approx(data["MISFIT:FOPR"][24]) == 1260.086789

    assert pytest.approx(data["MISFIT:TOTAL"][0]) == 767.008457
    assert pytest.approx(data["MISFIT:TOTAL"][24]) == 1359.172803

    # pylint: disable=pointless-statement
    # realization 20:
    data.loc[20]

    with pytest.raises(KeyError):
        # realization 60:
        data.loc[60]
