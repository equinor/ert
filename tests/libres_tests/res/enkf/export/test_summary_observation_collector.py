import pytest

from ert._c_wrappers.enkf.export import SummaryObservationCollector


def test_summary_observation_collector(monkeypatch, snake_oil_case):
    monkeypatch.setenv("TZ", "CET")  # The ert_statoil case was generated in CET

    ert = snake_oil_case

    assert SummaryObservationCollector.summaryKeyHasObservations(ert, "FOPR")
    assert not SummaryObservationCollector.summaryKeyHasObservations(ert, "FOPT")

    keys = SummaryObservationCollector.getAllObservationKeys(ert)
    assert "FOPR" in keys
    assert "WOPR:OP1" in keys
    assert "WOPR:OP2" not in keys

    data = SummaryObservationCollector.loadObservationData(ert, "default_0")

    assert pytest.approx(data["FOPR"]["2010-01-10"]) == 0.001696887
    assert pytest.approx(data["STD_FOPR"]["2010-01-10"]) == 0.1

    assert pytest.approx(data["WOPR:OP1"]["2010-03-31"]) == 0.1
    assert pytest.approx(data["STD_WOPR:OP1"]["2010-03-31"]) == 0.05

    # pylint: disable=pointless-statement
    with pytest.raises(KeyError):
        data["FGIR"]

    data = SummaryObservationCollector.loadObservationData(
        ert, "default_0", ["WOPR:OP1"]
    )

    assert pytest.approx(data["WOPR:OP1"]["2010-03-31"]) == 0.1
    assert pytest.approx(data["STD_WOPR:OP1"]["2010-03-31"]) == 0.05

    with pytest.raises(KeyError):
        data["FOPR"]
