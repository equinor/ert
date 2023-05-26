from typing import List

import pytest

from ert._c_wrappers.enkf import GenDataConfig


@pytest.mark.parametrize(
    "name, report_steps",
    [
        ("ORDERED_RESULTS", [1, 2, 3, 4]),
        ("UNORDERED_RESULTS", [5, 2, 3, 7, 1]),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_config(name: str, report_steps: List[int]):
    gdc = GenDataConfig(name=name, report_steps=report_steps)
    assert gdc.getKey() == name
    assert len(gdc.getReportSteps()) == len(report_steps)
    assert gdc.getReportSteps() == sorted(report_steps)
    for i in report_steps:
        assert gdc.hasReportStep(i)

    assert not gdc.hasReportStep(200)

    gen_data_default_step = GenDataConfig(name=name)
    assert len(gen_data_default_step.getReportSteps()) == 1
    assert gen_data_default_step.getReportSteps()[0] == 0


@pytest.mark.usefixtures("use_tmpdir")
def test_empty_gen_data_config():
    gdc = GenDataConfig(name="key")
    assert len(gdc.getReportSteps()) == 1
    assert gdc.hasReportStep(0)
    assert not gdc.hasReportStep(1)
    assert gdc.getReportSteps()[0] == 0


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_eq_config():
    alt1 = GenDataConfig(name="ALT1", report_steps=[2, 1, 3])
    alt2 = GenDataConfig(name="ALT1", report_steps=[2, 3, 1])
    alt3 = GenDataConfig(name="ALT1", report_steps=[3])
    alt4 = GenDataConfig(name="ALT4", report_steps=[3])
    alt5 = GenDataConfig(name="ALT4", report_steps=[4])
    alt6 = GenDataConfig(name="ALT4", report_steps=[4])

    obs_list = ["DEF", "ABC", "GHI"]
    alt6.update_observation_keys(obs_list)
    assert alt6.get_observation_keys() == sorted(obs_list)

    assert alt1 == alt2  # name and ordered steps ok
    assert alt1 != alt3  # amount steps differ
    assert alt3 != alt4  # name differ
    assert alt4 != alt5  # steps differ
    assert alt5 != alt6  # obs list differ
