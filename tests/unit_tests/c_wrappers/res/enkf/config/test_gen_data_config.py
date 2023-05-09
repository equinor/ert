from typing import List

import pytest

from ert._c_wrappers.enkf import GenDataConfig


@pytest.mark.parametrize(
    "key, active_list",
    [
        ("ORDERED_RESULTS", [1, 2, 3, 4]),
        ("NO_RESULTS", []),
        ("UNORDERED_RESULTS", [5, 2, 3, 7, 1]),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_config(key: str, active_list: List[int]):
    gdc = GenDataConfig(key=key, report_steps=active_list)
    assert gdc.getKey() == key
    assert gdc.getNumReportStep() == len(active_list)
    assert gdc.getReportSteps() == sorted(active_list)
    for i in active_list:
        assert gdc.hasReportStep(i)

    assert not gdc.hasReportStep(200)


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_eq_config():
    alt1 = GenDataConfig(key="ALT1", report_steps=[2, 1, 3])
    alt2 = GenDataConfig(key="ALT1", report_steps=[2, 3, 1])
    alt3 = GenDataConfig(key="ALT1", report_steps=[3])
    alt4 = GenDataConfig(key="ALT4", report_steps=[3])
    alt5 = GenDataConfig(key="ALT4", report_steps=[4])
    assert alt1 == alt2  # name and ordered steps ok
    assert alt1 != alt3  # amount steps differ
    assert alt3 != alt4  # name differ
    assert alt4 != alt5  # steps differ
