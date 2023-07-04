from typing import List

import pytest

from ert.config import GenDataConfig


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
    assert gdc.name == name
    assert gdc.report_steps == sorted(report_steps)


def test_gen_data_default_report_step():
    gen_data_default_step = GenDataConfig(name="name")
    assert gen_data_default_step.report_steps == [0]


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_eq_config():
    alt1 = GenDataConfig(name="ALT1", report_steps=[2, 1, 3])
    alt2 = GenDataConfig(name="ALT1", report_steps=[2, 3, 1])
    alt3 = GenDataConfig(name="ALT1", report_steps=[3])
    alt4 = GenDataConfig(name="ALT4", report_steps=[3])
    alt5 = GenDataConfig(name="ALT4", report_steps=[4])

    assert alt1 == alt2  # name and ordered steps ok
    assert alt1 != alt3  # amount steps differ
    assert alt3 != alt4  # name differ
    assert alt4 != alt5  # steps differ
