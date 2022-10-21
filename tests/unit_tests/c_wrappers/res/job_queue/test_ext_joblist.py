from ert._c_wrappers.job_queue import ExtJoblist


def test_that_eq_works_with_non_homogenous_lists():
    assert [1, 2, 3, ExtJoblist()].index(ExtJoblist()) == 3
