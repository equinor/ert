import pytest

from ert_utils import ErtTest

from ert_shared.models import BaseRunModel
from res.test import ErtTestContext


class BaseRunModelTest(ErtTest):
    def test_instantiation(self):
        config_file = self.createTestPath("local/simple_config/minimum_config")
        with ErtTestContext("kjell", config_file) as work_area:
            ert = work_area.getErt()
            brm = BaseRunModel(ert, ert.get_queue_config())
            assert brm.support_restart


class MockJob:
    def __init__(self, status):
        self.status = status


"""
NOTE: The test below is for the old job-status mechanism no
longer used in ERT. The mechanism is, however, used in Everest
so we leave this test in place.

If Everest at some point updates how it tracks job-status, this
test as well as BaseRunModel.is_forward_model_finished() can be
removed.
"""


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([MockJob("Success")], True),
        ([MockJob("Failure")], False),
        ([MockJob("Success"), MockJob("Success")], True),
        ([MockJob("Failure"), MockJob("Success")], False),
    ],
)
def test_is_forward_model_finished(test_input, expected):
    assert BaseRunModel.is_forward_model_finished(test_input) is expected
