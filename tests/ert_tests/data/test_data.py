from ert_utils import ErtTest

from res.test import ErtTestContext

# The data loaded in this test is not part of an automatic test, we
# just verify that the example cases can indeed be loaded.


class TestData(ErtTest):
    def test_poly(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("poly", config_file) as work_area:
            work_area.getErt()
