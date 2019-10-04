import os
import datetime

import pandas as pd

from tests import ErtTest
from res.test import ErtTestContext
from ert_gui.plottery.plot_data_gatherer import PlotDataGatherer
from ert_shared import ERT, EnkfFacade
from ert_shared.cli import ErtCliNotifier


class PlotGatherTest(ErtTest):

    def test_gatherSummaryRefcaseData(self):
        config_file = self.createTestPath(os.path.join("local", "snake_oil", "snake_oil.ert"))
        with ErtTestContext('SummaryRefcaseData', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            facade = EnkfFacade(ert)
            ERT.adapt(notifier, facade)

            key = "WOPRH:OP1"
            result_data = PlotDataGatherer.gatherSummaryRefcaseData(key)

            expected_data = [
                (0, datetime.date(2010,1,2),       1.03836009657e-05),
                (244, datetime.date(2010, 9, 3),   0.46973800659),
                (1267, datetime.date(2013, 6, 22), 0.11672365665),
                (-1, datetime.date(2015, 6, 23),   0.00820410997),
            ]

            self.assertEqual(len(result_data), 1999)

            for index, date, value in expected_data:
                self.assertAlmostEqual(value, result_data.iloc[index][key], delta=1E-10)
                self.assertEqual(date, result_data.iloc[index].name.date())

            key = "not_a_key"

            result_data = PlotDataGatherer.gatherSummaryRefcaseData(key)
            expected_data = pd.DataFrame()

            pd.testing.assert_frame_equal(result_data, expected_data, check_exact=True)
