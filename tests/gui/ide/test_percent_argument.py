from ert_gui.ide.keywords.definitions import PercentArgument
from tests import ErtTest


class PercentArgumentTest(ErtTest):

    def test_percent_range_argument(self):
        from_value = 10
        to_value = 20
        percent = PercentArgument(from_value=from_value, to_value=to_value)

        validation_status = percent.validate("%d%%" % to_value)
        self.assertTrue(validation_status)

        validation_status = percent.validate("%d%%" % from_value)
        self.assertTrue(validation_status)

        validation_status = percent.validate("%d%%" % 15)
        self.assertTrue(validation_status)

        value = 9
        validation_status = percent.validate("%d%%" % value)
        self.assertFalse(validation_status)

        range_string = "%g%% <= %g%% <= %g%%" % (from_value, value, to_value)
        self.assertEqual(validation_status.message(), PercentArgument.NOT_IN_RANGE % range_string)

        value = 21
        validation_status = percent.validate("%d%%" % value)
        self.assertFalse(validation_status)

        range_string = "%g%% <= %g%% <= %g%%" % (from_value, value, to_value)
        self.assertEqual(validation_status.message(), PercentArgument.NOT_IN_RANGE % range_string)
