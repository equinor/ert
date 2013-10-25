from ert_gui.ide.keywords.definitions import BoolArgument
from ert_tests import ExtendedTestCase


class BoolArgumentTest(ExtendedTestCase):



    def test_bool_argument(self):
        bool = BoolArgument()

        validation_status = bool.validate("TRUE")
        self.assertTrue(validation_status)
        self.assertTrue(validation_status.value())
        self.assertEqual(validation_status.message(), "")

        validation_status = bool.validate("FALSE")
        self.assertTrue(validation_status)
        self.assertFalse(validation_status.value())
        self.assertEqual(validation_status.message(), "")

        validation_status = bool.validate("True")
        self.assertFalse(validation_status)
        self.assertEqual(validation_status.message(), bool.NOT_BOOL)

        validation_status = bool.validate("False")
        self.assertFalse(validation_status)
        self.assertEqual(validation_status.message(), bool.NOT_BOOL)

        validation_status = bool.validate(" FALSE")
        self.assertFalse(validation_status)
        self.assertEqual(validation_status.message(), bool.NOT_BOOL)










