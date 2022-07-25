from res.enkf import CancelPluginException, ErtPlugin

from ...libres_utils import ResTest


class SimplePlugin(ErtPlugin):
    def run(self, parameter1, parameter2):
        assert parameter1 == "one"
        assert parameter2 == 2

    def getArguments(self, parent=None):
        return ["one", 2]


class FullPlugin(ErtPlugin):
    def getName(self):
        return "FullPlugin"

    def getDescription(self):
        return "Fully described!"

    def run(self, arg1, arg2, arg3=None):
        assert arg1 == 5
        assert arg2 == "No"
        assert arg3 is None

    def getArguments(self, parent=None):
        return [5, "No"]


class CanceledPlugin(ErtPlugin):
    def run(self, arg1):
        pass

    def getArguments(self, parent=None):
        raise CancelPluginException("Cancel test!")


class ErtPluginTest(ResTest):
    def test_simple_ert_plugin(self):

        simple_plugin = SimplePlugin("ert")

        arguments = simple_plugin.getArguments()

        self.assertTrue("SimplePlugin" in simple_plugin.getName())
        self.assertEqual("No description provided!", simple_plugin.getDescription())

        simple_plugin.initializeAndRun([str, int], arguments)

    def test_full_ert_plugin(self):
        plugin = FullPlugin("ert")

        self.assertEqual(plugin.getName(), "FullPlugin")
        self.assertEqual(plugin.getDescription(), "Fully described!")

        arguments = plugin.getArguments()

        plugin.initializeAndRun([int, str, float], arguments)

    def test_cancel_plugin(self):
        plugin = CanceledPlugin("ert")

        with self.assertRaises(CancelPluginException):
            plugin.getArguments()

    # def test_gui_ert_plugin(self):
    #     app = QApplication([])
    #     plugin = GUIPlugin("ert")
    #
    #     arguments = plugin.getArguments()
    #     plugin.initializeAndRun([int, int], arguments)
