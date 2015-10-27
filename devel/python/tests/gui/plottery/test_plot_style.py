from ert.test import ExtendedTestCase
from ert_gui.plottery import PlotStyle


class PlotStyleTest(ExtendedTestCase):

    def test_plot_style_test_defaults(self):
        style = PlotStyle("Test")

        self.assertEqual(style.name, "Test")
        self.assertEqual(style.color, "#000000")
        self.assertEqual(style.line_style, "-")
        self.assertEqual(style.alpha, 1.0)
        self.assertEqual(style.marker, "")
        self.assertEqual(style.width, 1.0)
        self.assertTrue(style.isEnabled())


    def test_plot_style_builtin_checks(self):
        style = PlotStyle("Test")

        style.name = None
        self.assertIsNone(style.name)

        style.color = "notacolor"
        self.assertEqual(style.color, "notacolor") # maybe make this a proper check in future ?

        style.line_style = None
        self.assertEqual(style.line_style, "")

        style.marker = None
        self.assertEqual(style.marker, "")

        style.width = -1
        self.assertEqual(style.width, 0.0)

        style.alpha = 1.1
        self.assertEqual(style.alpha, 1.0)

        style.alpha = -0.1
        self.assertEqual(style.alpha, 0.0)

        style.setEnabled(False)
        self.assertFalse(style.isEnabled())


    def test_plot_style_copy_style(self):
        style = PlotStyle("Test", "red", 0.5, ".", "o", 2.5)
        style.setEnabled(False)

        copy_style = PlotStyle("Copy")

        copy_style.copyStyleFrom(style)

        self.assertNotEqual(style.name, copy_style.name)
        self.assertEqual(style.color, copy_style.color)
        self.assertEqual(style.alpha, copy_style.alpha)
        self.assertEqual(style.line_style, copy_style.line_style)
        self.assertEqual(style.marker, copy_style.marker)
        self.assertEqual(style.width, copy_style.width)
        self.assertNotEqual(style.isEnabled(), copy_style.isEnabled())