from ert.test import ExtendedTestCase
from ert_gui.plottery import PlotStyle, PlotConfig


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

        another_copy_style = PlotStyle("Another Copy")
        another_copy_style.copyStyleFrom(style, copy_enabled_state=True)
        self.assertEqual(style.isEnabled(), another_copy_style.isEnabled())


    def test_plot_config(self):
        plot_config = PlotConfig("Golden Sample", x_label="x", y_label="y")

        plot_config.setDistributionLineEnabled(True)
        plot_config.setLegendEnabled(False)
        plot_config.setGridEnabled(False)
        plot_config.setRefcaseEnabled(False)
        plot_config.setObservationsEnabled(False)
        plot_config.deactivateDateSupport()

        plot_config.setDefaultStyle(".", "g")
        plot_config.setRefcaseStyle(".", "g")
        plot_config.setStatisticsStyle("mean", ".", "g")
        plot_config.setStatisticsStyle("min-max", ".", "g")
        plot_config.setStatisticsStyle("p50", ".", "g")
        plot_config.setStatisticsStyle("p10-p90", ".", "g")
        plot_config.setStatisticsStyle("p33-p67", ".", "g")

        copy_of_plot_config = PlotConfig("Copy of Golden Sample")
        copy_of_plot_config.copyConfigFrom(plot_config)

        self.assertEqual(plot_config.isLegendEnabled(), copy_of_plot_config.isLegendEnabled())
        self.assertEqual(plot_config.isGridEnabled(), copy_of_plot_config.isGridEnabled())
        self.assertEqual(plot_config.isObservationsEnabled(), copy_of_plot_config.isObservationsEnabled())
        self.assertEqual(plot_config.isDistributionLineEnabled(), copy_of_plot_config.isDistributionLineEnabled())
        self.assertEqual(plot_config.isDateSupportActive(), copy_of_plot_config.isDateSupportActive())

        self.assertEqual(plot_config.refcaseStyle(), copy_of_plot_config.refcaseStyle())
        self.assertEqual(plot_config.observationsStyle(), copy_of_plot_config.observationsStyle())

        self.assertEqual(plot_config.histogramStyle(), copy_of_plot_config.histogramStyle())
        self.assertEqual(plot_config.defaultStyle(), copy_of_plot_config.defaultStyle())
        self.assertEqual(plot_config.currentColor(), copy_of_plot_config.currentColor())

        self.assertEqual(plot_config.getStatisticsStyle("mean"), copy_of_plot_config.getStatisticsStyle("mean"))
        self.assertEqual(plot_config.getStatisticsStyle("min-max"), copy_of_plot_config.getStatisticsStyle("min-max"))
        self.assertEqual(plot_config.getStatisticsStyle("p50"), copy_of_plot_config.getStatisticsStyle("p50"))
        self.assertEqual(plot_config.getStatisticsStyle("p10-p90"), copy_of_plot_config.getStatisticsStyle("p10-p90"))
        self.assertEqual(plot_config.getStatisticsStyle("p33-p67"), copy_of_plot_config.getStatisticsStyle("p33-p67"))

        self.assertEqual(plot_config.title(), copy_of_plot_config.title())


        plot_config.currentColor()  # cycle state will not be copied
        plot_config.nextColor()

        copy_of_plot_config = PlotConfig("Another Copy of Golden Sample")
        copy_of_plot_config.copyConfigFrom(plot_config)

        self.assertEqual(plot_config.refcaseStyle(), copy_of_plot_config.refcaseStyle())
        self.assertEqual(plot_config.observationsStyle(), copy_of_plot_config.observationsStyle())

        self.assertNotEqual(plot_config.histogramStyle(), copy_of_plot_config.histogramStyle())
        self.assertNotEqual(plot_config.defaultStyle(), copy_of_plot_config.defaultStyle())
        self.assertNotEqual(plot_config.currentColor(), copy_of_plot_config.currentColor())

        self.assertNotEqual(plot_config.getStatisticsStyle("mean"), copy_of_plot_config.getStatisticsStyle("mean"))
        self.assertNotEqual(plot_config.getStatisticsStyle("min-max"), copy_of_plot_config.getStatisticsStyle("min-max"))
        self.assertNotEqual(plot_config.getStatisticsStyle("p50"), copy_of_plot_config.getStatisticsStyle("p50"))
        self.assertNotEqual(plot_config.getStatisticsStyle("p10-p90"), copy_of_plot_config.getStatisticsStyle("p10-p90"))
        self.assertNotEqual(plot_config.getStatisticsStyle("p33-p67"), copy_of_plot_config.getStatisticsStyle("p33-p67"))





