from ert.test import ExtendedTestCase, ErtTestContext
from ert.enkf import PlotSettings
from ert.enkf import EnkfPrototype

_plot_settings_init = EnkfPrototype("void plot_settings_init( plot_settings, config_content)", bind = False)
_plot_settings_config = EnkfPrototype("void plot_settings_add_config_items( config_parser)", bind = False)



class PlotSettingsTest(ExtendedTestCase):


    def test_create(self):
        pass
        
