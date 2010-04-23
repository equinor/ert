# ----------------------------------------------------------------------------------------------
# Plot tab
# ----------------------------------------------------------------------------------------------
from widgets.pathchooser import PathChooser
from widgets.combochoice import ComboChoice
from widgets.spinnerwidgets import IntegerSpinner

import ertwrapper

def createPlotPage(configPanel, parent):
    configPanel.startPage("Plot")

    r = configPanel.addRow(PathChooser(parent, "Output path", "config/plot/path"))
    r.initialize = lambda ert : [ert.setTypes("plot_config_get_path", ertwrapper.c_char_p),
                                 ert.setTypes("plot_config_set_path", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.plot_config_get_path(ert.plot_config)
    r.setter = lambda ert, value : ert.enkf.plot_config_set_path(ert.plot_config, str(value))

    r = configPanel.addRow(ComboChoice(parent, ["PLPLOT", "TEXT"], "Driver", "plot_driver"))
    r.initialize = lambda ert : [ert.setTypes("plot_config_get_driver", ertwrapper.c_char_p),
                                 ert.setTypes("plot_config_set_driver", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.plot_config_get_driver(ert.plot_config)
    r.setter = lambda ert, value : ert.enkf.plot_config_set_driver(ert.plot_config, str(value))

    r = configPanel.addRow(IntegerSpinner(parent, "Errorbar max", "plot_errorbar_max", 1, 10000000))
    r.initialize = lambda ert : [ert.setTypes("plot_config_get_errorbar_max", ertwrapper.c_int),
                                 ert.setTypes("plot_config_set_errorbar_max", None, [ertwrapper.c_int])]
    r.getter = lambda ert : ert.enkf.plot_config_get_errorbar_max(ert.plot_config)
    r.setter = lambda ert, value : ert.enkf.plot_config_set_errorbar_max(ert.plot_config, value)

    r = configPanel.addRow(IntegerSpinner(parent, "Width", "config/plot/width", 1, 10000))
    r.initialize = lambda ert : [ert.setTypes("plot_config_get_width", ertwrapper.c_int),
                                 ert.setTypes("plot_config_set_width", None, [ertwrapper.c_int])]
    r.getter = lambda ert : ert.enkf.plot_config_get_width(ert.plot_config)
    r.setter = lambda ert, value : ert.enkf.plot_config_set_width(ert.plot_config, value)

    r = configPanel.addRow(IntegerSpinner(parent, "Height", "plot_height", 1, 10000))
    r.initialize = lambda ert : [ert.setTypes("plot_config_get_height", ertwrapper.c_int),
                                 ert.setTypes("plot_config_set_height", None, [ertwrapper.c_int])]
    r.getter = lambda ert : ert.enkf.plot_config_get_height(ert.plot_config)
    r.setter = lambda ert, value : ert.enkf.plot_config_set_height(ert.plot_config, value)

    r = configPanel.addRow(PathChooser(parent, "Image Viewer", "image_viewer", True))
    r.initialize = lambda ert : [ert.setTypes("plot_config_get_viewer", ertwrapper.c_char_p),
                                 ert.setTypes("plot_config_set_viewer", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.plot_config_get_viewer(ert.plot_config)
    r.setter = lambda ert, value : ert.enkf.plot_config_set_viewer(ert.plot_config, str(value))

    r = configPanel.addRow(ComboChoice(parent, ["bmp", "jpg", "png", "tif"], "Image type", "image_type"))
    r.initialize = lambda ert : [ert.setTypes("plot_config_get_image_type", ertwrapper.c_char_p),
                                 ert.setTypes("plot_config_set_image_type", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.plot_config_get_image_type(ert.plot_config)
    r.setter = lambda ert, value : ert.enkf.plot_config_set_image_type(ert.plot_config, str(value))


    configPanel.endPage()