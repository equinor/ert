from os import path

from qtpy import uic

import ieverest.io.output_dispatcher

UI_DIRECTORY = path.join(path.dirname(path.realpath(__file__)), "ui")


def ui_real_path(ui_file):
    return path.realpath(path.join(UI_DIRECTORY, ui_file))


def load_ui(ui_file, baseinstance=None):
    return uic.loadUi(ui_real_path(ui_file), baseinstance)


def remove_layout_item(layout, index):
    """Remove the item at @index from the @layout

    If the index is out of range, the behaviour is undefined.
    """
    item = layout.takeAt(index)

    removed_widget = item.widget()
    if removed_widget is not None:
        removed_widget.setParent(None)
        return
    layout = item.layout()
    if layout is not None:
        layout.setParent(None)
    # item must be a QSpacerItem, which does not have a parent


APP_OUT_LOGGER = "everest_gui_logger"
APP_OUT_DIALOGS = "everest_dialogs"
APP_OUT_STATUS_BAR = "everest_status_bar"


def app_output():
    """Return an OutputDispatcher unique for the current application"""
    return _ApplicationOutput()


class _ApplicationOutputSingleton(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_ApplicationOutputSingleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instance


class _ApplicationOutput(ieverest.io.output_dispatcher.OutputDispatcher):
    __metaclass__ = _ApplicationOutputSingleton

    def __init__(self):
        super(_ApplicationOutput, self).__init__()
