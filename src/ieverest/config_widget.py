from io import StringIO

from qtpy.QtWidgets import QWidget
from ruamel.yaml import YAML

from everest.config import EverestConfig
from ieverest import utils


class ConfigWidget(QWidget):
    """A widget for setting up an Everest configuration"""

    def __init__(self, parent=None):
        super(ConfigWidget, self).__init__(parent)
        utils.load_ui("config_widget.ui", self)

        self._config = None

    def set_config(self, config: EverestConfig):
        """Set the config to be exposed by this widget."""
        self._config = config
        yaml = YAML(typ="safe", pure=True)
        yaml.default_flow_style = False
        with StringIO() as sio:
            yaml.dump(self._config.to_dict(), sio)
            self.config_edit.setPlainText(sio.getvalue())
