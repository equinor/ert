from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.libres_facade import LibresFacade


class TargetCaseModel(ValueModel):
    def __init__(
        self, facade: LibresFacade, notifier: ErtNotifier, format_mode: bool = False
    ):
        self.facade = facade
        self._format_mode = format_mode
        self._custom = False
        ValueModel.__init__(self, self.getDefaultValue())
        notifier.ertChanged.connect(self._caseChanged)

    def setValue(self, value: str):
        """Set a new target case"""
        if value is None or value.strip() == "" or value == self.getDefaultValue():
            self._custom = False
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            self._custom = True
            ValueModel.setValue(self, value)

    def getDefaultValue(self) -> str:
        if self._format_mode:
            analysis_config = self.facade.get_analysis_config()
            if analysis_config.case_format_is_set():
                return analysis_config.case_format
            else:
                case_name = self.facade.get_current_case_name()
                return f"{case_name}_%d"
        else:
            case_name = self.facade.get_current_case_name()
            return f"{case_name}_smoother_update"

    def _caseChanged(self):
        if not self._custom:
            ValueModel.setValue(self, self.getDefaultValue())
