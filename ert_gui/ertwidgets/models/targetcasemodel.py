from ert_gui.ertnotifier import ErtNotifier
from ert_gui.ertwidgets.models.valuemodel import ValueModel
from ert_shared.libres_facade import LibresFacade


class TargetCaseModel(ValueModel):
    def __init__(
        self, facade: LibresFacade, notifier: ErtNotifier, format_mode: bool = False
    ):
        self.facade = facade
        self._format_mode = format_mode
        self._custom = False
        ValueModel.__init__(self, self.getDefaultValue())
        notifier.ertChanged.connect(self._caseChanged)

    def setValue(self, target_case):
        if (
            target_case is None
            or target_case.strip() == ""
            or target_case == self.getDefaultValue()
        ):
            self._custom = False
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            self._custom = True
            ValueModel.setValue(self, target_case)

    def getDefaultValue(self):
        """@rtype: str"""
        if self._format_mode:
            if (
                self.facade.get_analysis_config()
                .getAnalysisIterConfig()
                .caseFormatSet()
            ):
                return (
                    self.facade.get_analysis_config()
                    .getAnalysisIterConfig()
                    .getCaseFormat()
                )
            else:
                case_name = self.facade.get_current_case_name()
                return "%s_%%d" % case_name
        else:
            case_name = self.facade.get_current_case_name()
            return "%s_smoother_update" % case_name

    def _caseChanged(self):
        if not self._custom:
            ValueModel.setValue(self, self.getDefaultValue())
