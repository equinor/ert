from qtpy.QtWidgets import QComboBox

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import addHelpToWidget
from ert.libres_facade import LibresFacade


class CaseSelector(QComboBox):
    def __init__(
        self,
        facade: LibresFacade,
        notifier: ErtNotifier,
        update_ert: bool = True,
        show_only_initialized: bool = False,
        ignore_current: bool = False,
        help_link: str = "init/current_case_selection",
    ):
        self.facade = facade
        QComboBox.__init__(self)
        self._update_ert = update_ert  # If true current case of ert will be change
        self._show_only_initialized = (
            show_only_initialized  # only show initialized cases
        )
        self._ignore_current = (
            ignore_current  # ignore the currently selected case if it changes
        )

        addHelpToWidget(self, help_link)
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.populate()

        self.currentIndexChanged[int].connect(self.selectionChanged)
        notifier.ertChanged.connect(self.populate)

    def _getAllCases(self):
        if self._show_only_initialized:
            return [
                case
                for case in self.facade.cases()
                if self.facade.case_initialized(case)
            ]
        else:
            return self.facade.cases()

    def selectionChanged(self, index):
        if self._update_ert:
            assert (
                0 <= index < self.count()
            ), f"Should not happen! Index out of range: 0 <= {index} < {self.count()}"

            item = self._getAllCases()[index]
            self.facade.select_or_create_new_case(item)

    def populate(self):
        block = self.signalsBlocked()
        self.blockSignals(True)

        case_list = self._getAllCases()
        self.clear()

        for case in case_list:
            self.addItem(case)

        current_index = 0
        current_case = self.facade.get_current_case_name()
        if current_case in case_list:
            current_index = case_list.index(current_case)

        if current_index != self.currentIndex() and not self._ignore_current:
            self.setCurrentIndex(current_index)

        self.blockSignals(block)
