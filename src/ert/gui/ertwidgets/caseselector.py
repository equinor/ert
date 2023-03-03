from qtpy.QtWidgets import QComboBox

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import addHelpToWidget
from ert.libres_facade import LibresFacade
from ert.storage import StorageAccessor


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
        self.notifier = notifier
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

        self.currentIndexChanged[int].connect(self.on_current_index_changed)
        notifier.ertChanged.connect(self.populate)
        notifier.storage_changed.connect(self.populate)

        if notifier._storage is not None:
            self.populate()

    @property
    def storage(self) -> StorageAccessor:
        return self.notifier.storage

    def on_current_index_changed(self, index: int) -> None:
        if self._update_ert:
            assert (
                0 <= index < self.count()
            ), f"Should not happen! Index out of range: 0 <= {index} < {self.count()}"

            self.notifier.set_current_case(self.itemData(index))

    def populate(self):
        block = self.signalsBlocked()
        self.blockSignals(True)

        if self._show_only_initialized:
            case_list = (x for x in self.storage.ensembles if x.is_initalized)
        else:
            case_list = self.storage.ensembles

        case_list = sorted(case_list, key=lambda x: x.started_at, reverse=True)

        self.clear()

        for case in case_list:
            self.addItem(case.name, userData=case)

        current_index = 0
        current_case = self.notifier.current_case
        if current_case in case_list:
            current_index = case_list.index(current_case)

        if current_index != self.currentIndex() and not self._ignore_current:
            self.setCurrentIndex(current_index)

        self.blockSignals(block)
