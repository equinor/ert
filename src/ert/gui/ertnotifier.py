from typing import Optional

from qtpy.QtCore import QObject, Signal, Slot

from ert.storage import EnsembleReader, StorageReader


class ErtNotifier(QObject):
    ertChanged = Signal()
    storage_changed = Signal(object, name="storageChanged")
    current_case_changed = Signal(object, name="currentCaseChanged")

    def __init__(self, config_file: str):
        QObject.__init__(self)
        self._config_file = config_file
        self._storage: Optional[StorageReader] = None
        self._current_case = None

    @property
    def storage(self) -> StorageReader:
        assert self._storage is not None
        return self._storage

    @property
    def config_file(self) -> str:
        return self._config_file

    @property
    def current_case(self) -> Optional[EnsembleReader]:
        if self._current_case is None and self._storage is not None:
            ensembles = list(self._storage.ensembles)
            if ensembles:
                self._current_case = ensembles[0]
        return self._current_case

    @property
    def current_case_name(self) -> str:
        if self._current_case is None:
            return "default"
        return self._current_case.name

    @Slot()
    def emitErtChange(self):
        self.ertChanged.emit()

    @Slot(object)
    def set_storage(self, storage: StorageReader) -> None:
        self._storage = storage
        self.storage_changed.emit(storage)

    @Slot(object)
    def set_current_case(self, case: Optional[EnsembleReader] = None) -> None:
        self._current_case = case
        self.current_case_changed.emit(case)
