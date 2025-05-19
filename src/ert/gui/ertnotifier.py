from uuid import UUID

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot

from ert.storage import Ensemble, Storage


class ErtNotifier(QObject):
    ertChanged = Signal()
    storage_changed = Signal(object, name="storageChanged")
    current_ensemble_changed = Signal(object, name="currentEnsembleChanged")

    def __init__(self) -> None:
        QObject.__init__(self)
        self._storage: Storage | None = None
        self._current_ensemble_id: UUID | None = None
        self._is_simulation_running = False

    @property
    def is_storage_available(self) -> bool:
        return self._storage is not None

    @property
    def storage(self) -> Storage:
        assert self.is_storage_available
        return self._storage  # type: ignore

    @property
    def current_ensemble(self) -> Ensemble | None:
        if self._storage is None:
            return None

        if self._current_ensemble_id is None:
            ensembles = list(self._storage.ensembles)
            if ensembles:
                self._current_ensemble_id = ensembles[0].id

        if self._current_ensemble_id is None:
            return None

        try:
            return self._storage.get_ensemble(self._current_ensemble_id)
        except KeyError:
            return None

    @property
    def current_ensemble_name(self) -> str:
        if self.current_ensemble is None:
            return "default"
        return self.current_ensemble.name

    @property
    def is_simulation_running(self) -> bool:
        return self._is_simulation_running

    @Slot()
    def emitErtChange(self) -> None:
        self.ertChanged.emit()

    @Slot(object)
    def set_storage(self, storage: Storage) -> None:
        self._storage = storage
        self.storage_changed.emit(storage)

    @Slot(object)
    def set_current_ensemble_id(self, ensemble_id: UUID | None = None) -> None:
        self._current_ensemble_id = ensemble_id
        self.current_ensemble_changed.emit(ensemble_id)

    @Slot(bool)
    def set_is_simulation_running(self, is_running: bool) -> None:
        self._is_simulation_running = is_running
