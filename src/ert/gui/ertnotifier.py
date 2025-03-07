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
        self._current_ensemble: Ensemble | None = None
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
        if self._current_ensemble is None and self._storage is not None:
            ensembles = list(self._storage.ensembles)
            if ensembles:
                self._current_ensemble = ensembles[0]
        return self._current_ensemble

    @property
    def current_ensemble_name(self) -> str:
        if self._current_ensemble is None:
            return "default"
        return self._current_ensemble.name

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
    def set_current_ensemble(self, ensemble: Ensemble | None = None) -> None:
        self._current_ensemble = ensemble
        self.current_ensemble_changed.emit(ensemble)

    @Slot(bool)
    def set_is_simulation_running(self, is_running: bool) -> None:
        self._is_simulation_running = is_running
