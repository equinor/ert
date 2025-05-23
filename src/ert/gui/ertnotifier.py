from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot

from ert.storage import Ensemble, Storage, open_storage


class ErtNotifier(QObject):
    ertChanged = Signal()
    simulationStarted = Signal()
    simulationEnded = Signal()
    current_ensemble_changed = Signal(object, name="currentEnsembleChanged")

    def __init__(self) -> None:
        QObject.__init__(self)
        self._storage: Storage | None = None
        self._storage_path: str | None = None
        self._current_ensemble_id: UUID | None = None
        self._is_simulation_running = False

    @property
    def is_storage_available(self) -> bool:
        return self._storage is not None

    @property
    def storage(self) -> Storage:
        assert self.is_storage_available
        return self._storage  # type: ignore

    @contextmanager
    def write_storage(self) -> Iterator[Storage]:
        assert self._storage_path is not None
        storage = open_storage(self._storage_path, mode="w")
        try:
            yield storage
        finally:
            # Assume some writing was done when opening a
            # _write_ storage
            self.emitErtChange()
            storage.close()

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
        if self._storage is not None:
            self._storage.refresh()

        self.ertChanged.emit()

    @Slot(object)
    def set_storage(self, storage_path: str) -> None:
        self._storage = open_storage(storage_path, mode="r")
        self._storage_path = storage_path
        self.emitErtChange()

    @Slot(object)
    def set_current_ensemble_id(self, ensemble_id: UUID | None = None) -> None:
        if ensemble_id != self._current_ensemble_id:
            self._current_ensemble_id = ensemble_id
            self.current_ensemble_changed.emit(ensemble_id)

    @Slot(bool)
    def set_is_simulation_running(self, is_running: bool) -> None:
        self._is_simulation_running = is_running
        if self._is_simulation_running:
            self.simulationStarted.emit()
        else:
            self.simulationEnded.emit()
