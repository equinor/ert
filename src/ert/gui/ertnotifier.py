from typing import List, Optional

from qtpy.QtCore import QObject, Signal, Slot

from ert.storage import LocalEnsemble, LocalStorage


class ErtNotifier(QObject):
    ertChanged = Signal()
    storage_changed = Signal(object, name="storageChanged")
    current_ensemble_changed = Signal(object, name="currentEnsembleChanged")

    def __init__(self, config_file: str):
        QObject.__init__(self)
        self._config_file = config_file
        self._storage: Optional[LocalStorage] = None
        self._current_ensemble: Optional[LocalEnsemble] = None
        self._is_simulation_running = False

    @property
    def is_storage_available(self) -> bool:
        return self._storage is not None

    @property
    def storage(self) -> LocalStorage:
        assert self.is_storage_available
        return self._storage  # type: ignore

    @property
    def config_file(self) -> str:
        return self._config_file

    @property
    def current_ensemble(self) -> Optional[LocalEnsemble]:
        if self._current_ensemble is None and self.is_storage_available:
            all_ensembles = self.get_all_ensembles()
            if all_ensembles:
                self._current_ensemble = all_ensembles[0]
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
    def set_storage(self, storage: LocalStorage) -> None:
        self._storage = storage
        self._current_ensemble = None
        self.storage_changed.emit(storage)

    @Slot(object)
    def set_current_ensemble(self, ensemble: Optional[LocalEnsemble] = None) -> None:
        self._current_ensemble = ensemble
        self.current_ensemble_changed.emit(ensemble)

    @Slot(bool)
    def set_is_simulation_running(self, is_running: bool) -> None:
        self._is_simulation_running = is_running

    def get_all_ensembles(self) -> List[LocalEnsemble]:
        if not self.is_storage_available:
            return []
        all_ensembles = []
        for experiment in self.storage.experiments:
            all_ensembles.extend(list(experiment.ensembles))
        return all_ensembles
