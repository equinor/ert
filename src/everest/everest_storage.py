from __future__ import annotations

import logging
from pathlib import Path

from ert.storage import LocalExperiment, open_storage

logger = logging.getLogger(__name__)


class EverestStorage:
    @classmethod
    def get_everest_experiment(cls, storage_path: Path) -> LocalExperiment:
        """
        Creates everest storage from a storage path. Note: This
        requires there to be at least one initialized batch/ensemble
        for it to be possible to detect the experiment.
        """
        storage = open_storage(storage_path, mode="r")
        experiment = next(storage.experiments)
        assert isinstance(experiment, LocalExperiment)
        return experiment
