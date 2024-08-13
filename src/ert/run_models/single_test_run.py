from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ert.config import ErtConfig
from ert.run_models import EnsembleExperiment

if TYPE_CHECKING:
    from queue import SimpleQueue

    from ert.storage import Storage

    from .base_run_model import StatusEvents


class SingleTestRun(EnsembleExperiment):
    """
    Single test is equivalent to EnsembleExperiment, in that it
    samples the prior and evaluates it. There are two key differences:
    1) Single test run always runs locally using the local queue
    2) Only a single realization is run
    """

    def __init__(
        self,
        ensemble_name: str,
        experiment_name: str,
        random_seed: Optional[int],
        config: ErtConfig,
        storage: Storage,
        status_queue: SimpleQueue[StatusEvents],
    ):
        local_queue_config = config.queue_config.create_local_copy()
        super().__init__(
            ensemble_name=ensemble_name,
            experiment_name=experiment_name,
            active_realizations=[True],
            minimum_required_realizations=1,
            config=config,
            storage=storage,
            queue_config=local_queue_config,
            status_queue=status_queue,
            random_seed=random_seed,
        )

    @classmethod
    def name(cls) -> str:
        return "Single realization test-run"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate (one realization)"
