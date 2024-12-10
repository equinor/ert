from __future__ import annotations

from typing import TYPE_CHECKING

from ert.config import ErtConfig
from ert.run_models import EnsembleExperiment

if TYPE_CHECKING:
    from queue import SimpleQueue

    from ert.storage import Storage

    from .base_run_model import StatusEvents

SINGLE_TEST_RUN_GROUP = "Forward model evaluation"


class SingleTestRun(EnsembleExperiment):
    """
    Single test is equivalent to EnsembleExperiment, in that it
    samples the prior and evaluates it.<br>There are two key differences:<br>
    1) Single test run always runs locally using the <b>local queue</b><br>
    2) Only a <b>single realization</b> (realization-0) is run<br>
    """

    def __init__(
        self,
        ensemble_name: str,
        experiment_name: str,
        random_seed: int | None,
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
        return "Sample parameters → evaluate single realization"

    @classmethod
    def group(cls) -> str | None:
        return SINGLE_TEST_RUN_GROUP
