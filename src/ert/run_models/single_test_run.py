from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from ert.config import ErtConfig
from ert.run_models import EnsembleExperiment

if TYPE_CHECKING:
    from queue import SimpleQueue

    from ert.run_models.run_arguments import SingleTestRunArguments
    from ert.storage import Storage

    from .base_run_model import StatusEvents


class SingleTestRun(EnsembleExperiment):
    simulation_arguments: SingleTestRunArguments

    def __init__(
        self,
        simulation_arguments: SingleTestRunArguments,
        config: ErtConfig,
        storage: Storage,
        status_queue: SimpleQueue[StatusEvents],
    ):
        local_queue_config = config.queue_config.create_local_copy()
        super().__init__(
            simulation_arguments, config, storage, local_queue_config, status_queue
        )

    @override
    @classmethod
    def run_message(cls) -> str:
        return "Running single realisation test ..."

    @classmethod
    def name(cls) -> str:
        return "Single realization test-run"
