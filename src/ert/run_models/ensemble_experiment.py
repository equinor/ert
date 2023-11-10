from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union
from uuid import UUID

import numpy as np

from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.realization_state import RealizationState
from ert.run_context import RunContext
from ert.storage import EnsembleAccessor, StorageAccessor

from .base_run_model import BaseRunModel

if TYPE_CHECKING:
    from ert.config import ErtConfig, QueueConfig
    from ert.run_models.run_arguments import (
        EnsembleExperimentRunArguments,
        SingleTestRunArguments,
    )


# pylint: disable=too-many-arguments
class EnsembleExperiment(BaseRunModel):
    _simulation_arguments: Union[SingleTestRunArguments, EnsembleExperimentRunArguments]

    def __init__(
        self,
        simulation_arguments: Union[
            SingleTestRunArguments, EnsembleExperimentRunArguments
        ],
        config: ErtConfig,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        id_: UUID,
    ):
        super().__init__(
            simulation_arguments,
            config,
            storage,
            queue_config,
            id_,
        )

    def runSimulations__(
        self,
        run_msg: str,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> RunContext:
        self.setPhaseName(run_msg, indeterminate=False)
        current_case = self._simulation_arguments.current_case
        try:
            ensemble = self._storage.get_ensemble_by_name(current_case)
            assert isinstance(ensemble, EnsembleAccessor)
        except KeyError:
            ensemble = self._storage.create_ensemble(
                self._experiment_id,
                name=current_case,
                ensemble_size=self._simulation_arguments.ensemble_size,
                iteration=self._simulation_arguments.iter_num,
            )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble.id))

        prior_context = RunContext(
            sim_fs=ensemble,
            runpaths=self.run_paths,
            initial_mask=np.array(
                self._simulation_arguments.active_realizations, dtype=bool
            ),
            iteration=self._simulation_arguments.iter_num,
        )

        if not prior_context.sim_fs.realizations_initialized(
            prior_context.active_realizations
        ):
            sample_prior(
                prior_context.sim_fs,
                prior_context.active_realizations,
                random_seed=self._simulation_arguments.random_seed,
            )
        else:
            state_map = prior_context.sim_fs.state_map
            for realization_nr in prior_context.active_realizations:
                if state_map[realization_nr] in [
                    RealizationState.UNDEFINED,
                    RealizationState.LOAD_FAILURE,
                ]:
                    state_map[realization_nr] = RealizationState.INITIALIZED
        iteration = prior_context.iteration
        phase_count = iteration + 1
        self.setPhaseCount(phase_count)
        self._evaluate_and_postprocess(prior_context, evaluator_server_config)

        self.setPhase(phase_count, "Simulations completed.")

        return prior_context

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        return self.runSimulations__(
            "Running ensemble experiment...", evaluator_server_config
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble experiment"

    def check_if_runpath_exists(self) -> bool:
        iteration = self._simulation_arguments.iter_num
        active_mask = self._simulation_arguments.active_realizations
        active_realizations = [i for i in range(len(active_mask)) if active_mask[i]]
        run_paths = self.run_paths.get_paths(active_realizations, iteration)
        return any(Path(run_path).exists() for run_path in run_paths)
