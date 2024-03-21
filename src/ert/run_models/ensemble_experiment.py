from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_context import RunContext
from ert.storage import Storage

from .base_run_model import BaseRunModel

if TYPE_CHECKING:
    from ert.config import ErtConfig, QueueConfig
    from ert.run_models.run_arguments import (
        EnsembleExperimentRunArguments,
        SingleTestRunArguments,
    )


class EnsembleExperiment(BaseRunModel):
    """
    This workflow will create a new experiment and a new ensemble from
    the user configuration. It will never overwrite existing ensembles, and
    will always sample parameters.
    """

    _simulation_arguments: Union[SingleTestRunArguments, EnsembleExperimentRunArguments]

    def __init__(
        self,
        simulation_arguments: Union[
            SingleTestRunArguments,
            EnsembleExperimentRunArguments,
        ],
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
    ):
        super().__init__(
            simulation_arguments,
            config,
            storage,
            queue_config,
        )

    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> RunContext:
        self.setPhaseName(self.run_message(), indeterminate=False)
        experiment = self._storage.create_experiment(
            name=self.simulation_arguments.experiment_name,
            parameters=self.ert_config.ensemble_config.parameter_configuration,
            observations=self.ert_config.observations,
            simulation_arguments=self._simulation_arguments,
            responses=self.ert_config.ensemble_config.response_configuration,
        )
        ensemble = self._storage.create_ensemble(
            experiment,
            name=self._simulation_arguments.current_ensemble,
            ensemble_size=self._simulation_arguments.ensemble_size,
            iteration=self._simulation_arguments.iter_num,
        )
        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble.id))

        prior_context = RunContext(
            ensemble=ensemble,
            runpaths=self.run_paths,
            initial_mask=np.array(
                self._simulation_arguments.active_realizations, dtype=bool
            ),
            iteration=self._simulation_arguments.iter_num,
        )
        sample_prior(
            prior_context.ensemble,
            prior_context.active_realizations,
            random_seed=self._simulation_arguments.random_seed,
        )

        iteration = prior_context.iteration
        phase_count = iteration + 1
        self.setPhaseCount(phase_count)
        self._evaluate_and_postprocess(prior_context, evaluator_server_config)

        self.setPhase(phase_count, "Simulations completed.")

        return prior_context

    @classmethod
    def run_message(cls) -> str:
        return "Running ensemble experiment..."

    @classmethod
    def name(cls) -> str:
        return "Ensemble experiment"

    def check_if_runpath_exists(self) -> bool:
        iteration = self._simulation_arguments.iter_num
        active_mask = self._simulation_arguments.active_realizations
        active_realizations = [i for i in range(len(active_mask)) if active_mask[i]]
        run_paths = self.run_paths.get_paths(active_realizations, iteration)
        return any(Path(run_path).exists() for run_path in run_paths)
