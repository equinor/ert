from __future__ import annotations

import functools
import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING

import numpy as np
from iterative_ensemble_smoother import steplength_exponential

from ert.analysis import ErtAnalysisError, iterative_smoother_update
from ert.config import ErtConfig, HookRuntime
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_context import RunContext
from ert.run_models.run_arguments import SIESRunArguments
from ert.storage import Ensemble, Storage

from ..config.analysis_config import UpdateSettings
from ..config.analysis_module import IESSettings
from .base_run_model import BaseRunModel, ErtRunError, StatusEvents
from .event import RunModelStatusEvent, RunModelUpdateBeginEvent

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.config import QueueConfig


logger = logging.getLogger(__file__)


class IteratedEnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: SIESRunArguments,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        analysis_config: IESSettings,
        update_settings: UpdateSettings,
        status_queue: SimpleQueue[StatusEvents],
    ):
        self.support_restart = False
        self.analysis_config = analysis_config
        self.update_settings = update_settings

        self.sies_step_length = functools.partial(
            steplength_exponential,
            min_steplength=analysis_config.ies_min_steplength,
            max_steplength=analysis_config.ies_max_steplength,
            halflife=analysis_config.ies_dec_steplength,
        )
        self.target_ensemble_format = simulation_arguments.target_ensemble
        self.experiment_name = simulation_arguments.experiment_name
        self.ensemble_size = simulation_arguments.ensemble_size

        super().__init__(
            config,
            storage,
            queue_config,
            status_queue,
            active_realizations=simulation_arguments.active_realizations,
            phase_count=2,
            number_of_iterations=simulation_arguments.number_of_iterations,
            random_seed=simulation_arguments.random_seed,
            minimum_required_realizations=simulation_arguments.minimum_required_realizations,
        )

        # Initialize sies_smoother to None
        # It is initialized later, but kept track of here
        self.sies_smoother = None
        self.num_retries_per_iter = simulation_arguments.num_retries_per_iter

    @property
    def iteration(self) -> int:
        """Returns the SIES iteration number, starting at 1."""
        if self.sies_smoother is None:
            return 1
        else:
            return self.sies_smoother.iteration

    def analyzeStep(
        self,
        prior_storage: Ensemble,
        posterior_storage: Ensemble,
        ensemble_id: str,
        iteration: int,
        initial_mask: npt.NDArray[np.bool_],
    ) -> None:
        self.setPhaseName("Pre processing update...")
        self.run_workflows(HookRuntime.PRE_UPDATE, self._storage, prior_storage)
        try:
            smoother_snapshot, self.sies_smoother = iterative_smoother_update(
                prior_storage,
                posterior_storage,
                self.sies_smoother,
                parameters=prior_storage.experiment.update_parameters,
                observations=prior_storage.experiment.observations.keys(),
                update_settings=self.update_settings,
                analysis_config=self.analysis_config,
                sies_step_length=self.sies_step_length,
                initial_mask=initial_mask,
                rng=self.rng,
                progress_callback=functools.partial(
                    self.send_smoother_event, iteration, ensemble_id
                ),
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Update algorithm failed with the following error: {e}"
            ) from e

        self.setPhaseName("Post processing update...")
        self.run_workflows(HookRuntime.POST_UPDATE, self._storage, posterior_storage)

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> RunContext:
        iteration_count = self.number_of_iterations
        phase_count = iteration_count + 1
        self.setPhaseCount(phase_count)

        log_msg = (
            f"Running SIES for {iteration_count} "
            f'iteration{"s" if (iteration_count != 1) else ""}.'
        )
        logger.info(log_msg)
        self.setPhaseName(log_msg)

        target_ensemble_format = self.target_ensemble_format
        experiment = self._storage.create_experiment(
            parameters=self.ert_config.ensemble_config.parameter_configuration,
            observations=self.ert_config.observations,
            responses=self.ert_config.ensemble_config.response_configuration,
            name=self.experiment_name,
        )
        prior = self._storage.create_ensemble(
            experiment=experiment,
            ensemble_size=self.ensemble_size,
            name=target_ensemble_format % 0,
        )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))

        initial_mask = np.array(self.active_realizations, dtype=bool)
        prior_context = RunContext(
            ensemble=prior,
            runpaths=self.run_paths,
            initial_mask=initial_mask,
            iteration=0,
        )

        sample_prior(
            prior_context.ensemble,
            prior_context.active_realizations,
            random_seed=self.random_seed,
        )
        self._evaluate_and_postprocess(prior_context, evaluator_server_config)

        self.run_workflows(
            HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_context.ensemble
        )
        for current_iter in range(1, iteration_count + 1):
            self.send_event(
                RunModelUpdateBeginEvent(
                    iteration=current_iter - 1, run_id=prior_context.run_id
                )
            )
            self.send_event(
                RunModelStatusEvent(
                    iteration=current_iter - 1,
                    run_id=prior_context.run_id,
                    msg="Creating posterior ensemble..",
                )
            )

            posterior = self._storage.create_ensemble(
                experiment,
                name=target_ensemble_format % current_iter,  # noqa
                ensemble_size=prior_context.ensemble.ensemble_size,
                iteration=current_iter,
                prior_ensemble=prior_context.ensemble,
            )
            posterior_context = RunContext(
                ensemble=posterior,
                runpaths=self.run_paths,
                initial_mask=(
                    prior_context.ensemble.get_realization_mask_with_parameters()
                    * prior_context.ensemble.get_realization_mask_with_responses()
                    * prior_context.ensemble.get_realization_mask_without_failure()
                ),
                iteration=current_iter,
            )
            update_success = False
            for _iteration in range(self.num_retries_per_iter):
                self.analyzeStep(
                    prior_storage=prior_context.ensemble,
                    posterior_storage=posterior_context.ensemble,
                    ensemble_id=str(prior_context.ensemble.id),
                    iteration=current_iter - 1,
                    initial_mask=initial_mask,
                )

                analysis_success = current_iter < self.iteration
                if analysis_success:
                    update_success = True
                    break
                self._evaluate_and_postprocess(prior_context, evaluator_server_config)

            if update_success:
                self._evaluate_and_postprocess(
                    posterior_context, evaluator_server_config
                )
            else:
                raise ErtRunError(
                    (
                        "Iterated ensemble smoother stopped: "
                        "maximum number of iteration retries "
                        f"({self.num_retries_per_iter} retries) reached "
                        f"for iteration {current_iter}"
                    )
                )
            prior_context = posterior_context

        self.setPhase(phase_count, "Experiment completed.")

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Iterated ensemble smoother"
