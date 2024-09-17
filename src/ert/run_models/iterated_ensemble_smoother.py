from __future__ import annotations

import functools
import logging
from queue import SimpleQueue
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from iterative_ensemble_smoother import steplength_exponential

from ert.analysis import ErtAnalysisError, iterative_smoother_update
from ert.config import ErtConfig, HookRuntime
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Ensemble, Storage

from ..config.analysis_config import UpdateSettings
from ..config.analysis_module import IESSettings
from ..run_arg import create_run_arguments
from .base_run_model import BaseRunModel, ErtRunError, StatusEvents
from .event import RunModelStatusEvent, RunModelUpdateBeginEvent

if TYPE_CHECKING:
    from uuid import UUID

    import numpy.typing as npt

    from ert.config import QueueConfig


logger = logging.getLogger(__name__)


class IteratedEnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        target_ensemble: str,
        experiment_name: str,
        num_retries_per_iter: int,
        number_of_iterations: int,
        active_realizations: List[bool],
        minimum_required_realizations: int,
        random_seed: Optional[int],
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
        self.target_ensemble_format = target_ensemble
        self.experiment_name = experiment_name

        super().__init__(
            config,
            storage,
            queue_config,
            status_queue,
            active_realizations=active_realizations,
            total_iterations=number_of_iterations,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
        )

        # Initialize sies_smoother to None
        # It is initialized later, but kept track of here
        self.sies_smoother = None
        self.num_retries_per_iter = num_retries_per_iter

    @property
    def sies_iteration(self) -> int:
        """Returns the SIES iteration number, starting at 1."""
        if self.sies_smoother is None:
            return 1
        else:
            return self.sies_smoother.iteration

    def analyzeStep(
        self,
        prior_storage: Ensemble,
        posterior_storage: Ensemble,
        ensemble_id: UUID,
        iteration: int,
        initial_mask: npt.NDArray[np.bool_],
    ) -> None:
        self.validate()
        self.run_workflows(HookRuntime.PRE_UPDATE, self._storage, prior_storage)
        try:
            _, self.sies_smoother = iterative_smoother_update(
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
        self.run_workflows(HookRuntime.POST_UPDATE, self._storage, posterior_storage)

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()

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
        prior_args = create_run_arguments(
            self.run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=prior,
        )

        sample_prior(
            prior,
            np.where(self.active_realizations)[0],
            random_seed=self.random_seed,
        )
        self._evaluate_and_postprocess(
            prior_args,
            prior,
            evaluator_server_config,
        )

        self.run_workflows(HookRuntime.PRE_FIRST_UPDATE, self._storage, prior)
        for prior_iter in range(self._total_iterations):
            self.send_event(
                RunModelUpdateBeginEvent(iteration=prior_iter, run_id=prior.id)
            )
            self.send_event(
                RunModelStatusEvent(
                    iteration=prior_iter,
                    run_id=prior.id,
                    msg="Creating posterior ensemble..",
                )
            )

            posterior = self._storage.create_ensemble(
                experiment,
                name=target_ensemble_format % (prior_iter + 1),
                ensemble_size=prior.ensemble_size,
                iteration=prior_iter + 1,
                prior_ensemble=prior,
            )
            posterior_args = create_run_arguments(
                self.run_paths,
                self.active_realizations,
                ensemble=posterior,
            )
            update_success = False
            for _iteration in range(self.num_retries_per_iter):
                self.analyzeStep(
                    prior_storage=prior,
                    posterior_storage=posterior,
                    ensemble_id=prior.id,
                    iteration=prior_iter,
                    initial_mask=initial_mask,
                )

                # sies iteration starts at 1, we keep iters at 0,
                # so we subtract sies to be 0-indexed
                analysis_success = prior_iter < (self.sies_iteration - 1)
                if analysis_success:
                    update_success = True
                    break
                self._evaluate_and_postprocess(
                    prior_args,
                    prior,
                    evaluator_server_config,
                )

            if update_success:
                self._evaluate_and_postprocess(
                    posterior_args,
                    posterior,
                    evaluator_server_config,
                )
            else:
                raise ErtRunError(
                    (
                        "Iterated ensemble smoother stopped: "
                        "maximum number of iteration retries "
                        f"({self.num_retries_per_iter} retries) reached "
                        f"for iteration {prior_iter}"
                    )
                )
            prior = posterior

    @classmethod
    def name(cls) -> str:
        return "Iterated ensemble smoother"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → [Evaluate → update] for N iterations"
