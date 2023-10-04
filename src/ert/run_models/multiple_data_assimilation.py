from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from uuid import UUID

from ert.analysis import ErtAnalysisError
from ert.config import HookRuntime
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.realization_state import RealizationState
from ert.storage import EnsembleAccessor, StorageAccessor

from .base_run_model import BaseRunModel, ErtRunError, SimulationArguments

if TYPE_CHECKING:
    from ert.config import QueueConfig
    from ert.enkf_main import EnKFMain
    from ert.run_context import RunContext

logger = logging.getLogger(__file__)


# pylint: disable=too-many-arguments
class MultipleDataAssimilation(BaseRunModel):
    """
    Run multiple data assimilation (MDA) ensemble smoother with custom weights.
    """

    default_weights = "4, 2, 1"

    def __init__(
        self,
        simulation_arguments: SimulationArguments,
        ert: EnKFMain,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        experiment_id: UUID,
        prior_ensemble: Optional[EnsembleAccessor],
    ):
        super().__init__(
            simulation_arguments,
            ert,
            storage,
            queue_config,
            experiment_id,
            phase_count=2,
        )
        self.weights = MultipleDataAssimilation.default_weights
        self.prior_ensemble = prior_ensemble

    async def run(self, _: EvaluatorServerConfig) -> None:
        raise NotImplementedError()

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        self._checkMinimumActiveRealizations(
            self._args.realizations_mask.count(True)
        )
        if not self._args.weights:
            raise ErtRunError(
                "Operation halted: ES-MDA requires weights to proceed. "
                "Please provide appropriate weights and try again."
            )

        weights = self.normalizeWeights(self._args.weights)
        iteration_count = len(weights)

        phase_count = iteration_count + 1
        self.setPhaseCount(phase_count)

        log_msg = f"Running ES-MDA with normalized weights {weights}"
        logger.info(log_msg)
        self.setPhaseName(log_msg, indeterminate=True)

        enumerated_weights = list(enumerate(weights))
        restart_run = self._args.restart_run
        target_case_format = self._args.target_case
        assert target_case_format is not None

        if restart_run:
            prior_ensemble = self._args.prior_ensemble
            assert prior_ensemble is not None
            try:
                prior = self._storage.get_ensemble_by_name(prior_ensemble)
                self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
                assert isinstance(prior, EnsembleAccessor)
                prior_context = self.ert().ensemble_context(
                    prior,
                    self._args.realizations_mask,
                    iteration=prior.iteration,
                )
            except KeyError as err:
                raise ErtRunError(
                    f"Prior ensemble: {prior_ensemble} does not exists"
                ) from err
        else:
            prior = self._storage.create_ensemble(
                self._experiment_id,
                ensemble_size=self._ert.getEnsembleSize(),
                iteration=0,
                name=target_case_format % 0,
            )
            self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
            prior_context = self.ert().ensemble_context(
                prior,
                self._args.realizations_mask,
                iteration=prior.iteration,
            )
            self.ert().sample_prior(
                prior_context.sim_fs, prior_context.active_realizations
            )
            self._evaluate_and_postprocess(prior_context, evaluator_server_config)
        starting_iteration = prior.iteration + 1
        weights_to_run = enumerated_weights[max(starting_iteration - 1, 0) :]

        for iteration, weight in weights_to_run:
            is_first_iteration = iteration == 0
            if is_first_iteration:
                self.ert().runWorkflows(
                    HookRuntime.PRE_FIRST_UPDATE, self._storage, prior
                )
            self.ert().runWorkflows(HookRuntime.PRE_UPDATE, self._storage, prior)
            states = [
                RealizationState.HAS_DATA,
                RealizationState.INITIALIZED,
            ]
            posterior_context = self.ert().ensemble_context(
                self._storage.create_ensemble(
                    self._experiment_id,
                    name=target_case_format % (iteration + 1),  # noqa
                    ensemble_size=self._ert.getEnsembleSize(),
                    iteration=iteration + 1,
                    prior_ensemble=prior_context.sim_fs,
                ),
                prior_context.sim_fs.get_realization_mask_from_state(states),
                iteration=iteration + 1,
            )
            self.update(
                prior_context,
                posterior_context,
                weight=weight,
            )
            self.ert().runWorkflows(
                HookRuntime.POST_UPDATE, self._storage, posterior_context.sim_fs
            )
            self._evaluate_and_postprocess(posterior_context, evaluator_server_config)
            prior_context = posterior_context

        self.setPhaseName("Post processing...", indeterminate=True)

        self.setPhase(iteration_count + 1, "Experiment completed.")

        return prior_context

    def update(
        self,
        prior_context: "RunContext",
        posterior_context: "RunContext",
        weight: float,
    ) -> None:
        next_iteration = prior_context.iteration + 1

        phase_string = f"Analyzing iteration: {next_iteration} with weight {weight}"
        self.setPhase(self.currentPhase() + 1, phase_string, indeterminate=True)

        try:
            self.facade.smoother_update(
                prior_context.sim_fs,
                posterior_context.sim_fs,
                prior_context.run_id,  # type: ignore
                global_std_scaling=weight,
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                "Update algorithm failed for iteration:"
                f"{next_iteration}. The following error occured {e}"
            ) from e

    @staticmethod
    def normalizeWeights(weights: Sequence[float]) -> List[float]:
        """Scale weights such that their reciprocals sum to 1.0,
        i.e., sum(1.0 / x for x in weights) == 1.0.
        See for example Equation 38 of evensen2018 - Analysis of iterative
        ensemble smoothers for solving inverse problems.
        """
        if not weights:
            return []
        weights = [weight for weight in weights if abs(weight) != 0.0]

        length = sum(1.0 / x for x in weights)
        return [x * length for x in weights]

    @classmethod
    def name(cls) -> str:
        return "Multiple Data Assimilation (ES MDA) - Recommended"
