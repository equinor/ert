from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from ert._c_wrappers.enkf.enums import HookRuntime, RealizationStateEnum
from ert.analysis import ErtAnalysisError
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import EnsembleAccessor, StorageAccessor

from .base_run_model import BaseRunModel, ErtRunError

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain, QueueConfig, RunContext

logger = logging.getLogger(__file__)


# pylint: disable=too-many-arguments
class MultipleDataAssimilation(BaseRunModel):
    """
    Run multiple data assimilation (MDA) ensemble smoother with custom weights.
    """

    default_weights = "4, 2, 1"

    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
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

    def setAnalysisModule(self, module_name: str) -> None:
        module_load_success = self.ert().analysisConfig().select_module(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> "RunContext":
        self._checkMinimumActiveRealizations(
            self._simulation_arguments["active_realizations"].count(True)
        )
        weights = self.parseWeights(self._simulation_arguments["weights"])

        if not weights:
            raise ErtRunError("Cannot perform ES_MDA with no weights provided!")

        iteration_count = len(weights)

        self.setAnalysisModule(self._simulation_arguments["analysis_module"])

        logger.info(
            f"Running MDA ES for {iteration_count}  "
            f'iterations\t{", ".join(str(weight) for weight in weights)}'
        )
        weights = self.normalizeWeights(weights)

        weight_string = ", ".join(str(round(weight, 3)) for weight in weights)
        logger.info(f"Running MDA ES on (weights normalized)\t{weight_string}")

        self.setPhaseCount(iteration_count + 1)  # weights + post
        phase_string = (
            f"Running MDA ES {iteration_count} "
            f'iteration{"s" if (iteration_count != 1) else ""}.'
        )
        self.setPhaseName(phase_string, indeterminate=True)

        enumerated_weights = list(enumerate(weights))
        restart_run = self._simulation_arguments["restart_run"]
        case_format = self._simulation_arguments["target_case"]
        prior_ensemble = self._simulation_arguments["prior_ensemble"]

        if restart_run:
            try:
                prior_fs = self._storage.get_ensemble_by_name(prior_ensemble)
                self.set_env_key("_ERT_ENSEMBLE_ID", str(prior_fs.id))
                assert isinstance(prior_fs, EnsembleAccessor)
                prior_context = self.ert().ensemble_context(
                    prior_fs,
                    self._simulation_arguments["active_realizations"],
                    iteration=prior_fs.iteration,
                )
            except KeyError as err:
                raise ErtRunError(
                    f"Prior ensemble: {prior_ensemble} does not exists"
                ) from err
        else:
            prior_fs = self._storage.create_ensemble(
                self._experiment_id,
                ensemble_size=self._ert.getEnsembleSize(),
                iteration=0,
                name=case_format % 0,
            )
            self.set_env_key("_ERT_ENSEMBLE_ID", str(prior_fs.id))
            prior_context = self.ert().ensemble_context(
                prior_fs,
                self._simulation_arguments["active_realizations"],
                iteration=prior_fs.iteration,
            )
            self.ert().sample_prior(
                prior_context.sim_fs, prior_context.active_realizations
            )
            self._simulateAndPostProcess(prior_context, evaluator_server_config)
        starting_iteration = prior_fs.iteration + 1
        weights_to_run = enumerated_weights[max(starting_iteration - 1, 0) :]

        for iteration, weight in weights_to_run:
            is_first_iteration = iteration == 0
            if is_first_iteration:
                self.ert().runWorkflows(
                    HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_fs
                )
            self.ert().runWorkflows(HookRuntime.PRE_UPDATE, self._storage, prior_fs)
            states = [
                RealizationStateEnum.STATE_HAS_DATA,
                RealizationStateEnum.STATE_INITIALIZED,
            ]
            posterior_context = self.ert().ensemble_context(
                self._storage.create_ensemble(
                    self._experiment_id,
                    name=case_format % (iteration + 1),
                    ensemble_size=self._ert.getEnsembleSize(),
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
            self._simulateAndPostProcess(posterior_context, evaluator_server_config)
            prior_context = posterior_context

        self.setPhaseName("Post processing...", indeterminate=True)

        self.setPhase(iteration_count + 1, "Simulations completed.")

        return prior_context

    def _count_active_realizations(self, run_context: "RunContext") -> int:
        return sum(run_context.mask)

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
                "Analysis of simulation failed for iteration:"
                f"{next_iteration}. The following error occured {e}"
            ) from e

    def _simulateAndPostProcess(
        self,
        run_context: "RunContext",
        evaluator_server_config: EvaluatorServerConfig,
    ) -> int:
        iteration = run_context.iteration

        phase_string = f"Running simulation for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert().createRunPath(run_context)

        phase_string = f"Pre processing for iteration: {iteration}"
        self.setPhaseName(phase_string)
        self.ert().runWorkflows(
            HookRuntime.PRE_SIMULATION, self._storage, run_context.sim_fs
        )

        phase_string = f"Running forecast for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        num_successful_realizations += self._simulation_arguments.get(
            "prev_successful_realizations", 0
        )
        self.checkHaveSufficientRealizations(num_successful_realizations)

        phase_string = f"Post processing for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert().runWorkflows(
            HookRuntime.POST_SIMULATION, self._storage, run_context.sim_fs
        )

        return num_successful_realizations

    @staticmethod
    def normalizeWeights(weights: List[float]) -> List[float]:
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

    @staticmethod
    def parseWeights(weights: str) -> List[float]:
        if not weights:
            return []

        elements = weights.split(",")
        elements = [
            element.strip() for element in elements if not element.strip() == ""
        ]

        result = []
        for element in elements:
            try:
                f = float(element)
                if f == 0:
                    logger.info("Warning: 0 weight, will ignore")
                else:
                    result.append(f)
            except ValueError:
                raise ValueError(f"Warning: cannot parse weight {element}")

        return result

    @classmethod
    def name(cls) -> str:
        return "Multiple Data Assimilation (ES MDA) - Recommended"
