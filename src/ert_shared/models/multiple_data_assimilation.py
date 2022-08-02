#  Copyright (C) 2016 Equinor ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import logging
from typing import Any, Dict, List, Tuple

from ert.analysis import ErtAnalysisError
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.models import BaseRunModel, ErtRunError
from res.enkf import EnkfSimulationRunner, RunContext
from res.enkf.enkf_main import EnKFMain, QueueConfig
from res.enkf.enums import HookRuntime, RealizationStateEnum

logger = logging.getLogger(__file__)


class MultipleDataAssimilation(BaseRunModel):
    """
    Run multiple data assimilation (MDA) ensemble smoother with custom weights.
    """

    default_weights = "4, 2, 1"

    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
    ):
        super().__init__(simulation_arguments, ert, queue_config, phase_count=2)
        self.weights = MultipleDataAssimilation.default_weights

    def setAnalysisModule(self, module_name: str) -> None:
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        context = self.create_context(0, initialize_mask_from_arguments=True)
        self._checkMinimumActiveRealizations(context)
        weights = self.parseWeights(self._simulation_arguments["weights"])
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

        run_context = None
        update_id = None
        enumerated_weights = list(enumerate(weights))
        weights_to_run = enumerated_weights[
            min(self._simulation_arguments["start_iteration"], len(weights)) :
        ]
        for iteration, weight in weights_to_run:
            is_first_iteration = iteration == 0
            run_context = self.create_context(
                iteration, initialize_mask_from_arguments=is_first_iteration
            )
            _, ensemble_id = self._simulateAndPostProcess(
                run_context, evaluator_server_config, update_id=update_id
            )
            if is_first_iteration:
                EnkfSimulationRunner.runWorkflows(
                    HookRuntime.PRE_FIRST_UPDATE, ert=self.ert()
                )
            EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_UPDATE, ert=self.ert())
            update_id = self.update(
                run_context=run_context, weight=weight, ensemble_id=ensemble_id
            )
            EnkfSimulationRunner.runWorkflows(HookRuntime.POST_UPDATE, ert=self.ert())

        self.setPhaseName("Post processing...", indeterminate=True)
        run_context = self.create_context(
            len(weights), initialize_mask_from_arguments=False, update=False
        )
        self._simulateAndPostProcess(
            run_context, evaluator_server_config, update_id=update_id
        )

        self.setPhase(iteration_count + 1, "Simulations completed.")

        return run_context

    def _count_active_realizations(self, run_context: RunContext) -> int:
        return sum(run_context.mask)

    def update(self, run_context: RunContext, weight: float, ensemble_id: str) -> str:
        next_iteration = run_context.iteration + 1

        phase_string = f"Analyzing iteration: {next_iteration} with weight {weight}"
        self.setPhase(self.currentPhase() + 1, phase_string, indeterminate=True)

        self.facade.set_global_std_scaling(weight)
        try:
            self.facade.smoother_update(run_context)
        except ErtAnalysisError as e:
            raise ErtRunError(
                "Analysis of simulation failed for iteration:"
                f"{next_iteration}. The following error occured {e}"
            ) from e
        # Push update data to new storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = self._post_update_data(
            parent_ensemble_id=ensemble_id, algorithm=analysis_module_name
        )

        return update_id

    def _simulateAndPostProcess(
        self,
        run_context: RunContext,
        evaluator_server_config: EvaluatorServerConfig,
        update_id: str = None,
    ) -> Tuple[int, str]:
        iteration = run_context.iteration

        phase_string = f"Running simulation for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(run_context)

        phase_string = f"Pre processing for iteration: {iteration}"
        self.setPhaseName(phase_string)
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())

        # Push ensemble, parameters, observations to new storage
        new_ensemble_id = self._post_ensemble_data(update_id=update_id)

        phase_string = f"Running forecast for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        # Push simulation results to storage
        self._post_ensemble_results(new_ensemble_id)

        num_successful_realizations += self._simulation_arguments.get(
            "prev_successful_realizations", 0
        )
        self.checkHaveSufficientRealizations(num_successful_realizations)

        phase_string = f"Post processing for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, ert=self.ert())

        return num_successful_realizations, new_ensemble_id

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
    def parseWeights(weights: str) -> List:
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

    def create_context(
        self,
        itr: int,
        initialize_mask_from_arguments: bool = True,
        update: bool = True,
    ) -> RunContext:
        target_case_format = self._simulation_arguments["target_case"]
        fs_manager = self.ert().getEnkfFsManager()

        source_case_name = target_case_format % itr
        if itr > 0 and not fs_manager.caseExists(source_case_name):
            raise ErtRunError(
                f"Source case {source_case_name} for iteration {itr} does not exists. "
                "If you are attempting to restart ESMDA from a iteration other than 0, "
                "make sure the target case format is the same as for the original run! "
                f"(Current target case format: {target_case_format})"
            )

        sim_fs = fs_manager.getFileSystem(source_case_name)
        if update:
            target_fs = fs_manager.getFileSystem(target_case_format % (itr + 1))
        else:
            target_fs = None

        if initialize_mask_from_arguments:
            mask = self._simulation_arguments["active_realizations"]
        else:
            mask = sim_fs.getStateMap().createMask(
                RealizationStateEnum.STATE_HAS_DATA
                | RealizationStateEnum.STATE_INITIALIZED
            )
            # Make sure to only run the realizations which was passed in as argument
            for idx, (valid_state, run_realization) in enumerate(
                zip(mask, self._initial_realizations_mask)
            ):
                mask[idx] = valid_state and run_realization

        run_context = self.ert().create_ensemble_smoother_run_context(
            source_filesystem=sim_fs,
            target_filesystem=target_fs,
            active_mask=mask,
            iteration=itr,
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.iteration
        self.ert().getEnkfFsManager().switchFileSystem(sim_fs)
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Multiple Data Assimilation (ES MDA) - Recommended"
