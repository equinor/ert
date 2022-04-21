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
from typing import List, Tuple, Dict, Any
from res.enkf import ErtRunContext, EnkfSimulationRunner
from res.enkf.enums import HookRuntime
from res.enkf.enums import RealizationStateEnum
from res.enkf.enkf_main import EnKFMain, QueueConfig

from ert_shared.models import BaseRunModel, ErtRunError
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig

import logging

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
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> ErtRunContext:
        context = self.create_context(0, initialize_mask_from_arguments=True)
        self._checkMinimumActiveRealizations(context)
        weights = self.parseWeights(self._simulation_arguments["weights"])
        iteration_count = len(weights)

        self.setAnalysisModule(self._simulation_arguments["analysis_module"])

        logger.info(
            "Running MDA ES for %s  iterations\t%s"
            % (iteration_count, ", ".join(str(weight) for weight in weights))
        )
        weights = self.normalizeWeights(weights)

        weight_string = ", ".join(str(round(weight, 3)) for weight in weights)
        logger.info("Running MDA ES on (weights normalized)\t%s" % weight_string)

        self.setPhaseCount(iteration_count + 1)  # weights + post
        phase_string = "Running MDA ES %d iteration%s." % (
            iteration_count,
            ("s" if (iteration_count != 1) else ""),
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

    def _count_active_realizations(self, run_context: ErtRunContext) -> int:
        return sum(run_context.get_mask())

    def update(
        self, run_context: ErtRunContext, weight: float, ensemble_id: str
    ) -> str:
        next_iteration = run_context.get_iter() + 1

        phase_string = "Analyzing iteration: %d with weight %f" % (
            next_iteration,
            weight,
        )
        self.setPhase(self.currentPhase() + 1, phase_string, indeterminate=True)

        es_update = self.ert().getESUpdate()
        es_update.setGlobalStdScaling(weight)
        success = es_update.smootherUpdate(run_context)

        # Push update data to new storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = self._post_update_data(
            parent_ensemble_id=ensemble_id, algorithm=analysis_module_name
        )

        if not success:
            raise UserWarning(
                "Analysis of simulation failed for iteration: %d!" % next_iteration
            )
        return update_id

    def _simulateAndPostProcess(
        self,
        run_context: ErtRunContext,
        evaluator_server_config: EvaluatorServerConfig,
        update_id: str = None,
    ) -> Tuple[int, str]:
        iteration = run_context.get_iter()

        phase_string = "Running simulation for iteration: %d" % iteration
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(run_context)

        phase_string = "Pre processing for iteration: %d" % iteration
        self.setPhaseName(phase_string)
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())

        # Push ensemble, parameters, observations to new storage
        new_ensemble_id = self._post_ensemble_data(update_id=update_id)

        phase_string = "Running forecast for iteration: %d" % iteration
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

        phase_string = "Post processing for iteration: %d" % iteration
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
                raise ValueError("Warning: cannot parse weight %s" % element)

        return result

    def create_context(
        self,
        itr: int,
        initialize_mask_from_arguments: bool = True,
        update: bool = True,
    ) -> ErtRunContext:
        target_case_format = self._simulation_arguments["target_case"]
        model_config = self.ert().getModelConfig()
        runpath_fmt = model_config.getRunpathFormat()
        jobname_fmt = model_config.getJobnameFormat()
        subst_list = self.ert().getDataKW()
        fs_manager = self.ert().getEnkfFsManager()

        source_case_name = target_case_format % itr
        if itr > 0 and not fs_manager.caseExists(source_case_name):
            raise ErtRunError(
                "Source case {} for iteration {} does not exists. "
                "If you are attempting to restart ESMDA from a iteration other than 0, "
                "make sure the target case format is the same as for the original run! "
                "(Current target case format: {})".format(
                    source_case_name, itr, target_case_format
                )
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

        run_context = ErtRunContext.ensemble_smoother(
            sim_fs, target_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.get_iter()
        self.ert().getEnkfFsManager().switchFileSystem(sim_fs)
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Multiple Data Assimilation (ES MDA) - Recommended"
