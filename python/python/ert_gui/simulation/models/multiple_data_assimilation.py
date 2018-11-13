#  Copyright (C) 2016 Statoil ASA, Norway.
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
from res.enkf.enums import EnkfInitModeEnum
from res.enkf.enums import HookRuntime
from res.enkf.enums import RealizationStateEnum
from res.enkf import ErtRunContext

from ert_gui.simulation.models import BaseRunModel, ErtRunError


class MultipleDataAssimilation(BaseRunModel):
    """
    Run Multiple Data Assimilation (MDA) Ensemble Smoother with custom weights.
    """

    def __init__(self, queue_config):
        super(MultipleDataAssimilation, self).__init__("Multiple Data Assimilation (ES MDA)", queue_config , phase_count=2)
        self.weights = "3, 2, 1" # default value

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        str_weights = str(weights)
        print("Weights changed: %s" % str_weights)
        self.weights = str_weights

    def setAnalysisModule(self, module_name):
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)


    def runSimulations(self, arguments):
        context = self.create_context(arguments, 0, None)
        self.checkMinimumActiveRealizations(context)
        weights = self.parseWeights(self.weights)
        iteration_count = len(weights)

        self.setAnalysisModule(arguments["analysis_module"])

        print("Running MDA ES for %s  iterations\t%s" % (iteration_count, ", ".join(str(weight) for weight in weights)))
        weights = self.normalizeWeights(weights)

        weight_string = ", ".join(str(round(weight,3)) for weight in weights)
        print("Running MDA ES on (weights normalized)\t%s" % weight_string)


        self.setPhaseCount(iteration_count+2) # pre + post + weights
        phase_string = "Running MDA ES %d iteration%s." % (iteration_count, ('s' if (iteration_count != 1) else ''))
        self.setPhaseName(phase_string, indeterminate=True)

        run_context = None
        for iteration, weight in enumerate(weights):
            run_context = self.create_context( arguments , iteration,  prior_context = run_context )
            self._simulateAndPostProcess(run_context )

            self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_UPDATE )
            self.update( run_context , weights[iteration])
            self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_UPDATE )

        self.setPhaseName("Post processing...", indeterminate=True)
        run_context = self.create_context( arguments , len(weights),  prior_context = run_context, update = False)
        self._simulateAndPostProcess(run_context)

        self.setPhase(iteration_count + 2, "Simulations completed.")

    def count_active_realizations(self, run_context):
        return sum(run_context.get_mask( ))


    def update(self, run_context, weight):
        source_fs = run_context.get_sim_fs( )
        next_iteration = run_context.get_iter( ) + 1
        target_fs = run_context.get_target_fs( )

        phase_string = "Analyzing iteration: %d with weight %f" % (next_iteration, weight)
        self.setPhase(self.currentPhase() + 1, phase_string, indeterminate=True)

        es_update = self.ert().getESUpdate( )
        es_update.setGlobalStdScaling(weight)
        success = es_update.smootherUpdate( run_context )

        if not success:
            raise UserWarning("Analysis of simulation failed for iteration: %d!" % next_iteration)


    def _simulateAndPostProcess(self, run_context):
        self._job_queue = self._queue_config.create_job_queue( )
        iteration = run_context.get_iter( )

        phase_string = "Running simulation for iteration: %d" % iteration
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(run_context)

        phase_string = "Pre processing for iteration: %d" % iteration
        self.setPhaseName(phase_string)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        phase_string = "Running forecast for iteration: %d" % iteration
        self.setPhaseName(phase_string, indeterminate=False)
        num_successful_realizations = self.ert().getEnkfSimulationRunner().runSimpleStep(self._job_queue, run_context)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        phase_string = "Post processing for iteration: %d" % iteration
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert().getEnkfSimulationRunner().runWorkflows(HookRuntime.POST_SIMULATION)
        self._job_queue = None
        return num_successful_realizations


    @staticmethod
    def normalizeWeights(weights):
        """ :rtype: list of float """
        if not weights:
            return []
        weights = [weight for weight in weights if abs(weight) != 0.0]
        from math import sqrt
        length = sqrt(sum((1.0 / x) * (1.0 / x) for x in weights))
        return [x * length for x in weights]


    @staticmethod
    def parseWeights(weights):
        if not weights:
            return []

        elements = weights.split(",")
        elements = [element.strip() for element in elements if not element.strip() == ""]

        result = []
        for element in elements:
            try:
                f = float(element)
                if f == 0:
                    print('Warning: 0 weight, will ignore')
                else:
                    result.append(f)
            except ValueError:
                raise ValueError('Warning: cannot parse weight %s' % element)

        return result


    def create_context(self, arguments, itr, prior_context = None, update = True):
        target_case_format = arguments["target_case"]
        model_config = self.ert().getModelConfig( )
        runpath_fmt = model_config.getRunpathFormat( )
        jobname_fmt = model_config.getJobnameFormat( )
        subst_list = self.ert().getDataKW( )
        fs_manager = self.ert().getEnkfFsManager()

        sim_fs = fs_manager.getFileSystem(target_case_format % itr)
        if update:
            target_fs = fs_manager.getFileSystem(target_case_format % (itr + 1))
        else:
            target_fs = None

        if prior_context is None:
            mask = arguments["active_realizations"]
        else:
            state = RealizationStateEnum.STATE_HAS_DATA | RealizationStateEnum.STATE_INITIALIZED
            mask = sim_fs.getStateMap().createMask(state)

        run_context = ErtRunContext.ensemble_smoother( sim_fs, target_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr)
        return run_context
