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
from ert.enkf.enums import EnkfInitModeEnum
from ert.enkf.enums import HookRuntime

from ert_gui.models.connectors.run import ActiveRealizationsModel,\
    TargetCaseFormatModel, AnalysisModuleModel, BaseRunModel
from ert_gui.models.mixins import ErtRunError
from ert_gui.models.mixins.connectorless import RelativeWeightsModel

from ert.util import BoolVector


class MultipleDataAssimilation(BaseRunModel):
    """
    Run Multiple Data Assimilation (MDA) Ensemble Smoother with custom weights.
    """

    def __init__(self):
        super(MultipleDataAssimilation, self).__init__(name="Multiple Data Assimilation", phase_count=2)

    def setAnalysisModule(self):
        module_name = AnalysisModuleModel().getCurrentChoice()
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError("Unable to load analysis module '%s'!" % module_name)


    def runSimulations(self):
        relativeWeights = RelativeWeightsModel()
        weights = relativeWeights.getValue()
        weights = self.parseWeights(weights)
        print("Running MDA ES with weights %s" % ", ".join(str(weight) for weight in weights))
        weights = self.normalizeWeights(weights)

        iteration_count = len(weights)
        self.setPhaseCount(iteration_count+2) # pre + post + weights

        print("Running MDA ES for %d iterations with the following normalized weights: %s" % (iteration_count, ", ".join(str(weight) for weight in weights)))

        target_case_format = TargetCaseFormatModel()
        print "target_case_format.getValue()", target_case_format.getValue()

        source_fs = self.ert().getEnkfFsManager().getCurrentFileSystem()
        target_case_name = target_case_format.getValue() % 0
        target_fs = self.ert().getEnkfFsManager().getFileSystem(target_case_name)

        if not source_fs == target_fs:
            self.ert().getEnkfFsManager().switchFileSystem(target_fs)
            self.ert().getEnkfFsManager().initializeCurrentCaseFromExisting(source_fs, 0)

        active_realization_mask = BoolVector(True, self.ert().getEnsembleSize())


        phaseName = "Running MDA ES for %d iterations with the following normalized weights: %s" % (iteration_count, ", ".join(str(weight) for weight in weights))
        self.setPhaseName(phaseName)

        self.ert().getEnkfSimulationRunner().createRunPath(active_realization_mask, 1)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )


        for iteration, weight in enumerate(weights):
            self.simulateAndPostProcess(target_case_format, active_realization_mask, iteration)
            self.update(target_case_format, iteration, weights[iteration])

        self.setPhaseName("Post processing...", indeterminate=True)
        self.simulateAndPostProcess(target_case_format, active_realization_mask, iteration_count)

        self.setPhase(iteration_count + 2, "Simulations completed.")


    def update(self, target_case_format, iteration, weight):
        self.setPhase(self.currentPhase() + 1, "Running simulations...", indeterminate=False)
        source_fs = self.ert().getEnkfFsManager().getCurrentFileSystem()
        next_iteration = (iteration + 1)
        next_target_case_name = target_case_format.getValue() % next_iteration
        target_fs = self.ert().getEnkfFsManager().getFileSystem(next_target_case_name)

        print("[%s] Analyzing iteration: %d with weight %f" % (next_target_case_name, next_iteration, weight))
        self.ert().analysisConfig().setGlobalStdScaling(weight)
        success = self.ert().getEnkfSimulationRunner().smootherUpdate(source_fs, target_fs)

        if not success:
            raise UserWarning("[%s] Analysis of simulation failed for iteration: %d!" % (next_target_case_name, next_iteration))


    def simulateAndPostProcess(self, target_case_format, active_realization_mask, iteration):
        self.setPhaseName("Post processing...", indeterminate=False)
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.POST_SIMULATION )

        self.setPhaseName("Analyzing...", indeterminate=False)

        target_case_name = target_case_format.getValue() % iteration

        target_fs = self.ert().getEnkfFsManager().getFileSystem(target_case_name)
        self.ert().getEnkfFsManager().switchFileSystem(target_fs)

        print("[%s] Running simulation for iteration: %d" % (target_case_name, iteration))
        self.ert().getEnkfSimulationRunner().createRunPath(active_realization_mask, iteration)

        print("[%s] Pre processing for iteration: %d" % (target_case_name, iteration))
        self.ert().getEnkfSimulationRunner().runWorkflows( HookRuntime.PRE_SIMULATION )

        success = self.ert().getEnkfSimulationRunner().runSimpleStep(active_realization_mask, EnkfInitModeEnum.INIT_CONDITIONAL, iteration)

        if not success:
            self.checkSuccessCount(active_realization_mask)

        print("[%s] Post processing for iteration: %d" % (target_case_name, iteration))
        self.ert().getEnkfSimulationRunner().runWorkflows(HookRuntime.POST_SIMULATION)


    def checkSuccessCount(self, active_realization_mask):
        min_realization_count = self.ert().analysisConfig().getMinRealisations()
        success_count = active_realization_mask.count()

        if min_realization_count > success_count:
            raise UserWarning("Simulation failed! Number of successful realizations less than MIN_REALIZATIONS %d < %d" % (success_count, min_realization_count))
        elif success_count == 0:
            raise UserWarning("Simulation failed! All realizations failed!")


    def normalizeWeights(self, weights):
        """ :rtype: list of float """
        from math import sqrt
        length = sqrt(sum((1.0 / x) * (1.0 / x) for x in weights))
        return [x * length for x in weights]


    def parseWeights(self, weights):
        if not weights:
            return []
        elements = weights.split(",")
        result = []
        for element in elements:
            element = element.strip()
            result.append(float(element))

        return result
