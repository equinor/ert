from ert.enkf.enkf_simulation_runner import EnkfSimulationRunner
from ert.util import BoolVector
from ert_gui.shell import ShellFunction, assertConfigLoaded


class Simulations(ShellFunction):
    def __init__(self, shell_context):
        super(Simulations, self).__init__("simulations", shell_context)
        self.addHelpFunction("settings", None, "Show simulations settings.")
        self.addHelpFunction("ensemble_experiment", None, "Run Ensemble Experiment.")


    @assertConfigLoaded
    def do_settings(self, line):
        runpath = self.ert().getModelConfig().getRunpathAsString()

        iteration_count = self.ert().analysisConfig().getAnalysisIterConfig().getNumIterations()
        realizations = self.ert().getEnsembleSize()

        print("Runpath: %s" % runpath)
        print("Iteration count: %d" % iteration_count)
        print("Realization count: %d" % realizations)


    @assertConfigLoaded
    def do_ensemble_experiment(self, line):
        simulation_runner = EnkfSimulationRunner(self.ert())

        print("Ensemble Experiment started!")
        success = simulation_runner.runEnsembleExperiment()

        if not success:
            print("Error: Simulations failed!")
            return

        print("Ensemble Experiment post processing!")
        simulation_runner.runPostWorkflow()

        print("Ensemble Experiment completed!")



