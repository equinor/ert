from ert.enkf import EnkfSimulationRunner
from ert_gui.shell import ShellFunction, assertConfigLoaded


class Smoother(ShellFunction):
    def __init__(self, shell_context):
        super(Smoother, self).__init__("smoother", shell_context)
        self.addHelpFunction("update", "<target_case>", "Run smoother update on the current case, placing the results in the target case.")
        self.addHelpFunction("overlap_alpha", "[alpha_value]", "Show or set the overlap alpha.")
        self.addHelpFunction("std_cutoff", "[cutoff_value]", "Show or set the standard deviation cutoff value (>0).")


    @assertConfigLoaded
    def do_update(self, line):
        arguments = self.splitArguments(line)

        if len(arguments) == 1:
            case_name = arguments[0]
            target_fs = self.ert().getEnkfFsManager().getFileSystem(case_name)
            simulation_runner = EnkfSimulationRunner(self.ert())
            success = simulation_runner.smootherUpdate(target_fs)

            if not success:
                self.lastCommandFailed("Unable to perform update")

        else:
            self.lastCommandFailed("Expected one argument: <target_fs> received: '%s'" % line)


    @assertConfigLoaded
    def do_overlap_alpha(self, line):
        value = line.strip()
        if value == "":
            analysis_config = self.shellContext().ert().analysisConfig()
            print("Overlap Alpha = %f" % analysis_config.getEnkfAlpha())
        else:
            try:
                value = float(value)
                analysis_config = self.shellContext().ert().analysisConfig()
                analysis_config.setEnkfAlpha(value)
                print("Overlap Alpha set to: %f" % analysis_config.getEnkfAlpha())
            except ValueError:
                self.lastCommandFailed("Expected a number")


    @assertConfigLoaded
    def do_std_cutoff(self, line):
        value = line.strip()
        if value == "":
            analysis_config = self.shellContext().ert().analysisConfig()
            print("Standard Deviation Cutoff = %f" % analysis_config.getStdCutoff())
        else:
            try:
                value = float(value)
                analysis_config = self.shellContext().ert().analysisConfig()
                analysis_config.setStdCutoff(value)
                print("Standard Deviation Cutoff set to: %f" % analysis_config.getStdCutoff())
            except ValueError:
                self.lastCommandFailed("Expected a number")



