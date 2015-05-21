from ert.enkf import EnkfSimulationRunner
from ert_gui.shell import ShellFunction, assertConfigLoaded


class Smoother(ShellFunction):
    def __init__(self, shell_context):
        super(Smoother, self).__init__("smoother", shell_context)
        self.addHelpFunction("update", "<target_case>", "Run smoother update on the current case, placing the results in the target case.")


    @assertConfigLoaded
    def do_update(self, line):
        arguments = self.splitArguments(line)

        if len(arguments) == 1:
            case_name = arguments[0]
            target_fs = self.ert().getEnkfFsManager().getFileSystem(case_name)
            simulation_runner = EnkfSimulationRunner(self.ert())
            success = simulation_runner.smootherUpdate(target_fs)

            if not success:
                print("Error: Unable to perform update")

        else:
            print("Error: Expected one argument: <target_fs> received: '%s'" % line)






