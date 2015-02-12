from cmd import Cmd
import os

from ert.enkf import EnKFMain, ErtImplType
from ert.job_queue import WorkflowRunner, ErtScript, CancelPluginException

def pathify(head, tail):
    path = os.path.join(head, tail)
    if os.path.isdir(path):
        return "%s/" % tail
    return tail


def getPossibleFilenameCompletions(text):
    head, tail = os.path.split(text.strip())
    if head == "":  # no head
        head = "."
    files = os.listdir(head)
    return [pathify(head, f) for f in files if f.startswith(tail)]


def extractFullArgument(line, endidx):
    newstart = line.rfind(" ", 0, endidx)
    return line[newstart:endidx]


def getWorkflowNames(ert):
    return [workflow for workflow in ert.getWorkflowList().getWorkflowNames()]


def getPluginNames(ert):
    plugin_jobs = ert.getWorkflowList().getPluginJobs()
    return [plugin.name() for plugin in plugin_jobs]


def getFileSystemNames(ert):
    return sorted([fs for fs in ert.getEnkfFsManager().getCaseList()])


def autoCompleteList(text, items):
    if not text:
        completions = items
    else:
        completions = [item for item in items if item.startswith(text)]
    return completions


def createHelpFunction(help_message):
    def helpFunction(self):
        print(help_message)

    return helpFunction


def assertConfigLoaded(func):
    def wrapper(self, *args, **kwargs):
        result = False
        if self.isConfigLoaded():
            result = func(self, *args, **kwargs)

        return result

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__

    return wrapper

class ErtShell(Cmd):
    prompt = "--> "
    intro = "  _________________________________ \n" \
            " /                                 \\\n" \
            " |   ______   ______   _______     |\n" \
            " |  |  ____| |  __  \ |__   __|    |\n" \
            " |  | |__    | |__) |    | |       |\n" \
            " |  |  __|   |  _  /     | |       |\n" \
            " |  | |____  | | \ \     | |       |\n" \
            " |  |______| |_|  \_\    |_|       |\n" \
            " |                                 |\n" \
            " |  Ensemble based Reservoir Tool  |\n" \
            " \_________________________________/\n" \
            "\n" \
            "Interactive shell for working with Ert." \
            "\n" \
            "-- Type help for a list of supported commands. Type Ctrl+D or exit to end the shell session."


    def __init__(self, site_config=None):
        Cmd.__init__(self)
        self.site_config = site_config
        self.ert = None
        """ :type: EnKFMain """

        self.addHelpFunctions()


    def addHelpFunctions(self):
        help_map = {"list_file_systems": "\n".join(("list_file_systems",
                                                    "    Shows a list of all available file systems.")),

                    "select_current_file_system": "\n".join(("select_current_file_system file_system_name",
                                                             "    Change the current file system to the named case.")),

                    "load_config": "\n".join(("load_config config_file",
                                              "    Loads a config file.")),

                    "list_workflows": "\n".join(("list_workflows",
                                                 "    Shows a list of available workflows.")),

                    "run_workflow": "\n".join(("run_workflow workflow_name",
                                               "    Run a named workflow.")),

                    "list_plugins": "\n".join(("list_plugins",
                                               "    Shows a list of available plugins.")),

                    "run_plugin": "\n".join(("run_plugin plugin_name",
                                             "    Run a named plugin.")),

                    "list_summary_keys": "\n".join(("list_summary_keys",
                                                    "    Shows a list of all available Summary keys.")),

                    "exit": "\n".join(("exit",
                                       "    End the shell session.")),

                    "EOF": "\n".join(("EOF",
                                      "    The same as exit. Ctrl+D to activate."))
        }

        for help_key in help_map:
            setattr(self.__class__, "help_%s" % help_key, createHelpFunction(help_map[help_key]))


    def isConfigLoaded(self):
        if self.ert is None:
            print("Error: A config file has not been loaded!")
            return False
        return True

    def do_load_config(self, config_file):
        if os.path.exists(config_file) and os.path.isfile(config_file):
            if self.ert is not None:
                self.ert.free()
                self.ert = None

            self.ert = EnKFMain(config_file, site_config=self.site_config)
        else:
            print("Error: Config file '%s' not found!\n" % config_file)


    def complete_load_config(self, text, line, begidx, endidx):
        argument = extractFullArgument(line, endidx)
        return getPossibleFilenameCompletions(argument)


    @assertConfigLoaded
    def do_list_workflows(self, line):
        workflows = getWorkflowNames(self.ert)
        if len(workflows) > 0:
            self.columnize(workflows)
        else:
            print("No workflows available.")

    @assertConfigLoaded
    def do_run_workflow(self, workflow):
        workflow = workflow.strip()
        if workflow in getWorkflowNames(self.ert):
            workflow_list = self.ert.getWorkflowList()
            workflow = workflow_list[workflow]
            context = workflow_list.getContext()

            runner = WorkflowRunner(workflow, self.ert, context)
            runner.run()
            runner.wait()
        else:
            print("Error: Unknown workflow: '%s'" % workflow)

    def complete_run_workflow(self, text, line, begidx, endidx):
        return autoCompleteList(text, getWorkflowNames(self.ert))

    @assertConfigLoaded
    def do_list_plugins(self, line):
        plugins = getPluginNames(self.ert)
        if len(plugins) > 0:
            self.columnize(plugins)
        else:
            print("No plugins available.")


    @assertConfigLoaded
    def do_run_plugin(self, plugin_name):
        plugin_name = plugin_name.strip()
        plugin_jobs = self.ert.getWorkflowList().getPluginJobs()
        plugin_job = next((job for job in plugin_jobs if job.name() == plugin_name), None)
        """ :type: ert.job_queue.WorkflowJob """

        if plugin_job is not None:
            try:
                script_obj = ErtScript.loadScriptFromFile(plugin_job.getInternalScriptPath())
                script = script_obj(self.ert)
                arguments = script.getArguments(None)
                # print(inspect.getargspec(script.run)) #todo: AutoComplete of workflow_job_script arguments...
                result = plugin_job.run(self.ert, arguments)

                print(result)
            except CancelPluginException:
                print("Plugin cancelled before execution!")
        else:
            print("Error: Unknown plugin: '%s'" % plugin_name)


    def complete_run_plugin(self, text, line, begidx, endidx):
        return autoCompleteList(text, getPluginNames(self.ert))

    @assertConfigLoaded
    def do_list_summary_keys(self, line):
        keys = [key for key in self.ert.ensembleConfig().getKeylistFromImplType(ErtImplType.SUMMARY)]
        self.columnize(sorted(keys))

    @assertConfigLoaded
    def do_select_current_file_system(self, case_name):
        case_name = case_name.strip()
        if case_name in getFileSystemNames(self.ert):
            fs = self.ert.getEnkfFsManager().getFileSystem(case_name)
            self.ert.getEnkfFsManager().switchFileSystem(fs)
        else:
            print("Error: Unknown file system '%s'" % case_name)

    def complete_select_current_file_system(self, text, line, begidx, endidx):
        return autoCompleteList(text, getFileSystemNames(self.ert))


    @assertConfigLoaded
    def do_list_file_systems(self, line):
        fs_list = getFileSystemNames(self.ert)
        current_fs = self.ert.getEnkfFsManager().getCurrentFileSystem().getCaseName()
        max_length = max([len(fs) for fs in fs_list])
        format = "%1s %-" + str(max_length) + "s  %s"
        for fs in fs_list:
            current = ""
            if fs == current_fs:
                current = "*"

            state = "No Data"
            if self.ert.getEnkfFsManager().caseHasData(fs):
                state = "Data"

            print(format % (current, fs, state))

    def do_exit(self, line):
        return self.exit()

    def do_EOF(self, line):
        return self.exit()

    def exit(self):
        if self.ert is not None:
            self.ert.free()
        return True
