from ert.job_queue import ErtScript, CancelPluginException
from ert_gui.shell import ShellFunction, assertConfigLoaded, autoCompleteList


class Plugins(ShellFunction):
    def __init__(self, cmd):
        super(Plugins, self).__init__("plugins", cmd)

        self.addHelpFunction("list", None, "Shows a list of all available plugins.")
        self.addHelpFunction("run", "<workflow_name>", "Run a named plugin.")

    @assertConfigLoaded
    def do_list(self, line):
        plugins = self.getPluginNames()
        if len(plugins) > 0:
            self.cmd.columnize(plugins)
        else:
            print("No plugins available.")


    @assertConfigLoaded
    def do_run(self, plugin_name):
        plugin_name = plugin_name.strip()
        plugin_jobs = self.ert().getWorkflowList().getPluginJobs()
        plugin_job = next((job for job in plugin_jobs if job.name() == plugin_name), None)
        """ :type: ert.job_queue.WorkflowJob """

        if plugin_job is not None:
            try:
                script_obj = ErtScript.loadScriptFromFile(plugin_job.getInternalScriptPath())
                script = script_obj(self.ert())
                arguments = script.getArguments(None)
                # print(inspect.getargspec(script.run)) #todo: AutoComplete of workflow_job_script arguments...
                result = plugin_job.run(self.ert(), arguments)

                print(result)
            except CancelPluginException:
                print("Plugin cancelled before execution!")
        else:
            print("Error: Unknown plugin: '%s'" % plugin_name)


    @assertConfigLoaded
    def complete_run(self, text, line, begidx, endidx):
        return autoCompleteList(text, self.getPluginNames())


    def getPluginNames(self):
        plugin_jobs = self.ert().getWorkflowList().getPluginJobs()
        return [plugin.name() for plugin in plugin_jobs]