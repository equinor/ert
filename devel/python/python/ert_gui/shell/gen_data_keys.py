from ert.enkf import ErtImplType
from ert_gui.shell import ShellFunction, extractFullArgument, autoCompleteListWithSeparator, ShellPlot, \
    assertConfigLoaded


class GenDataKeys(ShellFunction):
    def __init__(self, shell_context):
        super(GenDataKeys, self).__init__("gen_data", shell_context)
        self.addHelpFunction("list", None, "Shows a list of all available gen_data keys.")


    def fetchSupportedKeys(self):
        gen_data_keys = self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_DATA)
        gen_data_list = []
        for key in gen_data_keys:
            enkf_config_node = self.ert().ensembleConfig().getNode(key)
            gen_data_config = enkf_config_node.getDataModelConfig()

            for report_step in range(self.ert().getHistoryLength()):
                if gen_data_config.hasReportStep(report_step):
                    gen_data_list.append("%s@%d" % (key, report_step))

        return gen_data_list

    @assertConfigLoaded
    def do_list(self, line):
        keys = sorted(self.fetchSupportedKeys())

        self.columnize(keys)

