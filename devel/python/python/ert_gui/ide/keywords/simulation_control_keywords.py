from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition


class SimulationControlKeywords(object):
    def __init__(self, ert_keywords):
        super(SimulationControlKeywords, self).__init__()
        self.group = "Simulation Control"



        ert_keywords.addKeyword(self.addMaxRuntime())
        ert_keywords.addKeyword(self.addMinRealizations())


    def addMaxRuntime(self):
        max_runtime = ConfigurationLineDefinition(keyword=KeywordDefinition("MAX_RUNTIME"),
                                                   arguments=[IntegerArgument(from_value=1)],
                                                   documentation_link="control_simulations/max_runtime",
                                                   required=False,
                                                   group=self.group)
        return max_runtime




    def addMinRealizations(self):
        min_realizations = ConfigurationLineDefinition(keyword=KeywordDefinition("MIN_REALIZATIONS"),
                                                   arguments=[IntegerArgument(from_value=1)],
                                                   documentation_link="control_simulations/min_realizations",
                                                   required=False,
                                                   group=self.group)
        return min_realizations

