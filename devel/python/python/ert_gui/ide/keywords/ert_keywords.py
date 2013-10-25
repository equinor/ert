from ert_gui.ide.keywords.definitions import ConfigurationLineDefinition
from ert_gui.ide.keywords.eclipse_keywords import EclipseKeywords
from ert_gui.ide.keywords.ensemble_keywords import EnsembleKeywords
from ert_gui.ide.keywords.queue_system_keywords import QueueSystemKeywords
from ert_gui.ide.keywords.simulation_control_keywords import SimulationControlKeywords


class ErtKeywords(object):
    def __init__(self):
        super(ErtKeywords, self).__init__()

        self.keywords = {}

        EnsembleKeywords(self)
        EclipseKeywords(self)
        QueueSystemKeywords(self)
        SimulationControlKeywords(self)

    def addKeyword(self, keyword):
        assert isinstance(keyword, ConfigurationLineDefinition)

        name = keyword.keywordDefinition().name()
        if name in self.keywords:
            raise ValueError("Keyword %s already in Ert keyword list!" % name)

        self.keywords[name] = keyword

    def __contains__(self, item):
        return item in self.keywords

    def __getitem__(self, item):
        """ @rtype: ConfigurationLineDefinition """
        return self.keywords[item]

