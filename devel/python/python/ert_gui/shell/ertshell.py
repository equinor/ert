from cmd import Cmd
import os

from ert.enkf import EnKFMain
from ert_gui.shell.cases import Cases
from ert_gui.shell.gen_kw_keys import GenKWKeys
from ert_gui.shell.results import Results
from ert_gui.shell.plugins import Plugins
from ert_gui.shell.summary_keys import SummaryKeys
from ert_gui.shell.workflows import Workflows
from ert_gui.shell import extractFullArgument, getPossibleFilenameCompletions

import matplotlib

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
            "Interactive shell for working with ERT.\n" \
            "\n" \
            "-- Type help for a list of supported commands.\n" \
            "-- Type exit or press Ctrl+D to end the shell session.\n" \
            "-- Press Tab for auto completion.\n"


    def __init__(self, site_config=None):
        Cmd.__init__(self)
        self.__site_config = site_config
        self.__ert = None
        """ :type: EnKFMain """

        matplotlib.rcParams["backend"] = "Qt4Agg"
        matplotlib.rcParams["interactive"] = True

        try:
            matplotlib.style.use("ggplot") # available from version 1.4
        except AttributeError:
            pass

        Workflows(self)
        Cases(self)
        Plugins(self)
        SummaryKeys(self)
        GenKWKeys(self)
        Results(self)

    def ert(self):
        """ @rtype: ert.enkf.enkf_main.EnKFMain """
        return self.__ert

    def emptyline(self):
        pass

    def do_load_config(self, config_file):
        if os.path.exists(config_file) and os.path.isfile(config_file):
            if self.__ert is not None:
                self.__ert.free()
                self.__ert = None

            self.__ert = EnKFMain(config_file, site_config=self.__site_config)
        else:
            print("Error: Config file '%s' not found!\n" % config_file)

    def complete_load_config(self, text, line, begidx, endidx):
        argument = extractFullArgument(line, endidx)
        return getPossibleFilenameCompletions(argument)


    def help_load_config(self):
        print("\n".join(("load_config config_file",
                         "    Loads a config file.")))

    def do_exit(self, line):
        if self.ert() is not None:
            self.ert().free()
        return True

    def help_exit(self):
        return "\n".join(("exit",
                          "    End the shell session.")),

    do_EOF = do_exit

    def help_EOF(self):
        return "\n".join(("EOF",
                          "    The same as exit. (Ctrl+D)")),
