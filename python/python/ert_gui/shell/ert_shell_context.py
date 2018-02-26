from ert_gui.shell.libshell import ShellContext


class ErtShellContext(ShellContext):
    def __init__(self, shell):
        super(ErtShellContext, self).__init__(shell)
        self.__ert = None
        self.__res_config = None
        """ :type: EnKFMain """

    def ert(self):
        """ @rtype: res.enkf.enkf_main.EnKFMain """
        return self.__ert

    def setErt(self, ert):
        """ @type ert: res.enkf.enkf_main.EnKFMain """
        self.__ert = ert

    @property
    def res_config(self):
        return self.__res_config

    @res_config.setter
    def res_config(self, new_res_config):
        if self.__res_config not in [None, new_res_config]:
            self.__res_config.free()
            self.__res_config = None

        self.__res_config = new_res_config
