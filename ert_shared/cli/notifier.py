class ErtCliNotifier:
    """CLI Notifier to use in ERT Adapter"""

    def __init__(self, ert, config_file):
        self._ert = ert
        self._config_file = config_file

    @property
    def ert(self):
        """@rtype: EnKFMain"""
        if self._ert is None:
            raise ValueError("Ert is undefined.")
        return self._ert

    @property
    def config_file(self):
        """@rtype: str"""
        if self._ert is None:
            raise ValueError("Ert is undefined.")
        return self._config_file

    @property
    def ertChanged(self):
        pass

    def emitErtChange(self):
        pass

    def reloadERT(self, config_file):
        pass
