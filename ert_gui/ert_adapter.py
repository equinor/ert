class ErtAdapter():

    def adapt(self, implementation):
        self._implementation = implementation
    
    @property
    def ertChanged(self):
        return self._implementation.ertChanged

    @property
    def ert(self):
        return self._implementation.ert

    @property
    def config_file(self):
        return self._implementation.config_file
    
    def emitErtChange(self):
        self._implementation.emitErtChange()

    def reloadERT(self, config_file):
        self._implementation.reloadERT(config_file)

ERT = ErtAdapter()
