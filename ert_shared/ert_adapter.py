from ert_shared.libres_facade import LibresFacade


class ErtAdapter:
    """The adapter object is the global ERT variable used all
    over the place in the application, and is added due to legacy
    reasons and the need for us to share code across the GUI and CLI.
    There is currently two different types of 'notifiers'
    plugging into the adapter - ErtNotifier and ErtCliNotifier.
    One second main thing is the wrapping of EnkfMain inside
    enkf_facade object which allows us to loosen up on the dependencies
    towards the enkf_main object in the application.
    """

    def __init__(self):
        self._implementation = None
        self._enkf_facade = None

    def adapt(self, implementation):
        self._implementation = implementation
        self._enkf_facade = LibresFacade(implementation.ert)

    @property
    def enkf_facade(self):
        return self._enkf_facade

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
