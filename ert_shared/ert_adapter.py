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
        """Sets implementation of EnkfMain to use. Must be used in a with-statement, and can not be called again
        while the contextmanager is active, as this is intended to mutate global state."""
        if self._implementation:
            raise ValueError("Cannot call adapt twice")
        self._implementation = implementation
        return self

    def __enter__(self):
        if self._enkf_facade:
            raise ValueError("Cannot use adapter twice")
        self._enkf_facade = LibresFacade(self._implementation.ert)
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._implementation.ert.umount()
        self._implementation = None
        self._enkf_facade = None

    @property
    def enkf_facade(self):
        if not self._enkf_facade:
            raise ValueError("Must use ert_adapter in a with statement")
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
