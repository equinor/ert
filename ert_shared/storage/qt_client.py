import logging
from qtpy.QtCore import QObject, Signal, Slot, Property
from ert_shared.storage.client import StorageClient
from ert_shared.storage.qt_client_iface import QtClientIface


class QtStorageClient(StorageClient, QObject):

    readyChanged = Signal(bool)

    def __init__(self, base_url, project):
        StorageClient.__init__(self, base_url)
        QObject.__init__(self)

        self._ready = False

        if base_url is None:
            logging.info("Initializing QtStorageClient with discovery")
            self._iface = QtClientIface(project, self)
            self._iface.storage_api_added.connect(self._storage_api_added)
            self._iface.storage_api_removed.connect(self._storage_api_removed)
            self._iface.watch()
        else:
            logging.info(
                "Initializing QtStorageClient without discovery with base url {}".format(
                    base_url
                )
            )
            self.ready = True

    @Slot(str)
    def _storage_api_added(self, url):
        self._BASE_URI = url
        self.ready = True

    @Slot()
    def _storage_api_removed(self):
        self.ready = False

    @Property(int, notify=readyChanged)
    def ready(self):
        return self._ready

    @ready.setter
    def ready(self, value):
        if self._ready != value:
            self._ready = value
            self.readyChanged.emit(value)

    def shutdown_server(self):
        self._iface.shutdown()
