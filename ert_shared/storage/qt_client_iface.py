from PyQt5 import QtDBus
from qtpy.QtCore import QObject, Signal, Slot


class ServiceUnknownError(Exception):
    pass


class QtClientIface(QObject):

    storage_api_added = Signal(str)
    storage_api_removed = Signal()

    def __init__(self, project, parent=None):
        super(QtClientIface, self).__init__(parent)
        self._target_name = "com.equinor.ERT.{}".format(project)
        self._bus = QtDBus.QDBusConnection.sessionBus()

    def _get_url(self):
        proxy = QtDBus.QDBusInterface(
            self._target_name,
            "/com/equinor/ERT/API/Storage",
            "com.equinor.ERT.api.storage",
        )
        reply = proxy.call("GetURL")
        if reply.errorName() == "org.freedesktop.DBus.Error.ServiceUnknown":
            raise ServiceUnknownError(reply.errorMessage())
        elif reply.errorName() != "":
            raise RuntimeError("{}: {}".format(reply.errorName(), reply.errorMessage()))
        return reply.arguments()[0]

    @Slot("QString", "QString", "QString")
    def _service_owner_changed(self, name, old, new):
        if old == "" and new != "":
            self.storage_api_added.emit(self._get_url())
        elif old != "" and new == "":
            self.storage_api_removed.emit()

    @Slot()
    def shutdown(self):
        proxy = QtDBus.QDBusInterface(
            self._target_name,
            "/com/equinor/ERT/API/Storage",
            "com.equinor.ERT.api.storage",
        )
        reply = proxy.call(QtDBus.QDBus.CallMode.BlockWithGui, "Shutdown")
        if reply.errorName():
            raise RuntimeError("{}: {}".format(reply.errorName(), reply.errorMessage()))

    def watch(self):
        watcher = QtDBus.QDBusServiceWatcher(self._target_name, self._bus, parent=self)
        watcher.serviceOwnerChanged.connect(self._service_owner_changed)

        # Is there an API running right now? If so, emit that.
        try:
            url = self._get_url()
            self.storage_api_added.emit(url)
        except ServiceUnknownError as e:
            # FIXME: log
            print("not found", e)
