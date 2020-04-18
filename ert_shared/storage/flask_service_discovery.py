import socket
import os
import dbus
import dbus.service
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib
from multiprocessing import Process

BUS_IFACE = "com.equinor.ERT.api.storage"
BUS_NAME_FMT = "com.equinor.ERT.{}"
BUS_PATH = "/com/equinor/ERT/API/Storage"


class StorageAPIObject(dbus.service.Object):
    def __init__(self, bus_name, path, url, loop, api_thread):
        dbus.service.Object.__init__(self, object_path=path, bus_name=bus_name)
        self._url = url
        self._loop = loop
        self._api_thread = api_thread

    @dbus.service.method(
        BUS_IFACE, in_signature="", out_signature="s"
    )
    def GetURL(self):
        return self._url

    @dbus.service.method(
        BUS_IFACE, in_signature="", out_signature=""
    )
    def Shutdown(self):
        self._api_thread.terminate()
        self._loop.quit()


def flask_run_with_service_discovery(host, port, app, project):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, int(port)))
    sock.listen()

    # XXX: a hack to get flask to pick up our created socket, for more serious
    # application servers, this would be passed explicitly to the server
    # interface
    os.environ["WERKZEUG_SERVER_FD"] = str(sock.fileno())

    app.config["SERVER_NAME"] = "%s:%d" % (sock.getsockname())

    DBusGMainLoop(set_as_default=True)
    mainloop = GLib.MainLoop()
    session_bus = dbus.SessionBus()
    name = dbus.service.BusName(BUS_NAME_FMT.format(project), session_bus)

    app_thread = Process(
        target=app.run, args=(host, "{}".format(port)), kwargs={"debug": False}
    )

    address, port = sock.getsockname()
    StorageAPIObject(
        name,
        BUS_PATH,
        "http://{}:{}/".format(address, port),
        mainloop,
        app_thread,
    )
    app_thread.start()
    mainloop.run()
    app_thread.join()
