from threading import Thread

from ert.server import ErtRPCServer
from ert_gui.shell import assertConfigLoaded, ErtShellCollection


class Server(ErtShellCollection):
    def __init__(self, parent):
        super(Server, self).__init__("server", parent)

        self.addShellFunction(name="start",
                              function=Server.startServer,
                              help_arguments="[port_number]",
                              help_message="Start the ERT RPC Server using the optional port number or a random port")

        self.addShellFunction(name="stop",
                              function=Server.stopServer,
                              help_message="Stop the ERT RPC Server")

        self.addShellFunction(name="inspect",
                              function=Server.inspect,
                              help_message="Shows information about the current job queue")

        self._server = None
        """ :type: ErtRPCServer """

    @assertConfigLoaded
    def startServer(self, line):
        try:
            port = int(line.strip())
        except ValueError:
            port = 0

        if self._server is None:
            self._server = ErtRPCServer(self.ert(), port=port)
            thread = Thread(name="Shell Server Thread")
            thread.daemon = True
            thread.run = self._server.start
            thread.start()
            print("Server running on host: '%s' and port: %d" % (self._server.host, self._server.port))
        else:
            print("A server is already running at host: '%s' and port: %d" % (self._server.host, self._server.port))

    def _stopServer(self):
        if self._server is not None:
            self._server.stop()
            self._server = None
            print("Server stopped")

    def stopServer(self, line):
        if self._server is not None:
            self._stopServer()
        else:
            print("No server to stop")

    def cleanup(self):
        self._stopServer()
        ErtShellCollection.cleanup(self)

    def inspect(self, line):
        if self._server is not None:
            print("Queue is running: %s" % self._server.isRunning())
            print("Running: %d" % self._server.getRunningCount())
            print("Failed: %d" % self._server.getFailedCount())
            print("Succeeded: %d" % self._server.getSuccessCount())
        else:
            print("No server to inspect")


