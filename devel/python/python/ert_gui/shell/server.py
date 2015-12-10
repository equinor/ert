import socket
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
            try:
                self._server = ErtRPCServer(self.ert(), port=port)
            except socket.error as e:
                print("Unable to start the server on port: %d" % port)
            else:
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
            if self._server.isRunning():
                print("Waiting..: %d" % self._server.getWaitingCount())
                print("Running..: %d" % self._server.getRunningCount())
                print("Failed...: %d" % self._server.getFailedCount())
                print("Succeeded: %d" % self._server.getSuccessCount())
                print("Batch#...: %d" % self._server.getBatchNumber())
            else:
                print("Server is not running any simulations")
        else:
            print("No server is not available")


