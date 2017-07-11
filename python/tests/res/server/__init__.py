from threading import Thread

from res.server import ErtRPCServer
from res.test import ErtTest





class RPCServiceContext(object):
    def __init__(self, test_name, model_config, store_area=False):
        self._test_context = ErtTest(test_name, model_config, store_area=store_area)
        self._server = ErtRPCServer(self._test_context.getErt())

    def __enter__(self):
        """ @rtype: ErtRPCServer"""
        thread = Thread(name="ErtRPCServerTest")
        thread.run = self._server.start
        thread.start()

        return self._server

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._server.stop()
        del self._server
        del self._test_context
        return False
