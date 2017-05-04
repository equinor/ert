from threading import Thread

from res.enkf import EnkfFsManager, EnkfFs, EnkfVarType
from res.server import ErtRPCServer
from res.test import ErtTest


def initializeCase(ert, name, size):
    """
    @type ert: res.enkf.enkf_main.EnKFMain
    @type name: str
    @type size: int
    @rtype:
    """
    current_fs = ert.getEnkfFsManager().getCurrentFileSystem()
    fs = ert.getEnkfFsManager().getFileSystem(name)
    ert.getEnkfFsManager().switchFileSystem(fs)
    parameters = ert.ensembleConfig().getKeylistFromVarType(EnkfVarType.PARAMETER)
    ert.getEnkfFsManager().initializeFromScratch(parameters, 0, size - 1)

    ert.getEnkfFsManager().switchFileSystem(current_fs)
    return fs



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
