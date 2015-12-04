from ert.enkf import EnkfFsManager, EnkfFs, EnkfVarType


def initializeCase(ert, name, size):
    """
    @type ert: ert.enkf.enkf_main.EnKFMain
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