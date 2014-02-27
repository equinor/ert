from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB
from ert.enkf.data import EnkfConfigNode
from ert.enkf.enkf_fs import EnkfFs
from ert.enkf.enums.enkf_state_type_enum import EnkfStateType
from ert.enkf.observations import ObsVector
from ert.util import BoolVector


class PlotBlockVector(BaseCClass):

    def __init__(self, obs_vector, realization_number):
        assert isinstance(obs_vector, ObsVector)

        c_pointer = PlotBlockVector.cNamespace().alloc(obs_vector, realization_number)
        super(PlotBlockVector, self).__init__(c_pointer)


    def __len__(self):
        """ @rtype: int """
        return PlotBlockVector.cNamespace().size(self)


    def __getitem__(self, index):
        """ @rtype: double """
        assert isinstance(index, int)
        return PlotBlockVector.cNamespace().get(self, index)

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1


    def free(self):
        PlotBlockVector.cNamespace().free(self)



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("plot_block_vector", PlotBlockVector)
cwrapper.registerType("plot_block_vector_obj", PlotBlockVector.createPythonObject)
cwrapper.registerType("plot_block_vector_ref", PlotBlockVector.createCReference)

PlotBlockVector.cNamespace().free = cwrapper.prototype("void enkf_plot_blockvector_free(plot_block_vector)")
PlotBlockVector.cNamespace().alloc = cwrapper.prototype("c_void_p enkf_plot_blockvector_alloc(obs_vector, int)")
PlotBlockVector.cNamespace().size = cwrapper.prototype("int enkf_plot_blockvector_get_size(plot_block_vector)")
PlotBlockVector.cNamespace().get = cwrapper.prototype("double enkf_plot_blockvector_iget(plot_block_vector, int)")


