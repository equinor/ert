from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB
from ert.util import Matrix


class PcaPlotVector(BaseCClass):

    def __init__(self, component, principal_component_matrix, observation_principal_component_matrix):
        assert isinstance(component, int)
        assert isinstance(principal_component_matrix, Matrix)
        assert isinstance(observation_principal_component_matrix, Matrix)

        c_pointer = PcaPlotVector.cNamespace().alloc(component, principal_component_matrix, observation_principal_component_matrix)
        super(PcaPlotVector, self).__init__(c_pointer)


    def __len__(self):
        """ @rtype: int """
        return PcaPlotVector.cNamespace().size(self)


    def __getitem__(self, index):
        """ @rtype: double """
        assert isinstance(index, int)
        return PcaPlotVector.cNamespace().get(self, index)

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1

    def getObservation(self):
        """ @rtype: double """
        return PcaPlotVector.cNamespace().get_obs(self)

    def free(self):
        PcaPlotVector.cNamespace().free(self)



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("pca_plot_vector", PcaPlotVector)
cwrapper.registerType("pca_plot_vector_obj", PcaPlotVector.createPythonObject)
cwrapper.registerType("pca_plot_vector_ref", PcaPlotVector.createCReference)

PcaPlotVector.cNamespace().alloc   = cwrapper.prototype("c_void_p pca_plot_vector_alloc(pca_plot_vector, int)")
PcaPlotVector.cNamespace().free    = cwrapper.prototype("void pca_plot_vector_free(pca_plot_vector)")
PcaPlotVector.cNamespace().size    = cwrapper.prototype("int pca_plot_vector_get_size(pca_plot_vector)")
PcaPlotVector.cNamespace().get     = cwrapper.prototype("double pca_plot_vector_iget_sim_value(pca_plot_vector, int)")
PcaPlotVector.cNamespace().get_obs = cwrapper.prototype("double pca_plot_vector_get_obs_value(pca_plot_vector)")


