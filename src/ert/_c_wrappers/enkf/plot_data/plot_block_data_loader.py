from typing import TYPE_CHECKING

from ecl.util.util import DoubleVector, ThreadPool

from ert._c_wrappers.enkf import NodeId
from ert._c_wrappers.enkf.data import EnkfNode
from ert._c_wrappers.enkf.enums import ErtImplType, RealizationStateEnum
from ert._c_wrappers.enkf.plot_data.plot_block_data import PlotBlockData
from ert._c_wrappers.enkf.plot_data.plot_block_vector import PlotBlockVector

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import BlockObservation, EnkfFs, ObsVector


class PlotBlockDataLoader:
    def __init__(self, obs_vector: "ObsVector"):
        if obs_vector is None:
            raise ValueError(
                "Cannot construct PlotBlockDataLoader without obs_vector, was None."
            )
        super().__init__()
        self.__obs_vector = obs_vector
        self.__permutation_vector = None

    def getBlockObservation(self, report_step: int) -> "BlockObservation":
        return self.__obs_vector.getNode(report_step)

    def getDepthValues(self, report_step) -> DoubleVector:
        block_obs = self.getBlockObservation(report_step)

        depth = DoubleVector()
        for index in block_obs:
            value = block_obs.getDepth(index)
            depth.append(value)

        return depth

    def load(self, fs: "EnkfFs", report_step: int) -> PlotBlockData:
        state_map = fs.getStateMap()

        ens_mask = state_map.selectMatching(RealizationStateEnum.STATE_HAS_DATA)

        depth = self.getDepthValues(report_step)

        self.__permutation_vector = depth.permutationSort()
        depth.permute(self.__permutation_vector)

        plot_block_data = PlotBlockData(depth)

        thread_pool = ThreadPool()
        for index, mask in enumerate(ens_mask):
            if mask:
                thread_pool.addTask(
                    self.loadVector, plot_block_data, fs, report_step, index
                )

        thread_pool.nonBlockingStart()
        thread_pool.join()

        return plot_block_data

    def loadVector(
        self,
        plot_block_data: PlotBlockData,
        fs: "EnkfFs",
        report_step: int,
        realization_number: int,
    ) -> None:
        config_node = self.__obs_vector.getConfigNode()

        is_private_container = (
            config_node.getImplementationType() == ErtImplType.CONTAINER
        )
        data_node = EnkfNode(config_node, private=is_private_container)

        node_id = NodeId(report_step, realization_number)

        if data_node.tryLoad(fs, node_id):
            block_obs = self.getBlockObservation(report_step)

            data = DoubleVector()
            for index in range(len(block_obs)):
                value = block_obs.getData(data_node.valuePointer(), index, node_id)
                data.append(value)
            data.permute(self.__permutation_vector)

            plot_block_vector = PlotBlockVector(realization_number, data)
            plot_block_data.addPlotBlockVector(plot_block_vector)
