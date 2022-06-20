import numpy
from ecl.util.util import IntVector
from pandas import DataFrame
from res.enkf import EnKFMain
from res.enkf.enums import RealizationStateEnum
from res.enkf.plot_data import EnsemblePlotGenData


class GenDataCollector:
    @staticmethod
    def loadGenData(
        ert: EnKFMain,
        case_name: str,
        key: str,
        report_step: int,
        realization_index: int = None,
    ) -> DataFrame:
        """
        In the returned dataframe the realisation index runs along the
        rows, and the gen_data element index runs vertically along the
        columns.
        """
        fs = ert.getEnkfFsManager().getFileSystem(case_name, read_only=True)
        realizations = fs.realizationList(RealizationStateEnum.STATE_HAS_DATA)
        if realization_index:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = IntVector.active_list(str(realization_index))

        config_node = ert.ensembleConfig().getNode(key)
        config_node.getModelConfig()

        ensemble_data = EnsemblePlotGenData(config_node, fs, report_step)
        data_array = ensemble_data.getRealizations(realizations)

        realizations = numpy.array(realizations)
        return DataFrame(data=data_array, columns=realizations)
