import warnings
from collections import OrderedDict
import os
from res.enkf import ErtScript, RealizationStateEnum
from ecl.util.util import BoolVector
import deprecation
from ert_shared import __version__

"""
This job exports misfit data into a chosen file or to the default gen_kw export file (parameters.txt)
"""
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)


class ExportMisfitDataJob(ErtScript):
    @deprecation.deprecated(
        deprecated_in="2.30",
        removed_in="2.37",
        current_version=__version__,
        details=(
            f"EXPORT_MISFIT_DATA is deprecated, if you rely on this functionality "
            f"please contact the ert team by creating an issue at "
            f"www.github.com/equinor/ert",
        ),
    )
    def run(self, target_file=None):
        ert = self.ert()
        fs = ert.getEnkfFsManager().getCurrentFileSystem()

        if target_file is None:
            target_file = ert.getModelConfig().getGenKWExportFile()

        runpath_list = ert.getHookManager().getRunpathList()

        active_list = self.createActiveList(fs)

        for runpath_node in runpath_list:
            if runpath_node.realization in active_list:

                if not os.path.exists(runpath_node.runpath):
                    os.makedirs(runpath_node.runpath)

                target_path = os.path.join(runpath_node.runpath, target_file)

                parameters = self.parseTargetFile(target_path)

                misfit_sum = 0.0
                for obs_vector in ert.getObservations():
                    misfit = obs_vector.getTotalChi2(fs, runpath_node.realization)

                    key = "MISFIT:%s" % obs_vector.getObservationKey()
                    parameters[key] = misfit

                    misfit_sum += misfit

                parameters["MISFIT:TOTAL"] = misfit_sum

                self.dumpParametersToTargetFile(parameters, target_path)

    def parseTargetFile(self, target_path):
        parameters = OrderedDict()

        if os.path.exists(target_path) and os.path.isfile(target_path):
            with open(target_path, "r") as input_file:
                lines = input_file.readlines()

                for line in lines:
                    tokens = line.split()

                    if len(tokens) == 2:
                        parameters[tokens[0]] = tokens[1]
                    else:
                        raise UserWarning(
                            "The file '%s' contains errors. Expected format for each line: KEY VALUE"
                            % target_path
                        )

        return parameters

    def dumpParametersToTargetFile(self, parameters, target_path):
        with open(target_path, "w") as output:
            for key in parameters:
                output.write("%s %s\n" % (key, parameters[key]))

    def createActiveList(self, fs):
        state_map = fs.getStateMap()
        ens_mask = BoolVector(False, self.ert().getEnsembleSize())
        state_map.selectMatching(ens_mask, RealizationStateEnum.STATE_HAS_DATA)
        active_list = BoolVector.createActiveList(ens_mask)

        return active_list
