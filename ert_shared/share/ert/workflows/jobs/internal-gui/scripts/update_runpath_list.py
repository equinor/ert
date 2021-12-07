import deprecation
import warnings
from ert_shared import __version__
from res.enkf import ErtScript

"""
This job is useful if you are running a workflow that requires the hook_manager runpath_list
to be populated but your are not running any simulations.
"""

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)


class UpdateRunpathListJob(ErtScript):
    @deprecation.deprecated(
        deprecated_in="2.30",
        removed_in="2.37",
        current_version=__version__,
        details=(
            f"UPDATE_RUNPATH_LIST is deprecated, if you rely on this functionality "
            f"please contact the ert team by creating an issue at "
            f"www.github.com/equinor/ert",
        ),
    )
    def run(self):
        ert = self.ert()

        realization_count = ert.getEnsembleSize()
        iteration = 0

        ecl_config = ert.eclConfig()
        model_config = ert.getModelConfig()
        basename_fmt = ecl_config.getEclBase()
        runpath_fmt = model_config.getRunpathAsString()
        hook_manager = ert.getHookManager()

        runpath_list = hook_manager.getRunpathList()

        runpath_list.clear()

        for realization_number in range(realization_count):

            if basename_fmt is not None:
                basename = basename_fmt % realization_number
            else:
                raise UserWarning("EclBase not set!")

            if model_config.runpathRequiresIterations():
                runpath = runpath_fmt % (realization_number, iteration)
            else:
                runpath = runpath_fmt % realization_number

            runpath_list.add(realization_number, iteration, runpath, basename)
