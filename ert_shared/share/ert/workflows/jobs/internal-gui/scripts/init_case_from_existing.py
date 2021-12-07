import warnings
from res.enkf import ErtScript
import deprecation
from ert_shared import __version__

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)


class InitCaseFromExistingJob(ErtScript):
    @deprecation.deprecated(
        deprecated_in="2.30",
        removed_in="2.37",
        current_version=__version__,
        details=(
            f"INIT_CASE_FROM_EXISTING is deprecated, if you rely on this functionality "
            f"please contact the ert team by creating an issue at "
            f"www.github.com/equinor/ert",
        ),
    )
    def run(self, source_case, target_case=None):
        ert = self.ert()
        source_fs = ert.getEnkfFsManager().getFileSystem(source_case)

        if target_case is None:
            target_fs = ert.getEnkfFsManager().getCurrentFileSystem()

        else:
            target_fs = ert.getEnkfFsManager().getFileSystem(target_case)

        ert.getEnkfFsManager().initializeCaseFromExisting(source_fs, 0, target_fs)
