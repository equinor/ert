import warnings
import deprecation

from ert_shared import __version__
from res.enkf import ErtScript

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)


class CreateCaseJob(ErtScript):
    @deprecation.deprecated(
        deprecated_in="2.30",
        removed_in="2.37",
        current_version=__version__,
        details=(
            f"CREATE_CASE is deprecated, if you rely on this functionality "
            f"please contact the ert team by creating an issue at "
            f"www.github.com/equinor/ert",
        ),
    )
    def run(self, case_name):
        ert = self.ert()
        fs = ert.getEnkfFsManager().getFileSystem(case_name)
