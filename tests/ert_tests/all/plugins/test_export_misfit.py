import pandas as pd
import pytest

from ert._c_wrappers.enkf import EnKFMain
from ert.exceptions import StorageError
from ert.shared.hook_implementations.workflows.export_misfit_data import (
    ExportMisfitDataJob,
)
from ert.shared.plugins import ErtPluginManager


def test_export_misfit(setup_case, snapshot):
    res_config = setup_case("snake_oil", "snake_oil.ert")
    ert = EnKFMain(res_config)
    ExportMisfitDataJob(ert).run()
    result = pd.read_hdf("misfit.hdf")
    snapshot.assert_match(
        result.to_csv(),
        "csv_data.csv",
    )


def test_export_misfit_no_responses_in_storage(poly_case):
    with pytest.raises(StorageError, match="No responses loaded"):
        ExportMisfitDataJob(poly_case).run()


def test_export_misfit_data_job_is_loaded():
    pm = ErtPluginManager()
    assert "EXPORT_MISFIT_DATA" in pm.get_installable_workflow_jobs()
