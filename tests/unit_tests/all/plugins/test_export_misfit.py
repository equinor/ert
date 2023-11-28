import pandas as pd
import pytest

from ert.exceptions import StorageError
from ert.shared.hook_implementations.workflows.export_misfit_data import (
    ExportMisfitDataJob,
)
from ert.shared.plugins import ErtPluginManager


def test_export_misfit(snake_oil_case_storage, snake_oil_default_storage, snapshot):
    ExportMisfitDataJob(
        snake_oil_case_storage, storage=None, ensemble=snake_oil_default_storage
    ).run()
    result = pd.read_hdf("misfit.hdf")
    snapshot.assert_match(
        result.to_csv(),
        "csv_data.csv",
    )


def test_export_misfit_no_responses_in_storage(poly_case, new_ensemble):
    with pytest.raises(StorageError, match="No responses loaded"):
        ExportMisfitDataJob(poly_case, storage=None, ensemble=new_ensemble).run()


def test_export_misfit_data_job_is_loaded():
    pm = ErtPluginManager()
    assert "EXPORT_MISFIT_DATA" in pm.get_installable_workflow_jobs()
