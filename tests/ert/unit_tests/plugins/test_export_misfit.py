import sys

import pandas as pd
import pytest

from ert.exceptions import StorageError
from ert.plugins import ErtPluginManager
from ert.plugins.hook_implementations.workflows.export_misfit_data import (
    ExportMisfitDataJob,
)


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="https://github.com/equinor/ert/issues/7533",
)
def test_export_misfit(snake_oil_default_storage, snapshot):
    ExportMisfitDataJob().run(snake_oil_default_storage, [])
    result = pd.read_hdf("misfit.hdf")
    snapshot.assert_match(
        result.to_csv(),
        "csv_data.csv",
    )


def test_export_misfit_no_responses_in_storage(new_ensemble):
    with pytest.raises(StorageError, match="No responses loaded"):
        ExportMisfitDataJob().run(new_ensemble, [])


def test_export_misfit_data_job_is_loaded():
    pm = ErtPluginManager()
    assert "EXPORT_MISFIT_DATA" in pm.get_ertscript_workflows().get_workflows()
