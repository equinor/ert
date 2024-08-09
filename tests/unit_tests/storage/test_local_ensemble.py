import numpy as np
import pytest
import xarray as xr

from ert.config import (
    GenKwConfig,
)
from ert.storage import open_storage


def test_that_save_parameter_throws_exception_when_dataset_contains_nan(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        uniform_parameter = GenKwConfig(
            name="PARAMETER",
            forward_init=False,
            template_file="",
            transfer_function_definitions=[
                "KEY1 UNIFORM 0 1",
            ],
            output_file="kw.txt",
        )
        experiment = storage.create_experiment(parameters=[uniform_parameter])
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=1)

        with pytest.raises(
            expected_exception=ValueError, match="Dataset contains one or more 'nan'"
        ):

            ensemble.save_parameters(
                "PARAMETER",
                0,
                dataset=xr.Dataset(
                    {
                        "values": [0.7, np.nan, 0.2, 0.01],
                    }
                ),
            )
