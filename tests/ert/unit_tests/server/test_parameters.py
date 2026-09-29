import io
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException

from ert.config import ErtConfig, Field
from ert.field_utils import ErtboxParameters, FieldFileFormat
from ert.server.endpoints.parameters import data_for_parameter, get_parameter_mean
from ert.storage import open_storage


def _field_parameter() -> Field:
    return Field(
        name="PORO",
        ertbox_params=ErtboxParameters(2, 2, 2),
        file_format=FieldFileFormat.ROFF_BINARY,
        forward_init=False,
        forward_init_file="init.roff",
        output_file="output.roff",
        grid_file="grid.roff",
        update_strategy="global",
    )


def _storage_for_parameter(parameter: object, values: xr.Dataset) -> Mock:
    ensemble = Mock()
    ensemble.experiment.parameter_configuration = {"PORO": parameter}
    ensemble.load_parameters.return_value = values
    storage = Mock()
    storage.get_ensemble.return_value = ensemble
    return storage


def test_that_parameter_mean_returns_the_requested_field_layer_mean() -> None:
    values = xr.Dataset(
        {
            "values": (
                ("realizations", "x", "y", "z"),
                np.array(
                    [
                        [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]],
                        [[[5.0, 50.0], [6.0, 60.0]], [[7.0, 70.0], [8.0, 80.0]]],
                    ]
                ),
            )
        }
    )
    storage = _storage_for_parameter(_field_parameter(), values)

    response = get_parameter_mean(storage=storage, ensemble_id=Mock(), key="PORO", z=1)

    np.testing.assert_array_equal(
        np.load(io.BytesIO(response.body)),
        np.array([[30.0, 40.0], [50.0, 60.0]]),
    )


@pytest.mark.parametrize(
    ("parameter", "layer"),
    [
        (object(), 0),
        (_field_parameter(), -1),
        (_field_parameter(), 2),
    ],
    ids=["non-field-parameter", "negative-layer", "layer-after-last"],
)
def test_that_parameter_mean_rejects_invalid_field_or_layer(
    parameter: object, layer: int
) -> None:
    values = xr.Dataset(
        {"values": (("realizations", "x", "y", "z"), np.zeros((1, 2, 2, 2)))}
    )
    storage = _storage_for_parameter(parameter, values)

    with pytest.raises(HTTPException) as error:
        get_parameter_mean(storage=storage, ensemble_id=Mock(), key="PORO", z=layer)

    assert error.value.status_code == 404
    assert error.value.detail == "Data not found"


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
@pytest.mark.xdist_group(name="uses_heat_equation_storage")
def test_that_asking_for_non_existent_key_doesnt_raise(
    symlinked_heat_equation_storage_es,
):
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="r") as storage:
        ensemble = next(storage.ensembles)
        assert "variable" not in ensemble.experiment.parameter_configuration
        df = data_for_parameter(ensemble, "variable")
        assert df.empty


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward model")
@pytest.mark.xdist_group(name="uses_heat_equation_storage")
def test_that_asking_for_existing_key_with_group_returns_data(
    symlinked_heat_equation_storage_es,
):
    config = ErtConfig.from_file("config.ert")
    with open_storage(config.ens_path, mode="r") as storage:
        ensemble = next(storage.ensembles)
        assert "t" in ensemble.experiment.parameter_configuration
        df = data_for_parameter(ensemble, "t")
        assert not df.empty
