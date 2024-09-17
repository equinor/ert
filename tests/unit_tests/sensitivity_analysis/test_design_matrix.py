import pytest

from ert.config import ErtConfig
from ert.sensitivity_analysis.design_matrix import (
    initialize_parameters,
    read_design_matrix,
)
from ert.storage import open_storage


@pytest.mark.usefixtures("copy_poly_case")
def test_design_matrix(copy_poly_case):
    ert_config = ErtConfig.from_file("poly.ert")
    with open_storage(ert_config.ens_path, mode="w") as ert_storage:
        design_frame = read_design_matrix(ert_config, "design_matrix.xlsx")
        _ensemble = initialize_parameters(
            design_frame,
            ert_storage,
            ert_config,
            "my_experiment",
            "my_ensemble",
        )
