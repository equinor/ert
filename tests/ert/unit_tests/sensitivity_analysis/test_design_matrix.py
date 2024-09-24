import pandas as pd
import pytest

from ert.config import ErtConfig
from ert.sensitivity_analysis.design_matrix import (
    initialize_parameters,
    read_design_matrix,
)
from ert.storage import open_storage


@pytest.mark.usefixtures("copy_poly_case")
def test_design_matrix(copy_poly_case):
    design_matrix_df = pd.DataFrame(
        {"REAL": [0, 1, 2], "a": [1, 2, 3], "b": [0, 2, 0], "c": [3, 1, 3]}
    )
    default_sheet_df = pd.DataFrame()
    with pd.ExcelWriter("design_matrix.xlsx") as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(xl_write, index=False, sheet_name="DefaultValues")

    with open("poly.ert", "a", encoding="utf-8") as fhandle:
        fhandle.write("DESIGN_MATRIX design_matrix.xlsx")
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
