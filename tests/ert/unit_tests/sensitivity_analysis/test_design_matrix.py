import time

import pandas as pd
import pytest

from ert.config import ErtConfig


@pytest.mark.usefixtures("copy_poly_case")
def test_reading_design_matrix(copy_poly_case):
    design_matrix_df = pd.DataFrame(
        {"REAL": [0, 1, 2], "a": [1, 2, 3], "b": [0, 2, 0], "c": [3, 1, 3]}
    )
    default_sheet_df = pd.DataFrame()
    with pd.ExcelWriter("design_matrix.xlsx") as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(xl_write, index=False, sheet_name="DefaultValues")

    with open("poly.ert", "a", encoding="utf-8") as fhandle:
        fhandle.write(
            "DESIGN_MATRIX design_matrix.xlsx DESIGN_SHEET:DesignSheet01 DEFAULT_SHEET:DefaultValues"
        )
    ert_config = ErtConfig.from_file("poly.ert")
    parameter_configurations = ert_config.ensemble_config.parameter_configuration
    t = time.perf_counter()
    _design_frame = ert_config.analysis_config.design_matrix.read_design_matrix(
        parameter_configurations
    )
    print(f"Read design matrix time_used {(time.perf_counter() - t):.4f}s")
