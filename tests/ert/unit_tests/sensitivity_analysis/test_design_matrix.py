import pandas as pd

from ert.config import DesignMatrix


def test_reading_design_matrix(tmp_path):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {"REAL": [0, 1, 2], "a": [1, 2, 3], "b": [0, 2, 0], "c": [3, 1, 3]}
    )
    default_sheet_df = pd.DataFrame([["one", 1], ["b", 4], ["d", 6]])
    with pd.ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )
    design_matrix = DesignMatrix(design_path, "DesignSheet01", "DefaultValues")
    design_matrix.read_design_matrix()
    print("\n The design matrix:\n", design_matrix.design_matrix_df)
