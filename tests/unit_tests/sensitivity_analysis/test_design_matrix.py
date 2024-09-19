import pytest

from ert.config import ErtConfig
from ert.sensitivity_analysis.design_matrix import (
    initialize_parameters,
    read_design_matrix,
)
from ert.storage import open_storage


@pytest.mark.usefixtures("copy_poly_case")
def test_design_matrix(copy_poly_case):
    config_text = """
QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 50

RUNPATH poly_out/realization-<IENS>/iter-<ITER>

OBS_CONFIG observations
REALIZATION_MEMORY 50mb

DESIGN_MATRIX design_matrix.xlsx
NUM_REALIZATIONS 100
MIN_REALIZATIONS 1

GEN_DATA POLY_RES RESULT_FILE:poly.out

INSTALL_JOB poly_eval POLY_EVAL
FORWARD_MODEL poly_eval
"""
    with open("my_config.ert", "w", encoding="utf-8") as fhandle:
        fhandle.write(config_text)
    ert_config = ErtConfig.from_file("my_config.ert")
    with open_storage(ert_config.ens_path, mode="w") as ert_storage:
        design_frame = read_design_matrix(ert_config, "design_matrix.xlsx")
        _ensemble = initialize_parameters(
            design_frame,
            ert_storage,
            ert_config,
            "my_experiment",
            "my_ensemble",
        )
