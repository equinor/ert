import queue
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

from ert.config import ConfigWarning, ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.mode_definitions import (
    ENIF_MODE,
)
from ert.run_models import create_model
from ert.storage import open_storage
from tests.ert.conftest import _create_design_matrix


@pytest.mark.slow
def test_that_enif_update_does_not_update_design_matrix_parameters(
    copy_case,
):
    """
    Runs EnIF on poly.ert with hardcoded parameter "a" values in design matrix,
    then expect parameter "a" values to remain unchanged after second run
    """
    num_realizations = 10

    copy_case("poly_example")
    config_file = Path("poly.ert")

    _create_design_matrix(
        "poly_design.xlsx",
        pl.DataFrame(
            {
                "REAL": list(range(num_realizations)),
                "a": [1] * num_realizations,  # ["a"].to_list(),
            }
        ),
    )

    with open(config_file, "a", encoding="utf-8") as fh:
        fh.write(f"NUM_REALIZATIONS {num_realizations}\n")
        fh.write("RANDOM_SEED 123456789\n")
        fh.write("DESIGN_MATRIX poly_design.xlsx\n")

    evaluator_server_config = EvaluatorServerConfig()
    with pytest.warns(
        ConfigWarning, match=r"Parameters .* will be overridden by design matrix"
    ):
        ert_config_with_dm = ErtConfig.from_file("poly.ert")

    enif_with_dm = create_model(
        ert_config_with_dm,
        args=Namespace(
            mode=ENIF_MODE,
            experiment_name="enif_with_dm",
            target_ensemble="ens_with_dm_%d",
        ),
        status_queue=queue.SimpleQueue(),
    )

    enif_with_dm.start_simulations_thread(evaluator_server_config)

    with open_storage(enif_with_dm.storage_path, mode="r") as storage:
        experiment_with_dm = storage.get_experiment_by_name("enif_with_dm")

        prior_with_dm = experiment_with_dm.get_ensemble_by_name("ens_with_dm_0")
        posterior_with_dm = experiment_with_dm.get_ensemble_by_name("ens_with_dm_1")

        prior_with_dm_a = prior_with_dm.load_parameters("a")["a"]
        posterior_with_dm_a = posterior_with_dm.load_parameters("a")["a"]

        assert posterior_with_dm_a.equals(prior_with_dm_a)
