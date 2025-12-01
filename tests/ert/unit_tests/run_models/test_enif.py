import queue
from argparse import Namespace
from pathlib import Path

import polars as pl

from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    EVALUATE_ENSEMBLE_MODE,
    MANUAL_ENIF_UPDATE_MODE,
)
from ert.run_models import create_model
from ert.storage import open_storage
from tests.ert.conftest import _create_design_matrix


def test_that_enif_update_from_ensemble_experiment_is_not_affected_by_unupdated_params(
    copy_case,
):
    num_realizations = 10

    copy_case("poly_example")
    _create_design_matrix(
        "poly_design.xlsx",
        pl.DataFrame(
            {"REAL": list(range(num_realizations)), "a": list(range(num_realizations))}
        ),
        pl.DataFrame([], orient="row"),
    )

    # Make one ertconfig without DM
    # Make one ertconfig with DM that adds one non-updateable categorical param
    config_file = Path("poly.ert")

    with open(config_file, "a", encoding="utf-8") as fh:
        fh.write("DESIGN_MATRIX poly_design.xlsx DEFAULT_SHEET:DefaultSheet\n")
        fh.write(f"NUM_REALIZATIONS {num_realizations}\n")

    ert_config_with_dm = ErtConfig.from_file("poly.ert")
    ert_config_without_dm = ert_config_with_dm.model_copy(deep=True)
    ert_config_without_dm.analysis_config.design_matrix = None

    evaluator_server_config = EvaluatorServerConfig()
    ensemble_experiment = create_model(
        ert_config_without_dm,
        args=Namespace(
            mode=ENSEMBLE_EXPERIMENT_MODE,
            experiment_name="dummy",
            current_ensemble="ens%d",
        ),
        status_queue=queue.SimpleQueue(),
    )
    ensemble_experiment.start_simulations_thread(evaluator_server_config)

    with open_storage(ensemble_experiment.storage_path, mode="r") as storage:
        previous_experiment = storage.get_experiment_by_name("dummy")
        previous_ensemble = next(iter(previous_experiment.ensembles))
        assert previous_ensemble is not None
        ensemble_id_to_update = str(previous_ensemble.id)

    manual_update_model_with_dm = create_model(
        ert_config_with_dm,
        args=Namespace(
            mode=MANUAL_ENIF_UPDATE_MODE,
            ensemble_id=ensemble_id_to_update,
            target_ensemble="design_matrix_ensemble%d",
        ),
        status_queue=queue.SimpleQueue(),
    )
    manual_update_model_with_dm.start_simulations_thread(evaluator_server_config)

    manual_update_model_without_dm = create_model(
        ert_config_with_dm,
        args=Namespace(
            mode=MANUAL_ENIF_UPDATE_MODE,
            ensemble_id=ensemble_id_to_update,
            target_ensemble="nodesign_ensemble%d",
        ),
        status_queue=queue.SimpleQueue(),
    )
    manual_update_model_without_dm.start_simulations_thread(evaluator_server_config)

    with open_storage(manual_update_model_with_dm.storage_path, mode="r") as storage:
        updated_ens_with_dm = next(
            e for e in storage.ensembles if e.name == "design_matrix_ensemble1"
        )
        updated_ens_without_dm = next(
            e for e in storage.ensembles if e.name == "nodesign_ensemble1"
        )

    # Evaluate to get responses
    evaluate_without_dm = create_model(
        ert_config_without_dm,
        args=Namespace(
            mode=EVALUATE_ENSEMBLE_MODE, ensemble_id=str(updated_ens_without_dm.id)
        ),
        status_queue=queue.SimpleQueue(),
    )

    evaluate_without_dm.start_simulations_thread(evaluator_server_config)

    # Evaluate to get responses
    evaluate_with_dm = create_model(
        ert_config_with_dm,
        args=Namespace(
            mode=EVALUATE_ENSEMBLE_MODE, ensemble_id=str(updated_ens_with_dm.id)
        ),
        status_queue=queue.SimpleQueue(),
    )

    evaluate_with_dm.start_simulations_thread(evaluator_server_config)

    with open_storage(manual_update_model_with_dm.storage_path, mode="r") as storage:
        updated_ens_with_dm = next(
            e for e in storage.ensembles if e.name == "design_matrix_ensemble1"
        )
        updated_ens_without_dm = next(
            e for e in storage.ensembles if e.name == "nodesign_ensemble1"
        )

        active_reals = list(range(10))

        assert updated_ens_with_dm.load_responses("gen_data", active_reals).equals(
            updated_ens_without_dm.load_responses("gen_data", active_reals)
        )
