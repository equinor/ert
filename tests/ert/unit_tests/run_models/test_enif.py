import queue
from argparse import Namespace
from pathlib import Path

from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from ert.run_models import create_model
from ert.storage import open_storage


def test_that_enif_update_from_ensemble_experiment_is_not_affected_by_unupdated_params(
    copy_poly_case_with_design_matrix,
):
    num_realizations = 10
    a_values = list(range(num_realizations))

    copy_poly_case_with_design_matrix(
        design_dict={
            "REAL": list(range(num_realizations)),
            "a": a_values,
            "category": 5 * ["cat1"] + 5 * ["cat2"],
        },
        default_list=[["b", 1], ["c", 2]],
    )
    config_file = Path("poly.ert")

    Path("some_template.txt").write_text("<IENS>", encoding="utf-8")

    with open(config_file, "a", encoding="utf-8") as fh:
        fh.write(f"\nNUM_REALIZATIONS {num_realizations}\n")

    ert_config = ErtConfig.from_file("poly.ert")
    evaluator_server_config = EvaluatorServerConfig()
    ensemble_experiment = create_model(
        ert_config,
        args=Namespace(
            mode=ENSEMBLE_EXPERIMENT_MODE,
            experiment_name="dummy",
            current_ensemble="ens%d",
        ),
        status_queue=queue.SimpleQueue(),
    )
    ensemble_experiment.run_experiment(evaluator_server_config)

    previous_experiment = ensemble_experiment._storage.get_experiment_by_name("dummy")
    previous_ensemble = next(iter(previous_experiment.ensembles))
    assert previous_ensemble is not None
    ensemble_id_to_update = str(previous_ensemble.id)
    ensemble_experiment._storage.close()

    # Construct the ManualUpdate runmodel with the previous ensemble
    manual_update_model = create_model(
        ert_config,
        args=Namespace(
            mode=ENSEMBLE_EXPERIMENT_MODE,
            ensemble_id=ensemble_id_to_update,
            target_ensemble="updated_ens%d",
        ),
        status_queue=queue.SimpleQueue(),
    )

    # Executing this will clear the env and close the storage,
    # as opposed to a direct invocation of .run_experiment (at time of writing)
    manual_update_model.start_simulations_thread(evaluator_server_config)

    storage = open_storage(manual_update_model.storage_path, mode="r")
    manual_update_exp = next(
        e for e in storage.experiments if e.name.startswith("Manual update")
    )
    posterior_ens = manual_update_exp.get_ensemble_by_name("updated_ens1")
    assert posterior_ens is not None
    storage.close()
