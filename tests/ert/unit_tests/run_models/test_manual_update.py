import queue
from argparse import Namespace
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    MANUAL_ENIF_UPDATE_MODE,
    MANUAL_UPDATE_MODE,
)
from ert.run_models import create_model
from ert.storage import open_storage


@pytest.mark.integration_test
@pytest.mark.parametrize("mode", [MANUAL_UPDATE_MODE, MANUAL_ENIF_UPDATE_MODE])
def test_that_manual_update_from_ensemble_experiment_supports_all_update_modes(
    copy_poly_case, mode
):
    config_file = Path("poly.ert")

    Path("some_template.txt").write_text("<IENS>", encoding="utf-8")

    run_template = "RUN_TEMPLATE some_template.txt TEMPLATE_FILE:poly.tmpl"
    with open(config_file, "a", encoding="utf-8") as fh:
        fh.write(f"\n{run_template}\n")
        fh.write("\nNUM_REALIZATIONS 2\n")

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
    ensemble_experiment.start_simulations_thread(evaluator_server_config)

    with open_storage(ensemble_experiment.storage_path, mode="r") as storage:
        previous_experiment = storage.get_experiment_by_name("dummy")
        previous_ensemble = next(iter(previous_experiment.ensembles))
        assert previous_ensemble is not None
        ensemble_id_to_update = str(previous_ensemble.id)

    # Construct the ManualUpdate runmodel with the previous ensemble
    manual_update_model = create_model(
        ert_config,
        args=Namespace(
            mode=mode,
            ensemble_id=ensemble_id_to_update,
            target_ensemble="updated_ens%d",
        ),
        status_queue=queue.SimpleQueue(),
    )

    assert manual_update_model.ert_templates == ensemble_experiment.ert_templates
    # Executing this will clear the env and close the storage,
    # as opposed to a direct invocation of .run_experiment (at time of writing)
    manual_update_model.start_simulations_thread(evaluator_server_config)

    with open_storage(manual_update_model.storage_path, mode="r") as storage:
        manual_update_exp = next(
            e for e in storage.experiments if e.name.startswith("Manual update")
        )
        posterior_ens = manual_update_exp.get_ensemble_by_name("updated_ens1")
        assert posterior_ens is not None
