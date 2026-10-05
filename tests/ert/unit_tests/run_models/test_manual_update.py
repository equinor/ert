import queue
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ert.config import ErtConfig, GenKwConfig, LocalizationType
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.mode_definitions import (
    ENSEMBLE_EXPERIMENT_MODE,
    MANUAL_ENIF_UPDATE_MODE,
    MANUAL_UPDATE_MODE,
)
from ert.run_models import create_model
from ert.run_models.manual_update import ManualUpdate
from ert.run_models.manual_update_enif import ManualUpdateEnIF
from ert.run_models.run_model import ErtRunError
from ert.run_models.update_run_model import UpdateRunModel
from ert.storage import Storage, open_storage


@pytest.mark.slow
@pytest.mark.parametrize("mode", [MANUAL_UPDATE_MODE, MANUAL_ENIF_UPDATE_MODE])
def test_that_manual_update_from_ensemble_experiment_supports_all_update_modes(
    copy_poly_case, mode
):
    config_file = Path("poly.ert")

    Path("some_template.txt").write_text("<IENS>", encoding="utf-8")

    run_template = "RUN_TEMPLATE some_template.txt TEMPLATE_FILE:poly.tmpl"
    with Path(config_file).open("a", encoding="utf-8") as fh:
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
            experiment_name="my manual update",
        ),
        status_queue=queue.SimpleQueue(),
    )

    assert manual_update_model.ert_templates == ensemble_experiment.ert_templates
    # Executing this will clear the env and close the storage,
    # as opposed to a direct invocation of .run_experiment (at time of writing)
    manual_update_model.start_simulations_thread(evaluator_server_config)

    with open_storage(manual_update_model.storage_path, mode="r") as storage:
        manual_update_exp = storage.get_experiment_by_name("my manual update")
        posterior_ens = manual_update_exp.get_ensemble_by_name("updated_ens1")
        assert posterior_ens is not None


@pytest.mark.parametrize(
    ("update_strategy", "should_update"),
    [(None, False), (LocalizationType.GLOBAL, True)],
    ids=["disabled", "enabled"],
)
@pytest.mark.parametrize("model_cls", [ManualUpdate, ManualUpdateEnIF])
def test_that_manual_update_validates_prior_for_updatable_parameters(
    storage: Storage,
    update_strategy: LocalizationType | None,
    should_update: bool,
    model_cls,
):
    parameter = GenKwConfig(
        name="PARAMETER",
        distribution={"name": "uniform", "min": 0.8, "max": 1.2},
        update_strategy=update_strategy,
    )
    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [parameter.model_dump(mode="json")]
        }
    )
    prior = storage.create_ensemble(experiment, name="prior", ensemble_size=1)

    model = MagicMock(spec=model_cls)
    model._prior = prior
    model._validate_has_updatable_parameter = (
        UpdateRunModel._validate_has_updatable_parameter
    )
    model.target_ensemble = "updated_ens%d"

    if should_update:
        model_cls.run_experiment(model, MagicMock(spec=EvaluatorServerConfig))
        model._create_experiment_storage.assert_called_once()
        model.update.assert_called_once()
    else:
        with pytest.raises(ErtRunError, match="No parameters to update"):
            model_cls.run_experiment(model, MagicMock(spec=EvaluatorServerConfig))

        model._create_experiment_storage.assert_not_called()
        model.update.assert_not_called()
