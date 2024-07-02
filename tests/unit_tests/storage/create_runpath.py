from typing import Optional, Tuple

from ert.config import ErtConfig
from ert.enkf_main import create_run_path, sample_prior
from ert.libres_facade import LibresFacade
from ert.run_arg import create_run_arguments
from ert.runpaths import Runpaths
from ert.storage import Ensemble


def create_runpath(
    storage,
    config,
    active_mask=None,
    *,
    ensemble: Optional[Ensemble] = None,
    iteration=0,
    random_seed: Optional[int] = 1234,
) -> Tuple[ErtConfig, Ensemble]:
    active_mask = [True] if active_mask is None else active_mask
    ert_config = ErtConfig.from_file(config)

    if ensemble is None:
        experiment_id = storage.create_experiment(
            ert_config.ensemble_config.parameter_configuration
        )
        ensemble = storage.create_ensemble(
            experiment_id,
            name="default",
            ensemble_size=ert_config.model_config.num_realizations,
            iteration=iteration,
        )

    runpaths = Runpaths(
        jobname_format=ert_config.model_config.jobname_format_string,
        runpath_format=ert_config.model_config.runpath_format_string,
        filename=str(ert_config.runpath_file),
        substitution_list=ert_config.substitution_list,
    )
    run_args = create_run_arguments(runpaths, active_mask, ensemble)

    sample_prior(
        ensemble,
        [i for i, active in enumerate(active_mask) if active],
        random_seed=random_seed,
    )
    create_run_path(
        run_args,
        ensemble,
        ert_config,
        runpaths,
    )
    return ert_config.ensemble_config, ensemble


def load_from_forward_model(ert_config, ensemble):
    facade = LibresFacade.from_config_file(ert_config)
    realizations = [True] * facade.get_ensemble_size()
    return facade.load_from_forward_model(ensemble, realizations, 0)
