from typing import Optional, Tuple

from ert.config import ErtConfig
from ert.enkf_main import create_run_path, ensemble_context, sample_prior
from ert.libres_facade import LibresFacade
from ert.storage import EnsembleAccessor


def create_runpath(
    storage,
    config,
    active_mask=None,
    *,
    ensemble: Optional[EnsembleAccessor] = None,
    iteration=0,
    random_seed: Optional[int] = 1234,
) -> Tuple[ErtConfig, EnsembleAccessor]:
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
        )

    prior = ensemble_context(
        ensemble,
        active_mask,
        iteration,
        None,
        "",
        ert_config.model_config.runpath_format_string,
        "name",
    )

    sample_prior(
        ensemble,
        [i for i, active in enumerate(active_mask) if active],
        random_seed=random_seed,
    )
    create_run_path(prior, ert_config)
    return ert_config.ensemble_config, ensemble


def load_from_forward_model(ert_config, ensemble):
    facade = LibresFacade.from_config_file(ert_config)
    realizations = [True] * facade.get_ensemble_size()
    return facade.load_from_forward_model(ensemble, realizations, 0)
