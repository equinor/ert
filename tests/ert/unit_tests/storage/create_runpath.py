from ert.config import ErtConfig
from ert.run_arg import create_run_arguments
from ert.run_models._create_run_path import create_run_path
from ert.runpaths import Runpaths
from ert.sample_prior import sample_prior
from ert.storage import Ensemble


def create_runpath(
    storage,
    config,
    active_mask=None,
    *,
    ensemble: Ensemble | None = None,
    iteration=0,
) -> tuple[ErtConfig, Ensemble]:
    active_mask = [True] if active_mask is None else active_mask
    ert_config = ErtConfig.from_file(config)

    if ensemble is None:
        experiment_id = storage.create_experiment(
            ert_config.ensemble_config.parameter_configuration,
            templates=ert_config.ert_templates,
        )
        ensemble = storage.create_ensemble(
            experiment_id,
            name="default",
            ensemble_size=ert_config.runpath_config.num_realizations,
            iteration=iteration,
        )

    runpaths = Runpaths.from_config(ert_config)
    run_args = create_run_arguments(
        runpaths.from_config(ert_config), active_mask, ensemble
    )

    sample_prior(
        ensemble,
        [i for i, active in enumerate(active_mask) if active],
        random_seed=ert_config.random_seed,
    )
    create_run_path(
        run_args=run_args,
        ensemble=ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        parameters_file="parameters",
        runpaths=runpaths,
    )
    return ert_config.ensemble_config, ensemble
