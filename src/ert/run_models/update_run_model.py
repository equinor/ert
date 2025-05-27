import dataclasses
import functools
from collections import defaultdict
from pathlib import Path
from queue import SimpleQueue
from typing import Any

import numpy as np

from ert.analysis import ErtAnalysisError, smoother_update
from ert.config import (
    ConfigValidationError,
    ErtConfig,
    ESSettings,
    ForwardModelStep,
    HookRuntime,
    ModelConfig,
    ObservationSettings,
    QueueConfig,
    Workflow,
)
from ert.enkf_main import sample_prior, save_design_matrix_to_ensemble
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.plugins import (
    PostExperimentFixtures,
    PostUpdateFixtures,
    PreExperimentFixtures,
    PreFirstUpdateFixtures,
    PreUpdateFixtures,
)
from ert.run_arg import create_run_arguments
from ert.storage import Ensemble, Storage
from ert.storage.local_ensemble import LocalEnsemble
from ert.substitutions import Substitutions
from ert.trace import tracer

from .base_run_model import BaseRunModel, ErtRunError
from .event import RunModelStatusEvent, RunModelUpdateBeginEvent, StatusEvents


class UpdateRunModel(BaseRunModel):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        analysis_settings: ESSettings,
        update_settings: ObservationSettings,
        storage: Storage,
        runpath_file: Path,
        user_config_file: Path,
        env_vars: dict[str, str],
        env_pr_fm_step: dict[str, dict[str, Any]],
        model_config: ModelConfig,
        queue_config: QueueConfig,
        forward_model_steps: list[ForwardModelStep],
        status_queue: SimpleQueue[StatusEvents],
        substitutions: Substitutions,
        hooked_workflows: defaultdict[HookRuntime, list[Workflow]],
        active_realizations: list[bool],
        total_iterations: int,
        start_iteration: int,
        random_seed: int,
        minimum_required_realizations: int,
        log_path: Path,
        config: ErtConfig,
        target_ensemble: str,
        experiment_name: str | None,
    ):
        super().__init__(
            storage,
            runpath_file,
            user_config_file,
            env_vars,
            env_pr_fm_step,
            model_config,
            queue_config,
            forward_model_steps,
            status_queue,
            substitutions,
            hooked_workflows,
            active_realizations=active_realizations,
            total_iterations=total_iterations,
            start_iteration=start_iteration,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
            log_path=log_path,
        )
        self._analysis_settings: ESSettings = analysis_settings
        self._update_settings: ObservationSettings = update_settings
        self._parameter_configuration = config.ensemble_config.parameter_configuration
        self._response_configuration = config.ensemble_config.response_configuration
        self._design_matrix = config.analysis_config.design_matrix
        self._observations = config.observations
        self._templates = config.ert_templates

        self.experiment_name = experiment_name
        self.target_ensemble_format = target_ensemble
        self.simulation_arguments: dict[str, str] | None = None

    def update(
        self,
        prior: Ensemble,
        posterior_name: str,
        weight: float = 1.0,
    ) -> Ensemble:
        self.validate_successful_realizations_count()
        self.send_event(
            RunModelUpdateBeginEvent(iteration=prior.iteration, run_id=prior.id)
        )
        self.send_event(
            RunModelStatusEvent(
                iteration=prior.iteration,
                run_id=prior.id,
                msg="Creating posterior ensemble..",
            )
        )

        pre_first_update_fixtures = PreFirstUpdateFixtures(
            storage=self._storage,
            ensemble=prior,
            observation_settings=self._update_settings,
            es_settings=self._analysis_settings,
            random_seed=self.random_seed,
            reports_dir=self.reports_dir(experiment_name=prior.experiment.name),
            run_paths=self.run_paths,
        )

        posterior = self._storage.create_ensemble(
            prior.experiment,
            ensemble_size=prior.ensemble_size,
            iteration=prior.iteration + 1,
            name=posterior_name,
            prior_ensemble=prior,
        )
        if prior.iteration == 0:
            self.run_workflows(
                fixtures=pre_first_update_fixtures,
            )

        update_args_dict = {
            field.name: getattr(pre_first_update_fixtures, field.name)
            for field in dataclasses.fields(pre_first_update_fixtures)
        }

        self.run_workflows(
            fixtures=PreUpdateFixtures(
                **{**update_args_dict, "hook": HookRuntime.PRE_UPDATE}
            ),
        )
        try:
            smoother_update(
                prior,
                posterior,
                update_settings=self._update_settings,
                es_settings=self._analysis_settings,
                parameters=prior.experiment.update_parameters,
                observations=prior.experiment.observation_keys,
                global_scaling=weight,
                rng=self.rng,
                progress_callback=functools.partial(
                    self.send_smoother_event,
                    prior.iteration,
                    prior.id,
                ),
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                "Update algorithm failed for iteration:"
                f"{posterior.iteration}. The following error occurred: {e}"
            ) from e

        self.run_workflows(
            fixtures=PostUpdateFixtures(
                **{**update_args_dict, "hook": HookRuntime.POST_UPDATE}
            ),
        )
        return posterior

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        if restart and not self.support_restart:
            raise ErtRunError(f"The {self.name()} does not support restarting")
        self.restart = restart

        parameters_config, design_matrix, design_matrix_group = (
            self._merge_parameters_from_design_matrix()
        )

        self._preExperimentFixtures()

        prior = self._evaluate_prior(
            design_matrix,
            design_matrix_group,
            evaluator_server_config,
            parameters_config,
        )

        posterior = self._update_then_run_ensembles(evaluator_server_config, prior)

        self.run_workflows(
            fixtures=PostExperimentFixtures(
                random_seed=self.random_seed,
                storage=self._storage,
                ensemble=posterior,
            ),
        )

    def _preExperimentFixtures(self):
        self.run_workflows(fixtures=PreExperimentFixtures(random_seed=self.random_seed))

    def _update_then_run_ensembles(self, evaluator_server_config, prior):
        posterior = self.update(prior, self.target_ensemble_format % 1)
        posterior_args = create_run_arguments(
            self.run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=posterior,
        )
        self._evaluate_and_postprocess(
            posterior_args,
            posterior,
            evaluator_server_config,
        )
        return posterior

    def _evaluate_prior(
        self,
        design_matrix,
        design_matrix_group,
        evaluator_server_config,
        parameters_config,
    ) -> LocalEnsemble:
        prior = self._initialize_prior_ensemble(
            parameters_config + ([design_matrix_group] if design_matrix_group else [])
        )
        prior_args = create_run_arguments(
            self.run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=prior,
        )
        sample_prior(
            prior,
            np.where(self.active_realizations)[0],
            parameters=[param.name for param in parameters_config],
            random_seed=self.random_seed,
        )
        if design_matrix_group is not None and design_matrix is not None:
            save_design_matrix_to_ensemble(
                design_matrix.design_matrix_df,
                prior,
                np.where(self.active_realizations)[0],
                design_matrix_group.name,
            )
        self._evaluate_and_postprocess(
            prior_args,
            prior,
            evaluator_server_config,
        )
        return prior

    def _initialize_prior_ensemble(self, parameters_config):
        experiment = self._storage.create_experiment(
            parameters=parameters_config,
            observations=self._observations,
            responses=self._response_configuration,
            simulation_arguments=self.simulation_arguments,
            name=self.experiment_name,
            templates=self._templates,
        )
        ensemble = self._storage.create_ensemble(
            experiment,
            ensemble_size=self.ensemble_size,
            name=self.target_ensemble_format % 0,
        )
        self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble.id))
        return ensemble

    def _merge_parameters_from_design_matrix(self):
        parameters_config = self._parameter_configuration
        design_matrix = self._design_matrix
        design_matrix_group = None
        # If a design matrix is present, we try to merge design matrix parameters
        # to the experiment parameters and set new active realizations
        if design_matrix is not None and not self.restart:
            try:
                parameters_config, design_matrix_group = (
                    design_matrix.merge_with_existing_parameters(parameters_config)
                )
                if issubclass(type(self), UpdateRunModel) and not any(
                    p.update for p in parameters_config
                ):
                    raise ConfigValidationError(
                        "No parameters to update as all "
                        "parameters were set to update:false!",
                    )
            except ConfigValidationError as exc:
                raise ErtRunError(str(exc)) from exc
        return parameters_config, design_matrix, design_matrix_group
