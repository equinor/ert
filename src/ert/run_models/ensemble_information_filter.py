from __future__ import annotations

import dataclasses
import functools
import logging
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING

from ert.analysis import ErtAnalysisError, enif_update
from ert.config import ErtConfig, ESSettings, HookRuntime, ObservationSettings
from ert.plugins import (
    PostUpdateFixtures,
    PreFirstUpdateFixtures,
    PreUpdateFixtures,
)
from ert.storage import Ensemble, Storage

from .base_run_model import ErtRunError, StatusEvents
from .event import RunModelStatusEvent, RunModelUpdateBeginEvent
from .update_run_model import UpdateRunModel

if TYPE_CHECKING:
    from ert.config import QueueConfig


logger = logging.getLogger(__name__)


class EnsembleInformationFilter(UpdateRunModel):
    def __init__(
        self,
        target_ensemble: str,
        experiment_name: str,
        active_realizations: list[bool],
        minimum_required_realizations: int,
        random_seed: int,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        es_settings: ESSettings,
        update_settings: ObservationSettings,
        status_queue: SimpleQueue[StatusEvents],
    ):
        super().__init__(
            es_settings,
            update_settings,
            storage,
            config.runpath_file,
            Path(config.user_config_file),
            config.env_vars,
            config.env_pr_fm_step,
            config.runpath_config,
            queue_config,
            config.forward_model_steps,
            status_queue,
            config.substitutions,
            config.hooked_workflows,
            active_realizations=active_realizations,
            start_iteration=0,
            total_iterations=2,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
            log_path=config.analysis_config.log_path,
            config=config,
            target_ensemble=target_ensemble,
            experiment_name=experiment_name,
        )

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
            enif_update(
                prior,
                posterior,
                parameters=prior.experiment.update_parameters,
                observations=prior.experiment.observation_keys,
                global_scaling=weight,
                random_seed=self.random_seed,
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

    @classmethod
    def name(cls) -> str:
        return "Ensemble Information Filter (Experimental)"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → EnIF update → evaluate"
