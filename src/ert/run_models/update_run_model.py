import dataclasses
import functools
import logging

from ert.analysis import build_strategy_map, smoother_update
from ert.analysis._update_commons import ErtAnalysisError
from ert.analysis.event import (
    AnalysisCompleteEvent,
    AnalysisDataEvent,
    AnalysisErrorEvent,
    AnalysisEvent,
    AnalysisMatrixEvent,
    AnalysisRhoMatrixEvent,
    AnalysisScalingEvent,
    AnalysisStatusEvent,
    AnalysisTimeEvent,
)
from ert.config import (
    ConfigValidationError,
    HookRuntime,
    PostUpdateFixtures,
    PreFirstUpdateFixtures,
    PreUpdateFixtures,
)
from ert.config.parsing.validators import validate_has_updatable_parameter
from ert.run_models.event import (
    RunModelDataEvent,
    RunModelErrorEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
)
from ert.run_models.run_model import ErtRunError, RunModel
from ert.run_models.run_model_configs import UpdateRunModelConfig
from ert.storage import Ensemble, LocalExperiment

logger = logging.getLogger(__name__)


class UpdateRunModel(RunModel, UpdateRunModelConfig):
    def update_ensemble_parameters(
        self, prior: Ensemble, posterior: Ensemble, weight: float
    ) -> None:
        """
        Updates parameters of prior ensemble assumed to already contain responses.
        Writes resulting updated parameters into the posterior ensemble.

        Parameters
        ----------
        prior : Ensemble
            The prior ensemble, which must contain responses
            for the observations of the experiment.

        posterior : Ensemble
            The (initially empty) posterior ensemble
            where the updated parameters will be stored.

        weight : float
            The weight applied to this update step (only used in esmda).
        """
        progress_callback = functools.partial(
            self.send_smoother_event,
            posterior,
        )

        strategy_map = build_strategy_map(
            parameters=prior.experiment.update_parameters,
            param_configs=prior.experiment.parameter_configuration,
            enkf_truncation=self.analysis_settings.enkf_truncation,
            correlation_threshold=self.analysis_settings.correlation_threshold,
            progress_callback=progress_callback,
            experiment=prior.experiment,
        )

        smoother_update(
            prior,
            posterior,
            update_settings=self.update_settings,
            rng=self._rng,
            strategy_map=strategy_map,
            observations=prior.experiment.observation_keys,
            global_scaling=weight,
            progress_callback=progress_callback,
            active_realizations=self.active_realizations,
        )

    def update(
        self,
        prior: Ensemble,
        posterior_name: str,
        weight: float = 1.0,
        target_experiment: LocalExperiment | None = None,
    ) -> Ensemble:
        self._validate_has_updatable_parameter(prior)
        self.validate_successful_realizations_count()

        pre_first_update_fixtures = PreFirstUpdateFixtures(
            storage=self._storage,
            ensemble=prior,
            observation_settings=self.update_settings,
            es_settings=self.analysis_settings,
            random_seed=self.random_seed,
            reports_dir=self.reports_dir(experiment_name=prior.experiment.name),
            run_paths=self._runpaths,
        )

        posterior = self._storage.create_ensemble(
            target_experiment or prior.experiment,
            ensemble_size=prior.ensemble_size,
            iteration=prior.iteration + 1,
            name=posterior_name,
            prior_ensemble=prior,
        )

        self.send_event(
            RunModelUpdateBeginEvent(
                iteration=posterior.iteration,
                ensemble_id=posterior.id,
                prior_ensemble_id=prior.id,
            )
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
            self.update_ensemble_parameters(prior, posterior, weight)
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

    def send_smoother_event(
        self,
        posterior: Ensemble,
        event: AnalysisEvent,
    ) -> None:
        match event:
            case AnalysisStatusEvent():
                self.send_event(
                    RunModelStatusEvent(
                        iteration=posterior.iteration,
                        ensemble_id=posterior.id,
                        msg=event.msg,
                        detail=event.detail,
                    )
                )
            case AnalysisTimeEvent():
                self.send_event(
                    RunModelTimeEvent(
                        iteration=posterior.iteration,
                        ensemble_id=posterior.id,
                        elapsed_time=event.elapsed_time,
                        remaining_time=event.remaining_time,
                    )
                )
            case AnalysisErrorEvent():
                try:
                    posterior.save_blob(event)
                except Exception:
                    logger.exception(
                        "Could not store the report of the failed update of %s",
                        posterior.name,
                    )
                self.send_event(
                    RunModelErrorEvent(
                        iteration=posterior.iteration,
                        ensemble_id=posterior.id,
                        error_msg=event.error_msg,
                        data=event.data,
                    )
                )
            case AnalysisDataEvent():
                posterior.save_blob(event)
                self.send_event(
                    RunModelDataEvent(
                        iteration=posterior.iteration,
                        ensemble_id=posterior.id,
                        name=event.name,
                        data=event.data,
                    )
                )
            case AnalysisMatrixEvent():
                posterior.save_blob(event)
            case AnalysisScalingEvent():
                posterior.save_blob(event)
            case AnalysisRhoMatrixEvent():
                posterior.experiment.save_blob(event)
            case AnalysisCompleteEvent():
                posterior.save_blob(event)
                self.send_event(
                    RunModelUpdateEndEvent(
                        iteration=posterior.iteration,
                        ensemble_id=posterior.id,
                        data=event.data,
                    )
                )

    @staticmethod
    def _validate_has_updatable_parameter(prior: Ensemble) -> None:
        try:
            validate_has_updatable_parameter(
                list(prior.experiment.parameter_configuration.values())
            )
        except ConfigValidationError as err:
            raise ErtRunError(
                f"Cannot update prior ensemble '{prior.name}' (ID: {prior.id}): {err}"
            ) from err
