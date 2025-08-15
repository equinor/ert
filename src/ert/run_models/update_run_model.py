import dataclasses
import uuid
from abc import abstractmethod

from ert.analysis._update_commons import ErtAnalysisError
from ert.analysis.event import (
    AnalysisCompleteEvent,
    AnalysisDataEvent,
    AnalysisErrorEvent,
    AnalysisEvent,
    AnalysisStatusEvent,
    AnalysisTimeEvent,
)
from ert.config import (
    DesignMatrix,
    ESSettings,
    GenKwConfig,
    HookRuntime,
    ObservationSettings,
    ParameterConfig,
)
from ert.plugins.workflow_fixtures import (
    PostUpdateFixtures,
    PreFirstUpdateFixtures,
    PreUpdateFixtures,
)
from ert.run_models.event import (
    RunModelDataEvent,
    RunModelErrorEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
)
from ert.run_models.run_model import ErtRunError, RunModel
from ert.storage import Ensemble


class UpdateRunModel(RunModel):
    target_ensemble: str
    analysis_settings: ESSettings
    update_settings: ObservationSettings

    @abstractmethod
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
            observation_settings=self.update_settings,
            es_settings=self.analysis_settings,
            random_seed=self.random_seed,
            reports_dir=self.reports_dir(experiment_name=prior.experiment.name),
            run_paths=self._run_paths,
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
        self, iteration: int, run_id: uuid.UUID, event: AnalysisEvent
    ) -> None:
        match event:
            case AnalysisStatusEvent(msg=msg):
                self.send_event(
                    RunModelStatusEvent(iteration=iteration, run_id=run_id, msg=msg)
                )
            case AnalysisTimeEvent():
                self.send_event(
                    RunModelTimeEvent(
                        iteration=iteration,
                        run_id=run_id,
                        elapsed_time=event.elapsed_time,
                        remaining_time=event.remaining_time,
                    )
                )
            case AnalysisErrorEvent():
                self.send_event(
                    RunModelErrorEvent(
                        iteration=iteration,
                        run_id=run_id,
                        error_msg=event.error_msg,
                        data=event.data,
                    )
                )
            case AnalysisDataEvent(name=name, data=data):
                self.send_event(
                    RunModelDataEvent(
                        iteration=iteration, run_id=run_id, name=name, data=data
                    )
                )
            case AnalysisCompleteEvent():
                self.send_event(
                    RunModelUpdateEndEvent(
                        iteration=iteration, run_id=run_id, data=event.data
                    )
                )

    @classmethod
    def _merge_parameters_from_design_matrix(
        cls,
        parameters_config: list[ParameterConfig],
        design_matrix: DesignMatrix | None,
        rerun_failed_realizations: bool,
    ) -> tuple[list[ParameterConfig], DesignMatrix | None, GenKwConfig | None]:
        parameters_config, design_matrix, design_matrix_group = (
            super()._merge_parameters_from_design_matrix(
                parameters_config, design_matrix, rerun_failed_realizations
            )
        )

        if design_matrix and not any(p.update for p in parameters_config):
            raise ErtRunError(
                "No parameters to update as all parameters were set to update:false!"
            )

        return parameters_config, design_matrix, design_matrix_group
