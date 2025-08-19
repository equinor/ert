from typing import cast

import numpy as np

from ert.config import (
    ParameterConfig,
    ResponseConfig,
)
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_arg import create_run_arguments
from ert.run_models.ert_runmodel_configs import (
    InitialEnsembleRunModelConfig,
)
from ert.run_models.run_model import RunModel
from ert.sample_prior import sample_prior
from ert.storage.local_ensemble import LocalEnsemble


class InitialEnsembleRunModel(RunModel, InitialEnsembleRunModelConfig):
    def _sample_and_evaluate_ensemble(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        simulation_arguments: dict[str, str] | None,
        ensemble_name: str,
        ensemble_storage: LocalEnsemble | None = None,
    ) -> LocalEnsemble:
        if ensemble_storage is None:
            experiment_storage = self._storage.create_experiment(
                parameters=cast(list[ParameterConfig], self.parameter_configuration),
                observations={k: v.to_polars() for k, v in self.observations.items()}
                if self.observations is not None
                else None,
                responses=cast(list[ResponseConfig], self.response_configuration),
                simulation_arguments=simulation_arguments,
                name=self.experiment_name,
                templates=self.ert_templates,
            )

            experiment_storage.save_experiment_config(
                serialized_experiment=self.model_dump()
            )

            ensemble_storage = self._storage.create_ensemble(
                experiment_storage,
                ensemble_size=self.ensemble_size,
                name=ensemble_name,
            )
            if hasattr(self, "_ensemble_id"):
                setattr(self, "_ensemble_id", ensemble_storage.id)  # noqa: B010

        assert ensemble_storage is not None
        self.set_env_key("_ERT_EXPERIMENT_ID", str(ensemble_storage.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble_storage.id))

        sample_prior(
            ensemble_storage,
            np.where(self.active_realizations)[0],
            parameters=[param.name for param in self.parameter_configuration],
            random_seed=self.random_seed,
            design_matrix_df=self.design_matrix.to_polars()
            if self.design_matrix is not None
            else None,
        )

        prior_args = create_run_arguments(
            self._run_paths,
            np.array(self.active_realizations, dtype=bool),
            ensemble=ensemble_storage,
        )
        self._evaluate_and_postprocess(
            prior_args,
            ensemble_storage,
            evaluator_server_config,
        )
        return ensemble_storage
