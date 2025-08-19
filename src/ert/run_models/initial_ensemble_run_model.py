import numpy as np

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
        ensemble_storage: LocalEnsemble,
    ) -> LocalEnsemble:
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
