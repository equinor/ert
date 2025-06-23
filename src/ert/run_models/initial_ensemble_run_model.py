from abc import ABC
from typing import Any

import numpy as np
import polars as pl
from pydantic import PrivateAttr

from ert.config.design_matrix import DesignMatrix
from ert.config.parameter_config import ParameterConfig
from ert.config.response_config import ResponseConfig
from ert.enkf_main import sample_prior
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.run_arg import create_run_arguments
from ert.run_models.base_run_model import BaseRunModel
from ert.storage.local_ensemble import LocalEnsemble


class InitialEnsembleRunModel(BaseRunModel, ABC):
    experiment_name: str
    design_matrix: DesignMatrix | None
    parameter_configuration: list[ParameterConfig]
    response_configuration: list[ResponseConfig]
    ert_templates: list[tuple[str, str]]
    _observations: dict[str, pl.DataFrame] | None = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        observations = data.pop("observations", None)
        super().__init__(**data)
        self._observations = observations

    def _sample_prior_and_evaluate_ensemble(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        simulation_arguments: dict[str, str] | None,
        ensemble_name: str,
        rerun_failed_realizations: bool = False,
        ensemble_storage: LocalEnsemble | None = None,
    ) -> LocalEnsemble:
        parameters_config, design_matrix = self._merge_parameters_from_design_matrix(
            self.parameter_configuration,
            self.design_matrix,
            rerun_failed_realizations,
        )
        if ensemble_storage is None:
            experiment_storage = self._storage.create_experiment(
                parameters=parameters_config,
                observations=self._observations,
                responses=self.response_configuration,
                simulation_arguments=simulation_arguments,
                name=self.experiment_name,
                templates=self.ert_templates,
            )
            ensemble_storage = self._storage.create_ensemble(
                experiment_storage,
                ensemble_size=self.ensemble_size,
                name=ensemble_name,
            )
        else:
            assert ensemble_storage is not None
        self.set_env_key("_ERT_EXPERIMENT_ID", str(ensemble_storage.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble_storage.id))

        sample_prior(
            ensemble_storage,
            np.where(self.active_realizations)[0],
            parameters=[param.name for param in parameters_config],
            random_seed=self.random_seed,
            design_matrix_df=(
                design_matrix.design_matrix_df if design_matrix is not None else None
            ),
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
