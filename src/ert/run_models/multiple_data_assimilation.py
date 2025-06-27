from __future__ import annotations

import functools
import logging
from typing import Any, ClassVar
from uuid import UUID

from pydantic import PrivateAttr

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.initial_ensemble_run_model import InitialEnsembleRunModel
from ert.run_models.update_run_model import UpdateRunModel
from ert.storage import Ensemble
from ert.trace import tracer

from ..analysis import smoother_update
from ..plugins import PostExperimentFixtures, PreExperimentFixtures
from ..run_arg import create_run_arguments
from .run_model import ErtRunError

logger = logging.getLogger(__name__)

MULTIPLE_DATA_ASSIMILATION_GROUP = "Parameter update"


class MultipleDataAssimilation(UpdateRunModel, InitialEnsembleRunModel):
    """
    Run multiple data assimilation (MDA) ensemble smoother with custom weights.
    """

    default_weights: ClassVar[str] = "4, 2, 1"
    restart_run: bool
    prior_ensemble_id: str | None
    weights: str

    _parsed_weights: list[float] = PrivateAttr()
    _total_iterations: int = PrivateAttr(default=2)

    def model_post_init(self, ctx: Any) -> None:
        super().model_post_init(ctx)
        self._parsed_weights = self.parse_weights(self.weights)
        start_iteration = 0
        total_iterations = len(self._parsed_weights) + 1
        if self.restart_run:
            if not self.prior_ensemble_id:
                raise ValueError("For restart run, prior ensemble must be set")
            start_iteration = (
                self._storage.get_ensemble(self.prior_ensemble_id).iteration + 1
            )
            total_iterations -= start_iteration
        elif not self.experiment_name:
            raise ValueError("For non-restart run, experiment name must be set")

        self.start_iteration = start_iteration
        self._total_iterations = total_iterations

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        self.log_at_startup()
        if rerun_failed_realizations:
            raise ErtRunError("ESMDA does not support restart")

        if self.restart_run:
            id_ = self.prior_ensemble_id
            assert id_ is not None
            try:
                ensemble_id = UUID(id_)
                prior = self._storage.get_ensemble(ensemble_id)
                experiment = prior.experiment
                self.set_env_key("_ERT_EXPERIMENT_ID", str(experiment.id))
                self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
                assert isinstance(prior, Ensemble)
                if self.start_iteration != prior.iteration + 1:
                    raise ValueError(
                        "Experiment misconfigured, got starting "
                        f"iteration: {self.start_iteration},"
                        f"restart iteration = {prior.iteration + 1}"
                    )
            except (KeyError, ValueError) as err:
                raise ErtRunError(
                    f"Prior ensemble with ID: {id_} does not exists"
                ) from err
        else:
            self.run_workflows(
                fixtures=PreExperimentFixtures(random_seed=self.random_seed),
            )
            sim_args = {"weights": self.weights}
            prior = self._sample_and_evaluate_ensemble(
                evaluator_server_config,
                sim_args,
                self.target_ensemble % 0,
            )

        enumerated_weights: list[tuple[int, float]] = list(
            enumerate(self._parsed_weights)
        )
        weights_to_run = enumerated_weights[prior.iteration :]

        for iteration, weight in weights_to_run:
            posterior = self.update(
                prior,
                self.target_ensemble % (iteration + 1),
                weight=weight,
            )
            posterior_args = create_run_arguments(
                self._run_paths,
                self.active_realizations,
                ensemble=posterior,
            )
            self._evaluate_and_postprocess(
                posterior_args,
                posterior,
                evaluator_server_config,
            )
            prior = posterior

        self.run_workflows(
            fixtures=PostExperimentFixtures(
                random_seed=self.random_seed,
                storage=self._storage,
                ensemble=prior,
            ),
        )

    def update_ensemble_parameters(
        self, prior: Ensemble, posterior: Ensemble, weight: float
    ) -> None:
        smoother_update(
            prior,
            posterior,
            update_settings=self.update_settings,
            es_settings=self.analysis_settings,
            parameters=prior.experiment.update_parameters,
            observations=prior.experiment.observation_keys,
            global_scaling=weight,
            rng=self._rng,
            progress_callback=functools.partial(
                self.send_smoother_event,
                prior.iteration,
                prior.id,
            ),
        )

    @staticmethod
    def parse_weights(weights: str) -> list[float]:
        """Parse weights string and scale weights such that their reciprocals sum
        to 1.0, i.e., sum(1.0 / x for x in weights) == 1.0. See for example Equation
        38 of evensen2018 - Analysis of iterative ensemble
        smoothers for solving inverse problems.
        """
        if not weights:
            raise ValueError(f"Must provide weights, got {weights}")

        elements = weights.split(",")
        elements = [element.strip() for element in elements if element.strip()]

        result: list[float] = []
        for element in elements:
            try:
                f = float(element)
                if f == 0:
                    logger.info("Warning: 0 weight, will ignore")
                else:
                    result.append(f)
            except ValueError as e:
                raise ValueError(f"Warning: cannot parse weight {element}") from e
        if not result:
            raise ValueError(f"Invalid weights: {weights}")

        length = sum(1.0 / x for x in result)
        return [x * length for x in result]

    @classmethod
    def name(cls) -> str:
        return "Multiple data assimilation"

    @classmethod
    def display_name(cls) -> str:
        return cls.name() + " - Recommended algorithm"

    @classmethod
    def description(cls) -> str:
        return "[Sample|restart] → [evaluate → update] for each weight."

    @classmethod
    def group(cls) -> str | None:
        return MULTIPLE_DATA_ASSIMILATION_GROUP
