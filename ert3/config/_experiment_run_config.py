import ert
from . import ExperimentConfig, StagesConfig, EnsembleConfig


class ExperimentRunConfig:
    """The :py:class:`ExperimentRunConfig` class encapsulates the configuration
    objects needed to run an experiment.

    It encapsulates three specialized configuration objects: an experiment
    configuration object, a stages configuration object, and an ensemble
    configuration object. These objects must already have been constructed
    and validated when initializing the :py:class:`ExperimentRunConfig`
    object. These three configuration objects are then cross-validated to
    ensure that they are valid and consistent for configuring and running an
    experiment.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        stages_config: StagesConfig,
        ensemble_config: EnsembleConfig,
    ) -> None:
        """Create and cross-validates an configuration object for running an
        experiment.

        Args:
            experiment_config (ExperimentConfig): Experiment configuration object.
            stages_config (StagesConfig): Stages configuration object.
            ensemble_config (EnsembleConfig): Ensemble configuration object.
        """
        self._experiment_config = experiment_config
        self._stages_config = stages_config
        self._ensemble_config = ensemble_config

        self._validate_ensemble_size()
        self._validate_stage()

    @property
    def experiment_config(self) -> ExperimentConfig:
        """Access the experiment configuration object.

        Returns:
            ExperimentConfig: The encapsulated experiment configuration object.
        """
        return self._experiment_config

    @property
    def stages_config(self) -> StagesConfig:
        """Access the stages configuration object.

        Returns:
            StagesConfig: The encapsulated stages configuration object.
        """
        return self._stages_config

    @property
    def ensemble_config(self) -> EnsembleConfig:
        """Access the ensemble configuration object.

        Returns:
            EnsembleConfig: The encapsulated ensemble configuration object.
        """
        return self._ensemble_config

    def _validate_ensemble_size(self) -> None:
        if (
            self._experiment_config.type == "sensitivity"
            and self._ensemble_config.size is not None
        ):
            raise ert.exceptions.ConfigValidationError(
                "No ensemble size should be specified for a sensitivity analysis."
            )
        if (
            self._experiment_config.type != "sensitivity"
            and self._ensemble_config.size is None
        ):
            raise ert.exceptions.ConfigValidationError(
                "An ensemble size must be specified."
            )

    def _validate_stage(self) -> None:
        stage_name = self._ensemble_config.forward_model.stage
        stage = next(
            (stage for stage in self._stages_config if stage.name == stage_name), None
        )
        if stage is None:
            raise ert.exceptions.ConfigValidationError(
                f"Invalid stage in forward model: '{stage_name}'. "
                f"Must be one of: "
                + ", ".join(f"'{stage.name}'" for stage in self._stages_config)
            )
        stage_input_names = set(stage.input.keys())
        ensemble_input_names = set(
            input.record for input in self._ensemble_config.input
        )
        if ensemble_input_names != stage_input_names:
            raise ert.exceptions.ConfigValidationError(
                "Ensemble and stage inputs do not match."
            )
