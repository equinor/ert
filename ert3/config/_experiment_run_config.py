from typing import Dict, NamedTuple, cast

import ert

from . import (
    EnsembleConfig,
    ExperimentConfig,
    ParametersConfig,
    SourceNS,
    StagesConfig,
    Step,
)


class LinkedInput(NamedTuple):
    """The LinkedInput class provides all necessary information to use the
    inputs that are needed by an experiment.

    The LinkedInput class collects the needed configuration from the experiment,
    stages and ensemble configuration objects that are part of an
    `:py:class:ert3.config.ExperimentRunConfig` configuration object.
    """

    name: str
    source_mime: str
    source_namespace: SourceNS
    source_location: str
    source_is_directory: bool
    dest_location: str
    dest_mime: str
    dest_is_directory: bool


class ExperimentRunConfig:
    """The :py:class:`ExperimentRunConfig` class encapsulates the configuration
    objects needed to run an experiment.

    It encapsulates the specialized configuration objects: an experiment
    configuration object, a stages configuration object, an ensemble
    configuration object, and a parameter configuration object. These objects
    must already have been constructed and validated when initializing the
    :py:class:`ExperimentRunConfig` object. These configuration objects are then
    cross-validated to ensure that they are valid and consistent for configuring
    and running an experiment.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        stages_config: StagesConfig,
        ensemble_config: EnsembleConfig,
        parameters_config: ParametersConfig,
    ) -> None:
        """Create and cross-validates an configuration object for running an
        experiment.

        Args:
            experiment_config (ExperimentConfig): Experiment configuration object.
            stages_config (StagesConfig): Stages configuration object.
            ensemble_config (EnsembleConfig): Ensemble configuration object.
            parameters_config (ParametersConfig): Paramters configuration object.
        """
        self._experiment_config = experiment_config
        self._stages_config = stages_config
        self._ensemble_config = ensemble_config
        self._parameters_config = parameters_config

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

    @property
    def parameters_config(self) -> ParametersConfig:
        """Access the parameters configuration object.

        Returns:
            ParametersConfig: The encapsulated parameters configuration object.
        """
        return self._parameters_config

    def get_stage(self) -> Step:
        """Return the stage used by the forward model in this
        experiment.

        Returns:
            str: The name of the stage.
        """
        stage = self._stages_config.step_from_key(
            self._ensemble_config.forward_model.stage
        )
        # The stage has already been validated to exist:
        return cast(Step, stage)

    def get_linked_inputs(self) -> Dict[SourceNS, Dict[str, LinkedInput]]:
        """Return the linked inputs needed for this experiment run.

        Returns:
            Dict[SourceNS, Dict[str, LinkedInput]]: A dictionary of linked inputs.
        """
        inputs: Dict[SourceNS, Dict[str, LinkedInput]] = {
            SourceNS.storage: {},
            SourceNS.resources: {},
            SourceNS.stochastic: {},
        }
        stage = self.get_stage()
        for ensemble_input in self._ensemble_config.input:
            name = ensemble_input.record
            stage_is_directory = stage.input[name].is_directory
            stage_mime = stage.input[name].mime
            stage_location = stage.input[name].location

            if stage_mime != ensemble_input.mime:
                print(
                    f"Warning: Conflicting ensemble mime '{ensemble_input.mime}' and "
                    + f"stage mime '{stage_mime}' for input '{name}'."
                )

            # fall back on stage is_directory
            ensemble_is_directory = (
                ensemble_input.is_directory
                if ensemble_input.is_directory is not None
                else stage_is_directory
            )

            input_ = LinkedInput(
                name=name,
                source_namespace=ensemble_input.source_namespace,
                source_mime=ensemble_input.mime,
                source_is_directory=ensemble_is_directory,
                source_location=ensemble_input.source_location,
                dest_mime=stage_mime,
                dest_location=stage_location,
                dest_is_directory=stage_is_directory,
            )
            inputs[input_.source_namespace][input_.name] = input_
        return inputs

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
        stage = self._stages_config.step_from_key(stage_name)
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
