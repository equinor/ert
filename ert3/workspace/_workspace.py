import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

import ert
import ert3

_WORKSPACE_DATA_ROOT = ".ert"
_EXPERIMENTS_BASE = "experiments"
_RESOURCES_BASE = "resources"


def _locate_root(path: Path) -> Optional[Path]:
    while True:
        if (path / _WORKSPACE_DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


class Workspace:
    """The :py:class:`Workspace` class stores the configuration and input data
    needed to run ERT experiments.

    The workspace object is intended to be created repeatedly from a persistent
    source. In the current implementation the configuration and input data are
    persisted on disk in a workspace directory. The workspace object only stores
    the location of that directory and loads any needed data on demand.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Create a workspace object from a workspace directory path.

        The workspace directory must have been initialized using the
        :py:func:`ert3.workspace.initialize` function.

        Args:
            path (Union[str, pathlib.Path]): The path to the workspace.

        Raises:
            ert.exceptions._exceptions.IllegalWorkspaceOperation:
                Raised when the workspace has not been initialized.
        """
        root = _locate_root(Path(path))
        if root is None:
            raise ert.exceptions.IllegalWorkspaceOperation(
                "Not inside an ERT workspace."
            )
        self._path = root

    @property
    def name(self) -> str:
        """Returns the name of the workspace.

        Returns:
            str: The workspace name
        """
        return str(self._path)

    def assert_experiment_exists(self, experiment_name: str) -> None:
        """Asserts that the given experiment exists.

        Args:
            experiment_name (str): The name of the experiment.

        Raises:
            ert.exceptions._exceptions.IllegalWorkspaceOperation:
                Raised when the experiment does not exist.
        """

        experiment_root = self._path / _EXPERIMENTS_BASE / experiment_name
        if not experiment_root.is_dir():
            raise ert.exceptions.IllegalWorkspaceOperation(
                f"{experiment_name} is not an experiment "
                f"within the workspace {self.name}"
            )

    def get_experiment_names(self) -> Set[str]:
        """Return the names of all experiments in the workspace.

        Raises:
            ert.exceptions._exceptions.IllegalWorkspaceState:
                Raised when the workspace is in an illegal state.

        Returns:
            Set[str]: The names of all experiments.
        """
        experiment_base = self._path / _EXPERIMENTS_BASE
        if not experiment_base.is_dir():
            raise ert.exceptions.IllegalWorkspaceState(
                f"the workspace {self._path} cannot access experiments"
            )
        return {
            experiment.name
            for experiment in experiment_base.iterdir()
            if experiment.is_dir()
        }

    def load_experiment_run_config(
        self,
        experiment_name: str,
        plugin_registry: ert3.config.ConfigPluginRegistry,
    ) -> ert3.config.ExperimentRunConfig:
        """Load the configuration objects needed to run an experiment.

        This method loads, validates and returns a configuration object that
        encapsulates and validates the specialized configuration objects
        that are needed for running an experiment: an experiment configuration
        object, a stages configuration object, an ensemble configuration
        object, and a parameter configuration object.

        Args:
            experiment_name (str): The name of the experiment.

        Returns: ert3.config.ExperimentRunConfig: A configuration object for an
            experiment run.
        """

        experiment_config_path = (
            self._path / _EXPERIMENTS_BASE / experiment_name / "experiment.yml"
        )
        with open(experiment_config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        experiment_config = ert3.config.load_experiment_config(config_dict)

        stages_config_path = self._path / "stages.yml"
        with open(stages_config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        sys.path.append(str(self._path))
        stage_config = ert3.config.load_stages_config(
            config_dict, plugin_registry=plugin_registry
        )

        ensemble_config_path = (
            self._path / _EXPERIMENTS_BASE / experiment_name / "ensemble.yml"
        )
        with open(ensemble_config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        ensemble_config = ert3.config.load_ensemble_config(
            config_dict, plugin_registry=plugin_registry
        )
        self._validate_resources(ensemble_config)

        parameters_config_path = self._path / "parameters.yml"
        with open(parameters_config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        parameters_config = ert3.config.load_parameters_config(config_dict)

        return ert3.config.ExperimentRunConfig(
            experiment_config, stage_config, ensemble_config, parameters_config
        )

    def load_parameters_config(self) -> ert3.config.ParametersConfig:
        """Load the parameters configuration.

        Returns:
            ert3.config.ParametersConfig: A parameters configuration object.
        """
        with open(self._path / "parameters.yml", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return ert3.config.load_parameters_config(config_dict)

    async def load_resource(
        self,
        linked_input: ert3.config.LinkedInput,
        ensemble_size: int = 1,
    ) -> ert.data.RecordCollection:
        transformation = linked_input.source_transformation
        assert (
            transformation
        ), f"cannot load resource: no source transformation for '{linked_input.name}'"
        assert isinstance(transformation, ert.data.FileTransformation), (
            f"cannot load resource for '{linked_input.name}': not a file-based "
            + f"transformation, was {transformation.__class__.__name__}"
        )
        return await ert.data.load_collection_from_file(
            transformation=transformation,
            length=ensemble_size,
            root_path=self._path / _RESOURCES_BASE,
        )

    def export_json(
        self,
        experiment_name: str,
        data: Union[Dict[int, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]],
        output_file: Optional[str] = None,
    ) -> None:
        """Export data generated by an experiment to a JSON file.

        Args:
            experiment_name (str):
                The experiment name.
            data (Union[Dict[int, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]):
                The data to export.
            output_file (Optional[str]):
                The name of the output file. Defaults to :file:`data.json`.
        """
        experiment_root = self._path / _EXPERIMENTS_BASE / experiment_name
        if output_file is None:
            output_file = "data.json"
        self.assert_experiment_exists(experiment_name)
        with open(experiment_root / output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, sort_keys=True)

    def delete(self) -> None:
        shutil.rmtree(self._path / _WORKSPACE_DATA_ROOT)

    def _validate_resources(self, ensemble_config: ert3.config.EnsembleConfig) -> None:
        resource_inputs = [
            item
            for item in ensemble_config.input
            if item.source_namespace == ert3.config.SourceNS.resources
        ]
        for resource in resource_inputs:
            path = self._path / _RESOURCES_BASE / resource.source_location
            if not path.exists():
                raise ert.exceptions.ConfigValidationError(
                    f"Cannot locate resource: '{resource.source_location}'"
                )
            # dynamically created so type must be ignored
            if not resource.transformation:  # type: ignore  # dynamically created
                return
            transformation_instance = resource.get_transformation_instance()  # type: ignore  # noqa
            if (
                isinstance(transformation_instance, ert.data.TarTransformation)
                and not path.is_dir()
            ):
                raise ert.exceptions.ConfigValidationError(
                    f"Resource must be a directory: '{resource.source_location}'"
                )
            if path.is_dir() and not isinstance(
                transformation_instance, ert.data.TarTransformation
            ):
                raise ert.exceptions.ConfigValidationError(
                    f"Resource must be a regular file: '{resource.source_location}'"
                )

    def suggest_local_run_path(
        self, basename: str = "local-test-run", run_id: str = "abcdef"
    ) -> Path:
        """Suggest an absolute directory path that can be used for a local test
        run.

        Args:
            basename: This will be part of returned directory name, with
                an index appended.
            runid: An id that will always be included in the suggested runpath
        """
        return Path(self._path.absolute() / (basename + f"-{run_id}"))


def initialize(path: Union[str, Path]) -> Workspace:
    """Initialize a workspace directory

    Args:
        path (Union[str, pathlib.Path]): Path to the workspace to initialize.

    Raises:
        ert.exceptions.IllegalWorkspaceOperation:
            Raised when the workspace is already initialized.

    Returns:
        Workspace: A corresponding workspace object.
    """
    path = Path(path)
    root = _locate_root(path)
    if root is not None:
        raise ert.exceptions.IllegalWorkspaceOperation(
            f"Already inside an ERT workspace, found {root}/{_WORKSPACE_DATA_ROOT}"
        )
    (path / _WORKSPACE_DATA_ROOT).mkdir()
    return Workspace(path)
