from __future__ import annotations

import collections
import logging
import warnings
from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import pluggy
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ert.config import (
    ForwardModelStep,
    ForwardModelStepDocumentation,
    ForwardModelStepPlugin,
    KnownQueueOptions,
    LegacyWorkflowConfigs,
    LocalQueueOptions,
    WorkflowConfigs,
    WorkflowJob,
    forward_model_step_from_config_contents,
    workflow_job_from_file,
)
from ert.trace import add_span_processor

logger = logging.getLogger(__name__)

_PLUGIN_NAMESPACE = "ert"

hook_implementation = pluggy.HookimplMarker(_PLUGIN_NAMESPACE)
hook_specification = pluggy.HookspecMarker(_PLUGIN_NAMESPACE)


if TYPE_CHECKING:
    from .plugin_response import PluginMetadata, PluginResponse
K = TypeVar("K")
V = TypeVar("V")


class JobDoc(TypedDict):
    description: str
    examples: str | None
    config_file: str | None
    parser: Callable[[], ArgumentParser] | None
    source_package: str
    category: str


class ErtPluginManager(pluggy.PluginManager):
    def __init__(self, plugins: Sequence[object] | None = None) -> None:
        super().__init__(_PLUGIN_NAMESPACE)

        import ert.plugins.hook_implementations  # noqa
        import ert.plugins.hook_specifications  # noqa

        self.add_hookspecs(ert.plugins.hook_specifications)
        if plugins is None:
            self.register(ert.plugins.hook_implementations)
            with warnings.catch_warnings():
                # If a deprecated plugin is installed, it may not be possible
                # for the user to avoid the FutureWarning, hence it should not be
                # displayed. Warnings should be displayed and logged when deprecated
                # plugins are actually used, not on every startup of Ert.
                warnings.simplefilter("ignore", category=FutureWarning)

                # logger.warning() statements also need to be muted:
                logger = logging.getLogger()
                orig_level = logger.level
                try:
                    logger.setLevel(logging.ERROR)
                    self.load_setuptools_entrypoints(_PLUGIN_NAMESPACE)
                finally:
                    logger.setLevel(orig_level)
        else:
            for plugin in plugins:
                self.register(plugin)

    def __str__(self) -> str:
        self_str = "ERT Plugin manager:\n"
        for plugin in self.get_plugins():
            self_str += "\t" + str(self.get_name(plugin)) + "\n"
            callers = self.get_hookcallers(plugin)
            if callers is not None:
                for hook_caller in callers:
                    self_str += "\t\t" + str(hook_caller) + "\n"
        return self_str

    def get_help_links(self) -> dict[str, Any]:
        return ErtPluginManager._merge_dicts(self.hook.help_links())

    @property
    def forward_model_steps(
        self,
    ) -> list[type[ForwardModelStepPlugin]]:
        fm_steps_listed = [
            resp.data for resp in self.hook.installable_forward_model_steps()
        ]
        return [fm_step for fm_steps in fm_steps_listed for fm_step in fm_steps]

    @staticmethod
    def _evaluate_config_hook(hook: pluggy.HookCaller, config_name: str) -> str | None:
        response = hook()

        if response is None:
            logger.debug(f"Got no {config_name} config path from any plugins")
            return None

        logger.debug(
            f"Got {config_name} config path from "
            f"{response.plugin_metadata.plugin_name} "
            f"({(response.plugin_metadata.function_name,)})"
        )
        return response.data

    @staticmethod
    def _evaluate_job_doc_hook(
        hook: pluggy.HookCaller, job_name: str
    ) -> dict[Any, Any]:
        response = hook(job_name=job_name)

        if response is None:
            logger.debug(f"Got no documentation for {job_name} from any plugins")
            return {}

        return response.data

    def get_forward_model_configuration(self) -> dict[str, dict[str, Any]]:
        response: list[PluginResponse[dict[str, str]]] = (
            self.hook.forward_model_configuration()
        )
        if response == []:
            return {}

        fm_configs: dict[str, dict[str, Any]] = collections.defaultdict(dict)
        for res in response:
            if not isinstance(res.data, dict):
                raise TypeError(
                    f"{res.plugin_metadata.plugin_name} did not return a dict"
                )

            for fmstep_name, fmstep_config in res.data.items():
                if not isinstance(fmstep_name, str) or not isinstance(
                    fmstep_config, dict
                ):
                    raise TypeError(
                        f"{res.plugin_metadata.plugin_name} did not "
                        "provide dict[str, dict[str, Any]]"
                    )
                for key, value in fmstep_config.items():
                    if not isinstance(key, str):
                        raise TypeError(
                            f"{res.plugin_metadata.plugin_name} did not "
                            f"provide dict[str, dict[str, Any]], got {key} "
                            "which was not a string."
                        )
                    if key.lower() in [
                        existing.lower() for existing in fm_configs[fmstep_name]
                    ]:
                        raise RuntimeError(
                            "Duplicate configuration or fm_step "
                            f"{fmstep_name} for key {key} when parsing plugin "
                            f"{res.plugin_metadata.plugin_name}, it is already "
                            "registered by another plugin."
                        )
                    fm_configs[fmstep_name][key] = value
        return fm_configs

    def get_site_configurations(self) -> ErtRuntimePlugins | None:
        try:
            plugin_response = self.hook.site_configurations()

            if len(plugin_response) == 0:
                return None

            if len(plugin_response) > 1:
                plugin_names = [
                    plugin.plugin_metadata.plugin_name for plugin in plugin_response
                ]
                raise ValueError(
                    f"Only one site configuration is allowed, got {plugin_names}"
                )
            return ErtRuntimePlugins.model_validate(plugin_response[0].data)
        except AttributeError:
            return None

    def get_installable_workflow_jobs(self) -> dict[str, str]:
        config_workflow_jobs = self._get_config_workflow_jobs()
        return config_workflow_jobs

    @staticmethod
    def _merge_internal_jobs(
        config_jobs: dict[str, str],
        hooked_jobs: dict[str, str],
    ) -> dict[str, str]:
        conflicting_keys = set(config_jobs.keys()) & set(hooked_jobs.keys())
        for ck in conflicting_keys:
            logger.info(
                f"Duplicate key: {ck} in workflow hook implementations, "
                f"config path 1: {config_jobs[ck]}, "
                f"config path 2: {hooked_jobs[ck]}"
            )
        merged_jobs = config_jobs.copy()
        merged_jobs.update(hooked_jobs)
        return merged_jobs

    @staticmethod
    def _add_plugin_info_to_dict(
        d: dict[K, V], plugin_response: PluginResponse[Any]
    ) -> dict[K, tuple[V, PluginMetadata]]:
        return {k: (v, plugin_response.plugin_metadata) for k, v in d.items()}

    @overload
    @staticmethod
    def _merge_dicts(
        list_of_dicts: list[PluginResponse[dict[str, V]]],
        include_plugin_data: Literal[True],
    ) -> dict[str, tuple[V, PluginMetadata]]:
        pass

    @overload
    @staticmethod
    def _merge_dicts(
        list_of_dicts: list[PluginResponse[dict[str, V]]],
        include_plugin_data: Literal[False] = False,
    ) -> dict[str, V]:
        pass

    @staticmethod
    def _merge_dicts(
        list_of_dicts: list[PluginResponse[dict[str, V]]],
        include_plugin_data: bool = False,
    ) -> dict[str, V] | dict[str, tuple[V, PluginMetadata]]:
        list_of_dicts.reverse()
        merged_dict: dict[str, tuple[V, PluginMetadata]] = {}
        for d in list_of_dicts:
            conflicting_keys = set(merged_dict.keys()) & set(d.data.keys())
            for ck in conflicting_keys:
                logger.info(
                    f"Overwriting {ck} from "
                    f"{merged_dict[ck][1].plugin_name}"
                    f"({merged_dict[ck][1].function_name}) "
                    f"with data from {d.plugin_metadata.plugin_name}"
                    f"({d.plugin_metadata.function_name})"
                )
            merged_dict.update(ErtPluginManager._add_plugin_info_to_dict(d.data, d))

        if include_plugin_data:
            return merged_dict
        return {k: v[0] for k, v in merged_dict.items()}

    def get_installable_jobs(self) -> Mapping[str, str]:
        return ErtPluginManager._merge_dicts(self.hook.installable_jobs())

    def _get_config_workflow_jobs(self) -> dict[str, str]:
        return ErtPluginManager._merge_dicts(self.hook.installable_workflow_jobs())

    def get_documentation_for_jobs(self) -> dict[str, Any]:
        job_docs = {
            k: {
                "config_file": v[0],
                "source_package": v[1].plugin_name,
                "source_function_name": v[1].function_name,
            }
            for k, v in ErtPluginManager._merge_dicts(
                self.hook.installable_jobs(), include_plugin_data=True
            ).items()
        }
        for key, value in job_docs.items():
            value.update(
                ErtPluginManager._evaluate_job_doc_hook(
                    self.hook.job_documentation,
                    key,
                )
            )
        return job_docs

    def get_documentation_for_forward_model_steps(
        self,
    ) -> dict[str, ForwardModelStepDocumentation]:
        return {
            # Implementations of plugin fm step take no __init__ args
            # (name, command)
            # but mypy expects the subclasses to take in same arguments upon
            # initializations
            fm_step().name: fm_step.documentation()  # type: ignore
            for fm_step in self.forward_model_steps
            if fm_step.documentation() is not None
        }

    def get_documentation_for_workflows(self) -> dict[str, JobDoc]:
        workflow_config = self.get_ertscript_workflows()
        legacy_workflow_config = self.get_legacy_ertscript_workflows()

        job_docs: dict[str, JobDoc] = {  # type: ignore
            workflow.name: {
                "description": workflow.description,
                "examples": workflow.examples,
                "parser": workflow_config.parsers[workflow.name],
                "config_file": None,
                "source_package": workflow.source_package,
                "category": workflow.category,
            }
            for workflow in workflow_config._workflows
        } | {
            workflow.name: {
                "description": workflow.description,
                "examples": workflow.examples,
                "parser": legacy_workflow_config.parsers[workflow.name],
                "config_file": None,
                "source_package": workflow.source_package,
                "category": workflow.category,
            }
            for workflow in legacy_workflow_config._workflows
        }

        fm_step_doc = self.get_documentation_for_forward_model_steps()
        for workflow_job in self.get_installable_workflow_jobs():
            job_docs[workflow_job] = JobDoc(
                {
                    "description": fm_step_doc[workflow_job].description,
                    "examples": None,  # Can not reuse FORWARD_MODEL example
                    "parser": None,
                    "config_file": fm_step_doc[workflow_job].config_file,
                    "source_package": fm_step_doc[workflow_job].source_package,
                    "category": fm_step_doc[workflow_job].category,
                }
            )

        return job_docs

    def get_ertscript_workflows(self) -> WorkflowConfigs:
        config = WorkflowConfigs()
        self.hook.ertscript_workflow(config=config)
        return config

    def get_legacy_ertscript_workflows(self) -> WorkflowConfigs:
        config = LegacyWorkflowConfigs()
        self.hook.legacy_ertscript_workflow(config=config)
        return config

    def add_logging_handle_to_root(self, logger: logging.Logger) -> None:
        handles = self.hook.add_log_handle_to_root()
        for handle in handles:
            logger.addHandler(handle)

    def add_span_processor_to_trace_provider(self) -> None:
        span_processors = self.hook.add_span_processor()
        for span_processor in span_processors:
            add_span_processor(span_processor)


class ErtRuntimePlugins(BaseModel):
    installed_forward_model_steps: Mapping[str, ForwardModelStep] = Field(
        default_factory=dict
    )
    installed_workflow_jobs: Mapping[str, WorkflowJob] = Field(default_factory=dict)
    queue_options: KnownQueueOptions | None = Field(
        default_factory=LocalQueueOptions, discriminator="name"
    )
    environment_variables: Mapping[str, str] = Field(default_factory=dict)
    env_pr_fm_step: Mapping[str, Mapping[str, Any]] = Field(default_factory=dict)
    help_links: dict[str, str] = Field(default_factory=dict)


def get_site_plugins(
    plugin_manager: ErtPluginManager | None = None,
) -> ErtRuntimePlugins:
    if plugin_manager is None:
        plugin_manager = ErtPluginManager()

    site_configurations = plugin_manager.get_site_configurations()
    installable_workflow_jobs = plugin_manager.get_installable_workflow_jobs()

    all_forward_model_steps = (
        dict(site_configurations.installed_forward_model_steps)
        if site_configurations
        else {}
    )

    for job_name, job_path in plugin_manager.get_installable_jobs().items():
        fm_step = forward_model_step_from_config_contents(
            Path(job_path).read_text(encoding="utf-8"), job_path, job_name
        )
        all_forward_model_steps[job_name] = fm_step

    all_workflow_jobs: dict[str, WorkflowJob] = dict[str, WorkflowJob](
        plugin_manager.get_ertscript_workflows().get_workflows()
    ) | dict[str, WorkflowJob](
        plugin_manager.get_legacy_ertscript_workflows().get_workflows()
    )

    for _, job_path in installable_workflow_jobs.items():
        wf_job = workflow_job_from_file(job_path)
        all_workflow_jobs[wf_job.name] = wf_job

    for fm_step_subclass in plugin_manager.forward_model_steps:
        # we call without required arguments to
        # ForwardModelStepPlugin.__init__ as
        # we expect the subclass to override __init__
        # and provide those arguments
        fm_step = fm_step_subclass()  # type: ignore
        all_forward_model_steps[fm_step.name] = fm_step

    runtime_plugins = ErtRuntimePlugins(
        installed_forward_model_steps=all_forward_model_steps,
        installed_workflow_jobs=all_workflow_jobs,
        queue_options=site_configurations.queue_options
        if site_configurations
        else None,
        environment_variables=(
            dict(site_configurations.environment_variables)
            if site_configurations
            else {}
        ),
        env_pr_fm_step=plugin_manager.get_forward_model_configuration(),
        help_links=plugin_manager.get_help_links(),
    )

    return runtime_plugins


def setup_site_logging(root_logger: logging.Logger | None = None) -> None:
    pm = ErtPluginManager()
    if root_logger is not None:
        pm.add_logging_handle_to_root(logger=root_logger)

    pm.add_span_processor_to_trace_provider()
