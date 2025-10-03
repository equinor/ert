import itertools
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import everest
from ert.config import (
    EnsembleConfig,
    ExecutableWorkflow,
    ModelConfig,
    WorkflowJob,
)
from ert.config.ert_config import (
    _substitutions_from_dict,
)
from ert.config.parsing import ConfigDict, ConfigWarning, read_file
from ert.config.parsing import ConfigKeys as ErtConfigKeys
from ert.plugins import ErtPluginContext
from everest.config import EverestConfig
from everest.config.forward_model_config import SummaryResults
from everest.config.install_job_config import InstallJobConfig
from everest.config.simulator_config import SimulatorConfig
from everest.strings import STORAGE_DIR


def extract_summary_keys(ever_config: EverestConfig) -> list[str]:
    summary_fms = [
        fm
        for fm in ever_config.forward_model
        if fm.results is not None and fm.results.type == "summary"
    ]

    if not summary_fms:
        return []

    summary_fm = summary_fms[0]
    assert summary_fm.results is not None

    smry_results = cast(SummaryResults, summary_fm.results)

    requested_keys: list[str] = ["*"] if smry_results.keys == "*" else smry_results.keys

    data_keys = everest.simulator.DEFAULT_DATA_SUMMARY_KEYS
    field_keys = everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS
    well_sum_keys = everest.simulator.DEFAULT_WELL_SUMMARY_KEYS
    deprecated_user_specified_keys = (
        [] if ever_config.export is None else ever_config.export.keywords
    )

    wells = [well.name for well in ever_config.wells]

    well_keys = [
        f"{sum_key}:{wname}"
        for (sum_key, wname) in itertools.product(well_sum_keys, wells)
    ]

    all_keys = data_keys + field_keys + well_keys + deprecated_user_specified_keys

    return list(set(all_keys + requested_keys))


def _extract_environment(
    ever_config: EverestConfig, ert_config: dict[str, Any]
) -> None:
    default_runpath_file = os.path.join(ever_config.output_dir, ".res_runpath_list")
    default_ens_path = os.path.join(ever_config.output_dir, STORAGE_DIR)

    ert_config[ErtConfigKeys.ENSPATH] = default_ens_path
    ert_config[ErtConfigKeys.RUNPATH_FILE] = default_runpath_file


def _extract_simulator(ever_config: EverestConfig, ert_config: dict[str, Any]) -> None:
    """
    Extracts simulation data from ever_config and injects it into ert_config.
    """

    ever_simulation = ever_config.simulator or SimulatorConfig()

    # Resubmit number (number of submission retries)
    ert_config[ErtConfigKeys.MAX_SUBMIT] = ever_simulation.resubmit_limit + 1

    # Maximum number of seconds (MAX_RUNTIME) a forward model is allowed to run
    max_runtime = ever_simulation.max_runtime
    if max_runtime is not None:
        ert_config[ErtConfigKeys.MAX_RUNTIME] = max_runtime or 0

    # Maximum amount of memory (REALIZATION_MEMORY) a forward model is allowed to use
    max_memory = ever_simulation.max_memory
    if max_memory is not None:
        ert_config[ErtConfigKeys.REALIZATION_MEMORY] = max_memory or 0

    # Number of cores reserved on queue nodes (NUM_CPU)
    num_fm_cpu = ever_simulation.cores_per_node
    if num_fm_cpu is not None:
        ert_config[ErtConfigKeys.NUM_CPU] = num_fm_cpu


def _job_to_dict(job: dict[str, Any] | InstallJobConfig) -> dict[str, Any]:
    if isinstance(job, InstallJobConfig):
        return job.model_dump(exclude_none=True)
    return job


def _extract_jobs(
    ever_config: EverestConfig, ert_config: dict[str, Any], path: str
) -> None:
    ever_jobs = [_job_to_dict(j) for j in ever_config.install_jobs]
    res_jobs = ert_config.get(ErtConfigKeys.INSTALL_JOB, [])
    for job in ever_jobs:
        if job.get("source") is not None:
            source_path = os.path.join(path, job["source"])
            new_job = (job["name"], (source_path, read_file(source_path)))
            res_jobs.append(new_job)

    ert_config[ErtConfigKeys.INSTALL_JOB] = res_jobs


def _extract_workflow_jobs(
    ever_config: EverestConfig, ert_config: dict[str, Any], path: str
) -> None:
    workflow_jobs = [_job_to_dict(j) for j in (ever_config.install_workflow_jobs or [])]

    res_jobs = ert_config.get(ErtConfigKeys.LOAD_WORKFLOW_JOB, [])
    for job in workflow_jobs:
        if job.get("source") is not None:
            new_job = (os.path.join(path, job["source"]), job["name"])
            res_jobs.append(new_job)

    if res_jobs:
        ert_config[ErtConfigKeys.LOAD_WORKFLOW_JOB] = res_jobs


def _extract_workflows(
    ever_config: EverestConfig, ert_config: dict[str, Any], path: str
) -> None:
    trigger2res = {
        "pre_simulation": "PRE_SIMULATION",
        "post_simulation": "POST_SIMULATION",
    }

    res_workflows = ert_config.get(ErtConfigKeys.LOAD_WORKFLOW, [])
    res_hooks = ert_config.get(ErtConfigKeys.HOOK_WORKFLOW, [])

    for ever_trigger, (workflow_file, jobs) in _get_workflow_files(ever_config).items():
        if jobs:
            res_trigger = trigger2res[ever_trigger]
            res_workflows.append((str(workflow_file), ever_trigger))
            res_hooks.append((ever_trigger, res_trigger))

    if res_workflows:
        ert_config[ErtConfigKeys.LOAD_WORKFLOW] = res_workflows
        ert_config[ErtConfigKeys.HOOK_WORKFLOW] = res_hooks


def _extract_seed(ever_config: EverestConfig, ert_config: dict[str, Any]) -> None:
    random_seed = ever_config.environment.random_seed

    if random_seed:
        ert_config[ErtConfigKeys.RANDOM_SEED] = random_seed


def get_substitutions(
    config_dict: ConfigDict, model_config: ModelConfig, runpath_file: Path, num_cpu: int
) -> dict[str, str]:
    substitutions = _substitutions_from_dict(config_dict)
    substitutions["<RUNPATH_FILE>"] = str(runpath_file)
    substitutions["<RUNPATH>"] = model_config.runpath_format_string
    substitutions["<ECL_BASE>"] = model_config.eclbase_format_string
    substitutions["<ECLBASE>"] = model_config.eclbase_format_string
    substitutions["<NUM_CPU>"] = str(num_cpu)
    return substitutions


def get_workflow_jobs(ever_config: EverestConfig) -> dict[str, WorkflowJob]:
    workflow_jobs: dict[str, WorkflowJob] = {}
    for job in ever_config.install_workflow_jobs or []:
        if job.executable is not None:
            if job.name in workflow_jobs:
                ConfigWarning.warn(
                    f"Duplicate workflow job with name {job.name!r}, "
                    f"overriding it with {job.executable!r}.",
                    job.name,
                )
            executable = Path(job.executable)
            if not executable.is_absolute():
                executable = ever_config.config_directory / executable
            workflow_jobs[job.name] = ExecutableWorkflow(
                name=job.name,
                min_args=None,
                max_args=None,
                arg_types=[],
                executable=str(executable),
            )
    return workflow_jobs


def get_ensemble_config(
    config_dict: ConfigDict, everest_config: EverestConfig
) -> EnsembleConfig:
    ensemble_config = EnsembleConfig.from_dict(config_dict)

    return ensemble_config


def _everest_to_ert_config_dict(ever_config: EverestConfig) -> ConfigDict:
    """
    Takes as input an Everest configuration and converts it
    to a corresponding ert config dict.
    """
    ert_config: dict[str, Any] = {}

    config_dir = ever_config.config_directory
    ert_config[ErtConfigKeys.DEFINE] = [
        ("<CONFIG_PATH>", config_dir),
        ("<CONFIG_FILE>", Path(ever_config.config_file).stem),
    ]

    # Extract simulator and simulation related configs
    _extract_simulator(ever_config, ert_config)
    _extract_environment(ever_config, ert_config)
    _extract_jobs(ever_config, ert_config, config_dir)
    _extract_workflow_jobs(ever_config, ert_config, config_dir)
    _extract_workflows(ever_config, ert_config, config_dir)
    _extract_seed(ever_config, ert_config)

    return ert_config


def everest_to_ert_config_dict(everest_config: EverestConfig) -> ConfigDict:
    with ErtPluginContext():
        config_dict = _everest_to_ert_config_dict(everest_config)
    return config_dict


def _get_well_file(ever_config: EverestConfig) -> tuple[Path, str]:
    assert ever_config.output_dir is not None
    data_storage = (Path(ever_config.output_dir) / ".internal_data").resolve()
    return (
        data_storage / "wells.json",
        json.dumps(
            [
                x.model_dump(exclude_none=True, exclude_unset=True)
                for x in ever_config.wells or []
            ]
        ),
    )


def _get_workflow_files(ever_config: EverestConfig) -> dict[str, tuple[Path, str]]:
    data_storage = (Path(ever_config.output_dir) / ".internal_data").resolve()
    return {
        trigger: (
            data_storage / f"{trigger}.workflow",
            "\n".join(getattr(ever_config.workflows, trigger, [])),
        )
        for trigger in ("pre_simulation", "post_simulation")
    }


def _get_install_data_files(ever_config: EverestConfig) -> Iterator[tuple[Path, str]]:
    data_storage = (Path(ever_config.output_dir) / ".internal_data").resolve()
    for item in ever_config.install_data or []:
        if item.data is not None:
            target, data = item.inline_data_as_str()
            yield (data_storage / Path(target).name, data)


def get_internal_files(ever_config: EverestConfig) -> dict[Path, str]:
    return dict(
        [
            _get_well_file(ever_config),
            *(
                (workflow_file, jobs)
                for workflow_file, jobs in _get_workflow_files(ever_config).values()
                if jobs
            ),
            *_get_install_data_files(ever_config),
        ],
    )
