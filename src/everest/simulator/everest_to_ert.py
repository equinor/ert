import collections
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Any

import everest
from ert.config import (
    EnsembleConfig,
    ErtConfig,
    ExtParamConfig,
    ForwardModelStep,
    ModelConfig,
)
from ert.config.ert_config import (
    _substitutions_from_dict,
    create_list_of_forward_model_steps_to_run,
    installed_forward_model_steps_from_dict,
    uppercase_subkeys_and_stringify_subvalues,
)
from ert.config.parsing import ConfigDict
from ert.config.parsing import ConfigKeys as ErtConfigKeys
from ert.plugins import ErtPluginContext
from ert.plugins.plugin_manager import ErtPluginManager
from ert.substitutions import Substitutions
from everest.config import EverestConfig
from everest.config.control_variable_config import (
    ControlVariableConfig,
    ControlVariableGuessListConfig,
)
from everest.config.install_data_config import InstallDataConfig
from everest.config.install_job_config import InstallJobConfig
from everest.config.simulator_config import SimulatorConfig
from everest.strings import EVEREST, SIMULATION_DIR, STORAGE_DIR


def _get_datafiles(ever_config: EverestConfig) -> list[str]:
    ever_model = ever_config.model
    data_file = ever_model.data_file
    if data_file is None:
        return []

    # Make absolute path
    if os.path.relpath(data_file):
        config_path = ever_config.config_directory
        assert isinstance(config_path, str)
        data_file = os.path.join(config_path, data_file)
    data_file = os.path.realpath(data_file)

    # Render all iterations
    realizations = ever_model.realizations
    if not realizations:
        return [data_file]

    return [data_file.replace("<GEO_ID>", str(geo_id)) for geo_id in realizations]


def _extract_summary_keys(
    ever_config: EverestConfig, ert_config: dict[str, Any]
) -> None:
    data_files = _get_datafiles(ever_config)
    if len(data_files) == 0:
        return

    data_keys = everest.simulator.DEFAULT_DATA_SUMMARY_KEYS
    field_keys = everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS
    well_sum_keys = everest.simulator.DEFAULT_WELL_SUMMARY_KEYS
    user_specified_keys = (
        [] if ever_config.export is None else ever_config.export.keywords
    )

    wells = [well.name for well in ever_config.wells]

    well_keys = [
        f"{sum_key}:{wname}"
        for (sum_key, wname) in itertools.product(well_sum_keys, wells)
    ]

    all_keys = data_keys + field_keys + well_keys + user_specified_keys
    all_keys = list(set(all_keys))
    ert_config[ErtConfigKeys.SUMMARY] = [all_keys]


def _extract_environment(
    ever_config: EverestConfig, ert_config: dict[str, Any]
) -> None:
    simulation_fmt = os.path.join(
        "batch_<ITER>", "geo_realization_<GEO_ID>", SIMULATION_DIR
    )

    assert ever_config.simulation_dir is not None
    simulation_path = os.path.join(ever_config.simulation_dir, simulation_fmt)
    # load log configuration data

    assert ever_config.output_dir is not None
    default_runpath_file = os.path.join(ever_config.output_dir, ".res_runpath_list")
    default_ens_path = os.path.join(ever_config.output_dir, STORAGE_DIR)

    ert_config[ErtConfigKeys.RUNPATH] = simulation_path
    ert_config[ErtConfigKeys.ENSPATH] = default_ens_path
    ert_config[ErtConfigKeys.RUNPATH_FILE] = default_runpath_file


def _extract_simulator(ever_config: EverestConfig, ert_config: dict[str, Any]) -> None:
    """
    Extracts simulation data from ever_config and injects it into ert_config.
    """

    ever_simulation = ever_config.simulator or SimulatorConfig()

    # Resubmit number (number of submission retries)
    resubmit = ever_simulation.resubmit_limit
    if resubmit is not None:
        ert_config[ErtConfigKeys.MAX_SUBMIT] = resubmit + 1

    # Maximum number of seconds (MAX_RUNTIME) a forward model is allowed to run
    max_runtime = ever_simulation.max_runtime
    if max_runtime is not None:
        ert_config[ErtConfigKeys.MAX_RUNTIME] = max_runtime or 0

    # Number of cores reserved on queue nodes (NUM_CPU)
    num_fm_cpu = ever_simulation.cores_per_node
    if num_fm_cpu is not None:
        ert_config[ErtConfigKeys.NUM_CPU] = num_fm_cpu


def _fetch_everest_jobs(ever_config: EverestConfig) -> list[Any]:
    """This injects the default Everest jobs when configuring res. In the
    future, this should be reviewed when we have proper configuration
    mechanisms in place."""
    assert ever_config.output_dir is not None
    job_storage = os.path.join(ever_config.output_dir, ".jobs")
    logging.getLogger(EVEREST).debug(f"Creating job description files in {job_storage}")

    if not os.path.isdir(job_storage):
        os.makedirs(job_storage)

    ever_jobs = []
    Job = collections.namedtuple("Job", ["name", "source"])
    all_jobs = everest.jobs.script_names
    for default_job in all_jobs:
        script = everest.jobs.fetch_script(default_job)
        job_spec_file = os.path.join(job_storage, "_" + default_job)
        with open(job_spec_file, "w", encoding="utf-8") as f:
            f.write(f"EXECUTABLE {script}")

        ever_jobs.append(Job(name=default_job, source=job_spec_file))

    return ever_jobs


def _job_to_dict(job: dict[str, Any] | InstallJobConfig) -> dict[str, Any]:
    if isinstance(job, InstallJobConfig):
        return job.model_dump(exclude_none=True)
    return job


def _extract_jobs(
    ever_config: EverestConfig, ert_config: dict[str, Any], path: str
) -> None:
    ever_jobs = [_job_to_dict(j) for j in (ever_config.install_jobs or [])]

    std_ever_jobs = _fetch_everest_jobs(ever_config)

    # Add standard Everest jobs
    job_names = [job["name"] for job in ever_jobs]
    for default_job in std_ever_jobs:
        if default_job.name not in job_names:
            ever_jobs.append(
                {
                    "name": default_job.name,
                    "source": default_job.source,
                }
            )

    res_jobs = ert_config.get(ErtConfigKeys.INSTALL_JOB, [])
    for job in ever_jobs:
        new_job = (
            job["name"],
            os.path.join(path, job["source"]),
        )
        res_jobs.append(new_job)

    ert_config[ErtConfigKeys.INSTALL_JOB] = res_jobs
    if eclbase := ever_config.definitions.get("eclbase"):
        ert_config["ECLBASE"] = eclbase


def _extract_workflow_jobs(
    ever_config: EverestConfig, ert_config: dict[str, Any], path: str
) -> None:
    workflow_jobs = [_job_to_dict(j) for j in (ever_config.install_workflow_jobs or [])]

    res_jobs = ert_config.get(ErtConfigKeys.LOAD_WORKFLOW_JOB, [])
    for job in workflow_jobs:
        new_job = (
            os.path.join(path, job["source"]),
            job["name"],
        )
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

    for ever_trigger, res_trigger in trigger2res.items():
        jobs = getattr(ever_config.workflows, ever_trigger, None)
        if jobs is not None:
            name = os.path.join(path, f".{ever_trigger}.workflow")
            with open(name, "w", encoding="utf-8") as fp:
                fp.writelines(jobs)
            res_workflows.append((name, ever_trigger))
            res_hooks.append((ever_trigger, res_trigger))

    if res_workflows:
        ert_config[ErtConfigKeys.LOAD_WORKFLOW] = res_workflows
        ert_config[ErtConfigKeys.HOOK_WORKFLOW] = res_hooks


def _internal_data_files(ever_config: EverestConfig) -> tuple[str]:
    assert ever_config.output_dir is not None
    data_storage = os.path.join(ever_config.output_dir, ".internal_data")
    data_storage = os.path.realpath(data_storage)
    logging.getLogger(EVEREST).debug(f"Storing internal data in {data_storage}")

    if not os.path.isdir(data_storage):
        os.makedirs(data_storage)

    well_datafile = os.path.join(data_storage, "wells.json")
    with open(well_datafile, "w", encoding="utf-8") as fout:
        json.dump(
            [
                x.model_dump(exclude_none=True, exclude_unset=True)
                for x in ever_config.wells or []
            ],
            fout,
        )

    return (well_datafile,)


def _expand_source_path(source: str, ever_config: EverestConfig) -> str:
    """Expands <CONFIG_PATH> in @source and makes it absolute to config
    directory if relative.
    """
    assert ever_config.config_directory is not None
    config_dir = ever_config.config_directory
    source = source.replace("<CONFIG_PATH>", config_dir)
    if not os.path.isabs(source):
        source = os.path.join(config_dir, source)
    return source


def _is_dir_all_geo(source: str, ever_config: EverestConfig) -> bool:
    """Expands <GEO_ID> for all realizations and if:
    - all are directories, returns True,
    - all are files, returns False,
    - some are non-existing, raises an AssertionError
    """
    realizations = ever_config.model.realizations
    if not realizations:
        msg = "Expected realizations when analysing data installation source"
        raise ValueError(msg)

    is_dir = []
    for geo_id in realizations:
        geo_source = source.replace("<GEO_ID>", str(geo_id))
        if not os.path.exists(geo_source):
            msg = (
                "Expected source to exist for data installation, "
                f"did not find: {geo_source}"
            )
            raise ValueError(msg)

        is_dir.append(os.path.isdir(geo_source))

    if set(is_dir) == {True, False}:
        msg = f"Source: {source} represent both files and directories"
        raise ValueError(msg)

    return is_dir[0]


def _extract_data_operations(ever_config: EverestConfig) -> list[str]:
    symlink_fmt = "symlink {source} {link_name}"
    copy_dir_fmt = "copy_directory {source} {target}"
    copy_file_fmt = "copy_file {source} {target}"

    forward_model = []
    install_data = ever_config.install_data or []
    install_data += [
        InstallDataConfig(source=datafile, target=os.path.basename(datafile))
        for datafile in _internal_data_files(ever_config)
    ]

    for data_req in install_data:
        target = data_req.target

        source = _expand_source_path(data_req.source, ever_config)
        is_dir = _is_dir_all_geo(source, ever_config)

        if data_req.link:
            forward_model.append(symlink_fmt.format(source=source, link_name=target))
        elif is_dir:
            forward_model.append(copy_dir_fmt.format(source=source, target=target))
        else:
            forward_model.append(copy_file_fmt.format(source=source, target=target))

    return forward_model


def _extract_templating(ever_config: EverestConfig) -> list[str]:
    res_input = [control.name for control in ever_config.controls]
    res_input = [fn + ".json" for fn in res_input]
    res_input += _internal_data_files(ever_config)

    forward_model = []
    install_templates = ever_config.install_templates or []
    for tmpl_request in install_templates:
        # User can define a template w/ extra data to be used with it,
        # append file as arg to input_files if declared.
        if tmpl_request.extra_data is not None:
            res_input.append(tmpl_request.extra_data)

        args = " ".join(
            [
                "--output",
                tmpl_request.output_file,
                "--template",
                tmpl_request.template,
                "--input_files",
                *res_input,
            ]
        )
        forward_model.append(f"render {args}")

    return forward_model


def _extract_forward_model(
    ever_config: EverestConfig, ert_config: dict[str, Any]
) -> None:
    forward_model = _extract_data_operations(ever_config)
    forward_model += _extract_templating(ever_config)
    forward_model += ever_config.forward_model or []

    fm_steps = ert_config.get(ErtConfigKeys.FORWARD_MODEL, [])
    for job in forward_model:
        job_name, *args = job.split()
        match job_name:
            # All three reservoir simulator fm_steps map to
            # "run_reservoirsimulator" which requires the simulator name
            # as its first argument.
            case "eclipse100":
                fm_steps.append(["eclipse100", ["eclipse", *args]])
            case "eclipse300":
                fm_steps.append(["eclipse300", ["e300", *args]])
            case "flow":
                fm_steps.append(["flow", ["flow", *args]])
            case _:
                fm_steps.append([job_name, args])

    ert_config[ErtConfigKeys.FORWARD_MODEL] = fm_steps


def _extract_model(ever_config: EverestConfig, ert_config: dict[str, Any]) -> None:
    _extract_summary_keys(ever_config, ert_config)

    if ErtConfigKeys.NUM_REALIZATIONS not in ert_config:
        if ever_config.model.realizations is not None:
            ert_config[ErtConfigKeys.NUM_REALIZATIONS] = len(
                ever_config.model.realizations
            )
        else:
            ert_config[ErtConfigKeys.NUM_REALIZATIONS] = 1


def _extract_seed(ever_config: EverestConfig, ert_config: dict[str, Any]) -> None:
    random_seed = ever_config.environment.random_seed

    if random_seed:
        ert_config[ErtConfigKeys.RANDOM_SEED] = random_seed


def _extract_results(ever_config: EverestConfig, ert_config: dict[str, Any]) -> None:
    objectives_names = [objective.name for objective in ever_config.objective_functions]
    constraint_names = [
        constraint.name for constraint in (ever_config.output_constraints or [])
    ]
    gen_data = ert_config.get(ErtConfigKeys.GEN_DATA, [])
    for name in objectives_names + constraint_names:
        gen_data.append((name, {"RESULT_FILE": name}))
    ert_config[ErtConfigKeys.GEN_DATA] = gen_data


def get_substitutions(
    config_dict: ConfigDict, model_config: ModelConfig, runpath_file: Path, num_cpu: int
) -> Substitutions:
    substitutions = _substitutions_from_dict(config_dict)
    substitutions["<RUNPATH_FILE>"] = str(runpath_file)
    substitutions["<RUNPATH>"] = model_config.runpath_format_string
    substitutions["<ECL_BASE>"] = model_config.eclbase_format_string
    substitutions["<ECLBASE>"] = model_config.eclbase_format_string
    substitutions["<NUM_CPU>"] = str(num_cpu)
    return substitutions


def get_forward_model_steps(
    config_dict: ConfigDict, substitutions: Substitutions
) -> tuple[list[ForwardModelStep], dict[str, dict[str, Any]]]:
    installed_forward_model_steps: dict[str, ForwardModelStep] = {}
    pm = ErtPluginManager()
    for fm_step_subclass in pm.forward_model_steps:
        fm_step = fm_step_subclass()  # type: ignore
        installed_forward_model_steps[fm_step.name] = fm_step

    installed_forward_model_steps.update(
        installed_forward_model_steps_from_dict(config_dict)
    )

    env_pr_fm_step = uppercase_subkeys_and_stringify_subvalues(
        pm.get_forward_model_configuration()
    )

    forward_model_steps = create_list_of_forward_model_steps_to_run(
        installed_forward_model_steps,
        substitutions,
        config_dict,
        installed_forward_model_steps,
        env_pr_fm_step,
    )

    return forward_model_steps, env_pr_fm_step


def get_ensemble_config(
    config_dict: ConfigDict, everest_config: EverestConfig
) -> EnsembleConfig:
    ensemble_config = EnsembleConfig.from_dict(config_dict)

    def _get_variables(
        variables: list[ControlVariableConfig] | list[ControlVariableGuessListConfig],
    ) -> list[str] | dict[str, list[str]]:
        if (
            isinstance(variables[0], ControlVariableConfig)
            and getattr(variables[0], "index", None) is None
        ):
            return [var.name for var in variables]
        result: collections.defaultdict[str, list] = collections.defaultdict(list)  # type: ignore
        for variable in variables:
            if isinstance(variable, ControlVariableGuessListConfig):
                result[variable.name].extend(
                    str(index + 1) for index, _ in enumerate(variable.initial_guess)
                )
            else:
                result[variable.name].append(str(variable.index))
        return dict(result)

    # This adds an EXT_PARAM key to the ert_config, which is not a true ERT
    # configuration key. When initializing an ERT config object, it is ignored.
    # It is used by the Simulator object to inject ExtParamConfig nodes.
    for control in everest_config.controls or []:
        ensemble_config.parameter_configs[control.name] = ExtParamConfig(
            name=control.name,
            input_keys=_get_variables(control.variables),
            output_file=control.name + ".json",
        )

    return ensemble_config


def _everest_to_ert_config_dict(
    ever_config: EverestConfig, site_config: dict[Any, Any] | None = None
) -> ConfigDict:
    """
    Takes as input an Everest configuration, the site-config and converts them
    to a corresponding ert configuration.
    """
    ert_config = site_config if site_config is not None else {}

    config_dir = ever_config.config_directory
    ert_config[ErtConfigKeys.DEFINE] = [("<CONFIG_PATH>", config_dir)]

    # Extract simulator and simulation related configs
    _extract_simulator(ever_config, ert_config)
    _extract_forward_model(ever_config, ert_config)
    _extract_environment(ever_config, ert_config)
    _extract_jobs(ever_config, ert_config, config_dir)
    _extract_workflow_jobs(ever_config, ert_config, config_dir)
    _extract_workflows(ever_config, ert_config, config_dir)
    _extract_model(ever_config, ert_config)
    _extract_seed(ever_config, ert_config)
    _extract_results(ever_config, ert_config)
    if ert_config.get("SUMMARY") and not ert_config.get("ECLBASE"):
        raise ValueError(
            "When specifying model -> data_file, also need to configure definitions -> eclbase, this will trigger "
            "loading of summary data from the forward model"
        )
    return ert_config


def everest_to_ert_config_dict(everest_config: EverestConfig) -> ConfigDict:
    with ErtPluginContext():
        config_dict = _everest_to_ert_config_dict(
            everest_config, site_config=ErtConfig.read_site_config()
        )
    return config_dict
