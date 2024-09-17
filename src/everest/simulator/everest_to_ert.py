import collections
import itertools
import json
import logging
import os
from typing import Union

import everest
from everest.config import EverestConfig
from everest.config.install_data_config import InstallDataConfig
from everest.config.install_job_config import InstallJobConfig
from everest.config.simulator_config import SimulatorConfig
from everest.config_keys import ConfigKeys
from everest.queue_driver.queue_driver import _extract_queue_system
from everest.strings import EVEREST, SIMULATION_DIR, STORAGE_DIR
from everest.util.forward_models import collect_forward_models


def _get_datafiles(ever_config: EverestConfig):
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


def _load_all_groups(data_files):
    groups = []
    for data_file in data_files:
        groups += everest.util.read_groupnames(data_file)

    return set(groups)


def _load_all_wells(data_files):
    wells = []
    for data_file in data_files:
        wells += everest.util.read_wellnames(data_file)

    return set(wells)


def _extract_summary_keys(ever_config: EverestConfig, ert_config):
    data_files = _get_datafiles(ever_config)
    if len(data_files) == 0:
        return

    data_keys = everest.simulator.DEFAULT_DATA_SUMMARY_KEYS
    field_keys = everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS
    group_sum_keys = everest.simulator.DEFAULT_GROUP_SUMMARY_KEYS
    well_sum_keys = everest.simulator.DEFAULT_WELL_SUMMARY_KEYS
    user_specified_keys = (
        []
        if ever_config.export is None or ever_config.export.keywords is None
        else ever_config.export.keywords
    )

    # Makes it work w/ new config setup, default will be empty list
    # when old way of doing it is phased out
    if user_specified_keys is None:
        user_specified_keys = []

    try:
        groups = _load_all_groups(data_files)
    except Exception:
        warn_msg = (
            "Failed to load group names from {}. "
            "No group summary data will be internalized during run."
        )
        logging.getLogger("everest").warning(warn_msg.format(data_files))
        groups = []

    group_keys = [
        "{sum_key}:{gname}".format(sum_key=sum_key, gname=gname)
        for (sum_key, gname) in itertools.product(group_sum_keys, groups)
    ]

    try:
        data_wells = list(_load_all_wells(data_files))
    except Exception:
        warn_msg = (
            "Failed to load well names from {}. "
            "Only well data for wells specified in config file will be "
            "internalized during run."
        )
        logging.getLogger(EVEREST).warning(warn_msg.format(data_files))
        data_wells = []

    everest_wells = [well.name for well in ever_config.wells]
    wells = list(set(data_wells + everest_wells))

    well_keys = [
        "{sum_key}:{wname}".format(sum_key=sum_key, wname=wname)
        for (sum_key, wname) in itertools.product(well_sum_keys, wells)
    ]

    all_keys = (
        list(data_keys)
        + list(field_keys)
        + group_keys
        + well_keys
        + user_specified_keys
    )
    all_keys = list(set(all_keys))
    ert_config["SUMMARY"] = [all_keys]


def _extract_environment(ever_config: EverestConfig, ert_config):
    simulation_fmt = os.path.join(
        "<CASE_NAME>", "geo_realization_<GEO_ID>", SIMULATION_DIR
    )

    assert ever_config.simulation_dir is not None
    simulation_path = os.path.join(ever_config.simulation_dir, simulation_fmt)
    # load log configuration data

    assert ever_config.output_dir is not None
    default_runpath_file = os.path.join(ever_config.output_dir, ".res_runpath_list")
    default_ens_path = os.path.join(ever_config.output_dir, STORAGE_DIR)

    ert_config["RUNPATH"] = simulation_path
    ert_config["ENSPATH"] = default_ens_path
    ert_config["RUNPATH_FILE"] = default_runpath_file


def _inject_simulation_defaults(ert_config, ever_config: EverestConfig):
    """
    NOTE: This function is only to live until the effort of centralizing all
    default values is taken.
    """

    def inject_default(key, value):
        if key not in ert_config:
            ert_config[key] = value

    # Until dynamically configurable in res
    inject_default("NUM_REALIZATIONS", 10000)

    inject_default(
        "ECLBASE",
        (ever_config.definitions if ever_config.definitions is not None else {}).get(
            ConfigKeys.ECLBASE, "eclipse/ECL"
        ),
    )


def _extract_simulator(ever_config: EverestConfig, ert_config):
    """
    Extracts simulation data from ever_config and injects it into ert_config.
    """

    ever_simulation = ever_config.simulator or SimulatorConfig()

    # Resubmit number (number of submission retries)
    resubmit = ever_simulation.resubmit_limit
    if resubmit is not None:
        ert_config["MAX_SUBMIT"] = resubmit + 1

    # Maximum number of seconds (MAX_RUNTIME) a forward model is allowed to run
    max_runtime = ever_simulation.max_runtime
    if max_runtime is not None:
        ert_config["MAX_RUNTIME"] = max_runtime or 0

    # Number of cores reserved on queue nodes (NUM_CPU)
    num_fm_cpu = ever_simulation.cores_per_node
    if num_fm_cpu is not None:
        ert_config["NUM_CPU"] = num_fm_cpu

    _inject_simulation_defaults(ert_config, ever_config)


def _fetch_everest_jobs(ever_config: EverestConfig):
    """This injects the default Everest jobs when configuring res. In the
    future, this should be reviewed when we have proper configuration
    mechanisms in place."""
    assert ever_config.output_dir is not None
    job_storage = os.path.join(ever_config.output_dir, ".jobs")
    logging.getLogger(EVEREST).debug(
        "Creating job description files in %s" % job_storage
    )

    if not os.path.isdir(job_storage):
        os.makedirs(job_storage)

    ever_jobs = []
    Job = collections.namedtuple("Job", ["name", "source"])
    all_jobs = everest.jobs.script_names
    for default_job in all_jobs:
        script = everest.jobs.fetch_script(default_job)
        job_spec_file = os.path.join(job_storage, "_" + default_job)
        with open(job_spec_file, "w", encoding="utf-8") as f:
            f.write("EXECUTABLE %s" % script)

        ever_jobs.append(Job(name=default_job, source=job_spec_file))
    for job in collect_forward_models():
        ever_jobs.append(Job(name=job["name"], source=job["path"]))

    return ever_jobs


def _job_to_dict(job: Union[dict, InstallJobConfig]) -> Union[dict, InstallJobConfig]:
    if type(job) is InstallJobConfig:
        return job.model_dump(exclude_none=True)
    return job


def _extract_jobs(ever_config, ert_config, path):
    ever_jobs = [_job_to_dict(j) for j in (ever_config.install_jobs or [])]

    std_ever_jobs = _fetch_everest_jobs(ever_config)

    # Add standard Everest jobs
    job_names = [job[ConfigKeys.NAME] for job in ever_jobs]
    for default_job in std_ever_jobs:
        if default_job.name not in job_names:
            ever_jobs.append(
                {
                    ConfigKeys.NAME: default_job.name,
                    ConfigKeys.SOURCE: default_job.source,
                }
            )

    res_jobs = ert_config.get("INSTALL_JOB", [])
    for job in ever_jobs:
        new_job = (
            job[ConfigKeys.NAME],
            os.path.join(path, job[ConfigKeys.SOURCE]),
        )
        res_jobs.append(new_job)

    ert_config["INSTALL_JOB"] = res_jobs


def _extract_workflow_jobs(ever_config, ert_config, path):
    workflow_jobs = [_job_to_dict(j) for j in (ever_config.install_workflow_jobs or [])]

    res_jobs = ert_config.get("LOAD_WORKFLOW_JOB", [])
    for job in workflow_jobs:
        new_job = (
            os.path.join(path, job[ConfigKeys.SOURCE]),
            job[ConfigKeys.NAME],
        )
        res_jobs.append(new_job)

    if res_jobs:
        ert_config["LOAD_WORKFLOW_JOB"] = res_jobs


def _extract_workflows(ever_config, ert_config, path):
    trigger2res = {
        "pre_simulation": "PRE_SIMULATION",
        "post_simulation": "POST_SIMULATION",
    }

    res_workflows = ert_config.get("LOAD_WORKFLOW", [])
    res_hooks = ert_config.get("HOOK_WORKFLOW", [])

    for ever_trigger, res_trigger in trigger2res.items():
        jobs = getattr(ever_config.workflows, ever_trigger, None)
        if jobs is not None:
            name = os.path.join(path, f".{ever_trigger}.workflow")
            with open(name, "w", encoding="utf-8") as fp:
                fp.writelines(jobs)
            res_workflows.append((name, ever_trigger))
            res_hooks.append((ever_trigger, res_trigger))

    if res_workflows:
        ert_config["LOAD_WORKFLOW"] = res_workflows
        ert_config["HOOK_WORKFLOW"] = res_hooks


def _internal_data_files(ever_config: EverestConfig):
    assert ever_config.output_dir is not None
    data_storage = os.path.join(ever_config.output_dir, ".internal_data")
    data_storage = os.path.realpath(data_storage)
    logging.getLogger(EVEREST).debug("Storing internal data in %s" % data_storage)

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


def _expand_source_path(source, ever_config: EverestConfig):
    """Expands <CONFIG_PATH> in @source and makes it absolute to config
    directory if relative.
    """
    assert ever_config.config_directory is not None
    config_dir = ever_config.config_directory
    source = source.replace("<CONFIG_PATH>", config_dir)
    if not os.path.isabs(source):
        source = os.path.join(config_dir, source)
    return source


def _is_dir_all_geo(source, ever_config: EverestConfig):
    """Expands <GEO_ID> for all realizations and if:
    - all are directories, returns True,
    - all are files, returns False,
    - some are non-existing, raises an AssertionError
    """
    realizations = ever_config.model.realizations
    if not realizations:
        msg = "Expected realizations when analysing data installation source"
        raise AssertionError(msg)

    is_dir = []
    for geo_id in realizations:
        geo_source = source.replace("<GEO_ID>", str(geo_id))
        if not os.path.exists(geo_source):
            msg = (
                "Expected source to exist for data installation, "
                "did not find: {}".format(geo_source)
            )
            raise AssertionError(msg)

        is_dir.append(os.path.isdir(geo_source))

    if set(is_dir) == {True, False}:
        msg = "Source: {} represent both files and directories".format(source)
        raise ValueError(msg)

    return is_dir[0]


def _extract_data_operations(ever_config: EverestConfig):
    symlink_fmt = "symlink {source} {link_name}"
    copy_dir_fmt = "copy_directory {source} {target}"
    copy_file_fmt = "copy_file {source} {target}"

    forward_model = []
    install_data = ever_config.install_data or []
    install_data += [
        InstallDataConfig(
            **{
                ConfigKeys.SOURCE: datafile,
                ConfigKeys.TARGET: os.path.basename(datafile),
            }
        )
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


def _extract_templating(ever_config: EverestConfig):
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


def _insert_strip_dates_job(everest_config: EverestConfig, forward_model):
    report_steps = everest_config.model.report_steps

    if report_steps:
        simulation_idx = [
            idx
            for idx, model in enumerate(forward_model)
            if "eclipse" in model.split()[0] or "flow" in model.split()[0]
        ]

        strip_dates_job_str = "{job_name} {args}".format(
            job_name="strip_dates",
            args="--summary {file} --dates {dates}".format(
                file="<ECLBASE>.UNSMRY", dates=" ".join(report_steps)
            ),
        )

        for idx in simulation_idx:
            forward_model.insert(idx + 1, strip_dates_job_str)
    return forward_model


def _extract_forward_model(ever_config: EverestConfig, ert_config):
    forward_model = _extract_data_operations(ever_config)
    forward_model += _extract_templating(ever_config)
    forward_model += ever_config.forward_model or []
    forward_model = _insert_strip_dates_job(ever_config, forward_model)

    sim_job = ert_config.get("SIMULATION_JOB", [])
    for job in forward_model:
        tmp = job.split()
        sim_job.append(tuple(tmp))

    ert_config["SIMULATION_JOB"] = sim_job


def _extract_model(ever_config: EverestConfig, ert_config):
    _extract_summary_keys(ever_config, ert_config)


def _extract_seed(ever_config: EverestConfig, ert_config):
    assert ever_config.environment is not None
    random_seed = ever_config.environment.random_seed

    if random_seed:
        ert_config["RANDOM_SEED"] = random_seed


def everest_to_ert_config(ever_config: EverestConfig, site_config=None):
    """
    Takes as input an Everest configuration, the site-config and converts them
    to a corresponding ert configuration.
    """
    ert_config = site_config if site_config is not None else {}

    config_dir = ever_config.config_directory
    ert_config["DEFINE"] = [("<CONFIG_PATH>", config_dir)]

    # Extract simulator and simulation related configs
    _extract_simulator(ever_config, ert_config)
    _extract_forward_model(ever_config, ert_config)
    _extract_environment(ever_config, ert_config)
    _extract_jobs(ever_config, ert_config, config_dir)
    _extract_workflow_jobs(ever_config, ert_config, config_dir)
    _extract_workflows(ever_config, ert_config, config_dir)
    _extract_model(ever_config, ert_config)
    _extract_queue_system(ever_config, ert_config)
    _extract_seed(ever_config, ert_config)

    return ert_config
