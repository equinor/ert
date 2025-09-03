import contextlib
import datetime
import operator
import os
import os.path
import pathlib
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin
from warnings import filterwarnings

import hypothesis.strategies as st
from hypothesis import assume, note
from py import path as py_path
from pydantic import PositiveInt

from ert.config._read_summary import make_summary_key
from ert.config.field import TRANSFORM_FUNCTIONS
from ert.config.parsing import (
    ConfigKeys,
    HistorySource,
    QueueSystem,
    QueueSystemWithGeneric,
    parse_observations,
)
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
)

from .egrid_generator import EGrid, egrids
from .observations_generator import (
    HistoryObservation,
    Observation,
    SummaryObservation,
    observations,
)
from .summary_generator import Date, Smspec, Unsmry, smspecs, summary_variables, unsmrys

words = st.text(
    min_size=4,
    max_size=8,
    alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
)


def touch(filename):
    with open(file=filename, mode="w", encoding="utf-8") as fh:
        fh.write(" ")


file_names = words
format_result_file_name = st.builds(lambda file_name: file_name + "-%d", file_names)
format_runpath_file_name = st.builds(
    lambda file_name: file_name + "-<IENS>", file_names
)


def small_list(*arg, max_size=5, **kw_args):
    return st.lists(*arg, **kw_args, max_size=max_size)


@st.composite
def field_output_names(draw):
    fname = draw(words)
    ext = draw(
        st.sampled_from(["roff_binary", "roff_ascii", "roff", "grdecl", "bgrdecl"])
    )
    return f"{fname}.{ext}"


@st.composite
def directory_names(draw):
    return "dir" + draw(words)


@st.composite
def report_steps(draw):
    rep_steps = draw(
        st.lists(st.integers(min_value=0, max_value=100), min_size=4, unique=True)
    )
    # We generate report_steps so that it always contains 0,
    # this is so that there is always a report_step we can match
    if 0 not in rep_steps:
        rep_steps.append(0)
    return ",".join(str(step) for step in sorted(rep_steps))


booleans = st.booleans()
transforms = st.sampled_from(list(TRANSFORM_FUNCTIONS))
small_floats = st.floats(min_value=1.0, max_value=10.0, allow_nan=False)
positives = st.integers(min_value=1, max_value=10000)
queue_systems = st.sampled_from(QueueSystem)
memory_unit_torque = st.sampled_from(["kb", "mb", "gb"])
memory_unit_lsf = st.sampled_from(
    ["", "KB", "K", "MB", "M", "GB", "G", "TB", "T", "PB", "P", "EB", "E", "ZB", "Z"]
)


@st.composite
def memory_with_unit_torque(draw):
    return f"{draw(positives)}{draw(memory_unit_torque)}"


@st.composite
def memory_with_unit_lsf(draw):
    return f"{draw(positives)}{draw(memory_unit_lsf)}"


memory_with_unit = {
    QueueSystemWithGeneric.TORQUE: memory_with_unit_torque,
    QueueSystemWithGeneric.LSF: memory_with_unit_lsf,
    QueueSystemWithGeneric.LOCAL: memory_with_unit_lsf,  # Just a dummy value
    QueueSystemWithGeneric.GENERIC: memory_with_unit_lsf,  # Just a dummy value
}

queue_systems_and_options = {
    QueueSystemWithGeneric.LOCAL: LocalQueueOptions,
    QueueSystemWithGeneric.LSF: LsfQueueOptions,
    QueueSystemWithGeneric.TORQUE: TorqueQueueOptions,
    QueueSystemWithGeneric.SLURM: SlurmQueueOptions,
    QueueSystemWithGeneric.GENERIC: QueueOptions,
}


def valid_queue_options(queue_system: str):
    return [
        name.upper()
        for name in queue_systems_and_options[
            QueueSystemWithGeneric(queue_system)
        ].model_fields
        if name != "name"
    ]


def has_base_type(
    field_type, base_type: type[int] | bool | type[str] | type[float]
) -> bool:
    if field_type is base_type:
        return True
    origin = get_origin(field_type)
    if origin:
        args = get_args(field_type)
        if any(arg is base_type for arg in args):
            return True
        return any(has_base_type(arg, base_type) for arg in args)
    return False


queue_options_by_type: dict[str, dict[str, list[str]]] = defaultdict(dict)
for system, options in queue_systems_and_options.items():
    queue_options_by_type["string"][system.name] = [
        name.upper()
        for name, field in options.model_fields.items()
        if has_base_type(field.annotation, str) and "memory" not in name
    ]
    queue_options_by_type["bool"][system.name] = [
        name.upper()
        for name, field in options.model_fields.items()
        if has_base_type(field.annotation, bool)
    ]
    queue_options_by_type["posint"][system.name] = [
        name.upper()
        for name, field in options.model_fields.items()
        if has_base_type(field.annotation, int)
    ]
    queue_options_by_type["posfloat"][system.name] = [
        name.upper()
        for name, field in options.model_fields.items()
        if has_base_type(field.annotation, float)
    ]
    queue_options_by_type["memory"][system.name] = [
        name.upper() for name in options.model_fields if "memory" in name
    ]


def valid_queue_values(option_name, queue_system):
    if option_name in queue_options_by_type["string"][queue_system]:
        return words
    elif option_name in queue_options_by_type["posfloat"][queue_system]:
        return small_floats.map(str)
    elif option_name in queue_options_by_type["posint"][queue_system]:
        return positives.map(str)
    elif option_name in queue_options_by_type["bool"][queue_system]:
        return booleans.map(str)
    elif option_name in queue_options_by_type["memory"][queue_system]:
        return memory_with_unit[queue_system]()
    else:
        raise ValueError(
            "config_dict_generator does not know how to "
            f"generate values for {option_name}"
        )


@st.composite
def queue_options(draw, systems):
    queue_system = draw(systems)
    queue_option = draw(st.sampled_from(valid_queue_options(queue_system)))

    # The JOB_SCRIPT queue_option must have a value
    if draw(booleans) or queue_option == "JOB_SCRIPT":
        return [
            queue_system,
            queue_option,
            draw(valid_queue_values(queue_option, queue_system.name)),
        ]
    else:
        # Missing VALUE means unset
        return [queue_system, queue_option]


def default_forward_model_names():
    return (
        st.sampled_from(
            [
                "DELETE_FILE",
                "move_directory",
                "move_file",
                "make_directory",
                "rms",
                "CAREFUL_COPY_FILE",
                "RMS",
                "flow",
                "COPY_FILE",
                "careful_copy_file",
                "ECLIPSE100",
                "delete_file",
                "delete_directory",
                "DELETE_DIRECTORY",
                "FLOW",
                "ECLIPSE300",
                "eclipse300",
                "MOVE_DIRECTORY",
                "MOVE_FILE",
                "template_render",
                "TEMPLATE_RENDER",
                "COPY_DIRECTORY",
                "symlink",
                "copy_directory",
                "SYMLINK",
                "MAKE_SYMLINK",
                "eclipse100",
                "make_symlink",
                "MAKE_DIRECTORY",
                "copy_file",
            ]
        )
        if os.getenv("ERT_SITE_CONFIG", None) is None
        else st.nothing()
    )


@st.composite
def random_forward_model_names(draw, some_words, some_file_names):
    return draw(
        st.tuples(
            some_words,
            st.just(draw(some_file_names) + "job_config"),
        )
    )


@dataclass
class ErtConfigValues:
    num_realizations: PositiveInt
    eclbase: str | None
    runpath_file: str
    run_template: list[str]
    enkf_alpha: float
    update_log_path: str
    std_cutoff: float
    max_runtime: PositiveInt
    min_realizations: str
    define: list[tuple[str, str]]
    forward_model: tuple[str, list[tuple[str, str]]]
    stop_long_running: bool
    data_kw_key: list[tuple[str, str]]
    data_file: str
    grid_file: str
    job_script: str
    jobname: str | None
    runpath: str
    enspath: str
    time_map: tuple[str, str]
    obs_config: str
    history_source: HistorySource
    refcase: str
    gen_kw_export_name: str
    field: list[tuple[str, str, str, dict[str, str]]]
    gen_data: list[tuple[str, dict[str, str]]]
    max_submit: PositiveInt
    num_cpu: PositiveInt
    queue_system: Literal["LSF", "LOCAL", "TORQUE", "SLURM"]
    queue_option: list[tuple[str, str] | tuple[str, str, str]]
    analysis_set_var: list[tuple[str, str, Any]]
    install_job: list[tuple[str, str]]
    install_job_directory: list[str]
    license_path: str
    random_seed: int
    setenv: list[tuple[str, str]]
    observations: list[Observation]
    refcase_smspec: Smspec
    refcase_unsmry: Unsmry
    egrid: EGrid

    def to_config_dict(self, config_file, cwd, all_defines=True):
        result = {
            ConfigKeys.FORWARD_MODEL: self.forward_model,
            ConfigKeys.NUM_REALIZATIONS: self.num_realizations,
            ConfigKeys.RUNPATH_FILE: self.runpath_file,
            ConfigKeys.RUN_TEMPLATE: self.run_template,
            ConfigKeys.ENKF_ALPHA: self.enkf_alpha,
            ConfigKeys.UPDATE_LOG_PATH: self.update_log_path,
            ConfigKeys.STD_CUTOFF: self.std_cutoff,
            ConfigKeys.MAX_RUNTIME: self.max_runtime,
            ConfigKeys.MIN_REALIZATIONS: self.min_realizations,
            ConfigKeys.DEFINE: (
                self.all_defines(config_file, cwd) if all_defines else self.define
            ),
            ConfigKeys.STOP_LONG_RUNNING: self.stop_long_running,
            ConfigKeys.DATA_KW: self.data_kw_key,
            ConfigKeys.DATA_FILE: self.data_file,
            ConfigKeys.GRID: self.grid_file,
            ConfigKeys.JOB_SCRIPT: self.job_script,
            ConfigKeys.RUNPATH: self.runpath,
            ConfigKeys.ENSPATH: self.enspath,
            ConfigKeys.TIME_MAP: self.time_map,
            ConfigKeys.OBS_CONFIG: (
                self.obs_config,
                parse_observations(
                    "\n".join(str(o) for o in self.observations), config_file
                ),
            ),
            ConfigKeys.HISTORY_SOURCE: self.history_source,
            ConfigKeys.REFCASE: self.refcase,
            ConfigKeys.GEN_KW_EXPORT_NAME: self.gen_kw_export_name,
            ConfigKeys.FIELD: self.field,
            ConfigKeys.GEN_DATA: self.gen_data,
            ConfigKeys.MAX_SUBMIT: self.max_submit,
            ConfigKeys.NUM_CPU: 1,
            ConfigKeys.QUEUE_SYSTEM: self.queue_system,
            ConfigKeys.QUEUE_OPTION: self.queue_option,
            ConfigKeys.ANALYSIS_SET_VAR: self.analysis_set_var,
            ConfigKeys.INSTALL_JOB: [
                (
                    name,
                    (
                        fn,
                        pathlib.Path(fn).read_text(encoding="utf-8"),
                    ),
                )
                for name, fn in self.install_job
            ],
            ConfigKeys.INSTALL_JOB_DIRECTORY: self.install_job_directory,
            ConfigKeys.RANDOM_SEED: self.random_seed,
            ConfigKeys.SETENV: self.setenv,
        }
        if self.eclbase is not None:
            result[ConfigKeys.ECLBASE] = self.eclbase
        if self.jobname is not None:
            result[ConfigKeys.JOBNAME] = self.jobname
        return result

    def all_defines(self, config_file, cwd):
        """
        We combine default defines that are magically populated in the config
        content path using the predefined keyword mechanism with random key values.
        When initializing from dict, unlike when reading from file, defines are not
        resolved so we must make sure a define key is not present in any file names
        etc. Therefore all file_names are upper case and define keys are prefixed
        with the lower case 'key'.
        """
        config_file_name = os.path.basename(config_file)
        result = [
            ("<CONFIG_PATH>", cwd),
            ("<CONFIG_FILE_BASE>", config_file_name.split(".")[0]),
            ("<DATE>", datetime.date.today().isoformat()),
            ("<CWD>", cwd),
            ("<CONFIG_FILE>", config_file_name),
        ]
        result.extend(self.define)
        result.extend(
            [
                ("<RUNPATH_FILE>", self.runpath_file),
                ("<NUM_CPU>", str(self.num_cpu)),
            ]
        )
        return result


def summary_keys(smspec: Smspec) -> st.SearchStrategy[str]:
    keys = []
    for index in range(1, len(smspec.keywords)):  # assume index 0 is time
        summary_key = smspec.keywords[index]
        keys.append(
            make_summary_key(
                summary_key,
                smspec.region_numbers[index],
                smspec.well_names[index].rstrip(),
                smspec.nx,
                smspec.ny,
                smspec.lgrs[index].rstrip() if smspec.lgrs else None,
                smspec.numlx[index] if smspec.numlx else None,
                smspec.numly[index] if smspec.numly else None,
                smspec.numlz[index] if smspec.numlz else None,
            )
        )

    return st.sampled_from(keys)


@st.composite
def ert_config_values(draw, use_eclbase=booleans):
    queue_system = draw(queue_systems)
    install_jobs = draw(small_list(random_forward_model_names(words, file_names)))
    forward_model = draw(small_list(job(install_jobs))) if install_jobs else []
    gen_data = draw(
        small_list(
            st.tuples(
                st.builds(lambda x: f"GEN_DATA-{x}", words),
                st.fixed_dictionaries(
                    {
                        "RESULT_FILE": format_result_file_name,
                        "INPUT_FORMAT": st.just("ASCII"),
                        "REPORT_STEPS": report_steps(),
                    }
                ),
            ),
            unique_by=operator.itemgetter(0),
        )
    )
    sum_keys = draw(small_list(summary_variables(), min_size=1))
    first_date = datetime.datetime.strptime("1999-1-1", "%Y-%m-%d")
    smspec = draw(
        smspecs(
            sum_keys=st.just(sum_keys),
            start_date=st.just(Date.from_datetime(first_date)),
            use_days=st.just(True),
        )
    )
    std_cutoff = draw(small_floats)
    obs = draw(
        observations(
            st.sampled_from([g[0] for g in gen_data]) if gen_data else None,
            summary_keys(smspec) if len(smspec.keywords) > 1 else None,
            std_cutoff=std_cutoff,
            start_date=first_date,
        )
    )
    need_eclbase = any(
        isinstance(val, HistoryObservation | SummaryObservation) for val in obs
    )
    use_eclbase = draw(use_eclbase) if not need_eclbase else True
    dates = _observation_dates(obs, first_date)
    time_diffs = [d - first_date for d in dates]
    time_diff_floats = [diff.total_seconds() / (3600 * 24) for diff in time_diffs]
    unsmry = draw(
        unsmrys(
            len(sum_keys),
            report_steps=st.just(list(range(1, len(dates) + 1))),
            mini_steps=st.just(list(range(len(dates) + 1))),
            days=st.just(time_diff_floats),
        )
    )
    return draw(
        st.builds(
            ErtConfigValues,
            forward_model=st.just(forward_model),
            num_realizations=positives,
            eclbase=st.just(draw(words) + "%d") if use_eclbase else st.just(None),
            runpath_file=st.just(draw(file_names) + "runpath"),
            run_template=small_list(
                st.builds(lambda fil: [fil + ".templ", fil], file_names)
            ),
            std_cutoff=st.just(std_cutoff),
            enkf_alpha=small_floats,
            update_log_path=directory_names(),
            max_runtime=positives,
            min_realizations=st.builds(
                (lambda a, b: str(a) if b else str(a) + "%"),
                st.integers(),
                booleans,
            ),
            define=small_list(
                st.tuples(st.builds(lambda x: f"<key-{x}>", words), words)
            ),
            stop_long_running=booleans,
            data_kw_key=small_list(
                st.tuples(st.builds(lambda x: f"<{x}>", words), words)
            ),
            data_file=st.just(draw(file_names) + ".DATA"),
            grid_file=st.just(draw(words) + ".EGRID"),
            job_script=st.just(draw(file_names) + "job_script"),
            jobname=(
                st.just("JOBNAME-" + draw(words)) if not use_eclbase else st.just(None)
            ),
            runpath=st.just("runpath-" + draw(format_runpath_file_name)),
            enspath=st.just(draw(words) + ".enspath"),
            time_map=st.tuples(
                st.builds(lambda fn: fn + ".timemap", file_names),
                st.just("\n".join(dt.date().isoformat() for dt in dates)),
            ),
            obs_config=st.just("obs-config-" + draw(file_names)),
            history_source=st.just(HistorySource.REFCASE_SIMULATED),
            refcase=st.just("refcase/" + draw(file_names)),
            gen_kw_export_name=st.just("gen-kw-export-name-" + draw(file_names)),
            field=small_list(
                st.tuples(
                    st.builds(lambda w: "FIELD-" + w, words),
                    st.just("PARAMETER"),
                    field_output_names(),
                    st.fixed_dictionaries(
                        {
                            "FORWARD_INIT": st.builds(str, booleans),
                            "INIT_TRANSFORM": transforms,
                            "OUTPUT_TRANSFORM": transforms,
                            "MIN": st.builds(str, small_floats),
                            "MAX": st.builds(str, small_floats),
                            "INIT_FILES": file_names,
                        }
                    ),
                ),
                unique_by=operator.itemgetter(0),
            ),
            gen_data=st.just(gen_data),
            max_submit=positives,
            num_cpu=positives,
            queue_system=st.just(queue_system),
            queue_option=small_list(queue_options(st.just(queue_system))),
            analysis_set_var=small_list(
                st.tuples(
                    st.just("STD_ENKF"),
                    st.just("ENKF_TRUNCATION"),
                    st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
                )
            ),
            install_job=st.just(install_jobs),
            install_job_directory=small_list(directory_names()),
            license_path=directory_names(),
            random_seed=st.integers(),
            setenv=small_list(st.tuples(words, words)),
            observations=st.just(obs),
            refcase_smspec=st.just(smspec),
            refcase_unsmry=st.just(unsmry),
            egrid=egrids,
        )
    )


def job(installed_jobs):
    possible_job_names = st.sampled_from([job_name for job_name, _ in installed_jobs])
    args = st.lists(st.tuples(st.builds(lambda arg: f"<{arg}>", words), words))
    return st.builds(lambda name, args: [name, args], possible_job_names, args)


def sim_job(installed_jobs):
    possible_job_names = [job_name for job_name, _ in installed_jobs]
    args = small_list(words)
    x = st.builds(
        lambda job_name, args: [job_name, *args],
        st.sampled_from(possible_job_names),
        args,
    )
    return x


def _observation_dates(
    observations, start_date: datetime.datetime
) -> list[datetime.datetime]:
    """
    :returns: the dates that need to exist in the refcase for ert to accept the
        observations
    """
    dates = list(set([start_date] + [o.get_date(start_date) for o in observations]))
    restart_obs = [
        o for o in observations if hasattr(o, "restart") and o.restart is not None
    ]
    segments = [s for o in observations if hasattr(o, "segment") for s in o.segment]
    restart_indecies = (
        [o.restart for o in restart_obs]
        + [s.start for s in segments]
        + [s.stop for s in segments]
    )
    min_restart = max(restart_indecies) if restart_indecies else 2
    i = 0
    while len(dates) <= max(min_restart, 2):
        dates.append(start_date + datetime.timedelta(days=i))
        dates = list(set(dates))
        i += 1
    return dates


@st.composite
def config_generators(draw, use_eclbase=booleans):
    filterwarnings(
        "ignore",
        message=".*Too small observation error in observation.*",
        category=UserWarning,
    )
    filterwarnings(
        "ignore",
        message=".*MIN_REALIZATIONS set to more than NUM_REALIZATIONS.*",
        category=UserWarning,
    )
    filterwarnings(
        "ignore", message=r".*Segment [^\s]* start after stop.*", category=UserWarning
    )
    filterwarnings(
        "ignore",
        message=r".*Segment [^\s]* does not contain any time steps.*",
        category=UserWarning,
    )
    filterwarnings(
        "ignore", message=".*input contained no data.*", category=UserWarning
    )
    filterwarnings(
        "ignore", message=".*overflow encountered in.*", category=RuntimeWarning
    )
    filterwarnings(
        "ignore",
        message=".*JOB_PREFIX as QUEUE_OPTION to the TORQUE system.*",
        category=UserWarning,
    )
    config_values = draw(ert_config_values(use_eclbase=use_eclbase))

    should_exist_files = [job_path for _, job_path in config_values.install_job]
    should_exist_files.extend(
        [
            config_values.data_file,
            config_values.job_script,
            config_values.time_map[0],
            config_values.obs_config,
        ]
    )
    obs = config_values.observations
    for o in obs:
        if hasattr(o, "obs_file") and o.obs_file is not None:
            should_exist_files.append(o.obs_file)
        if hasattr(o, "index_file") and o.index_file is not None:
            should_exist_files.append(o.index_file)

    should_exist_files.extend([src for src, _ in config_values.run_template])
    should_be_executable_files = [config_values.job_script]
    should_exist_directories = config_values.install_job_directory

    def generate_job_config(job_path, name):
        return job_path, name + ".exe"

    should_exist_job_configs = [
        generate_job_config(job_path, name)
        for name, job_path in config_values.install_job
    ] + [
        generate_job_config(job_dir + "/" + draw(file_names), draw(file_names))
        for job_dir in config_values.install_job_directory
    ]

    # Context manager is returned from the generator and given to test function
    # Run this function to get dict, and as a side effect all required files
    # (not config) will then be written to a temp dir which is returned in a
    # tuple together with the dict as (dict, path). If given a config_file_name
    # will write the config to that file.
    @contextlib.contextmanager
    def generate_files_and_dict(tmp_path_factory, config_file_name=None):
        tmp = py_path.local(tmp_path_factory.mktemp("config_dict"))
        note(f"Using tmp dir: {tmp!s}")
        with tmp.as_cwd():
            config_values.run_template = [
                [os.path.join(tmp, src), target]
                for src, target in config_values.run_template
            ]
            for key in [
                "runpath",
                "runpath_file",
                "enspath",
                "obs_config",
                "data_file",
                "job_script",
            ]:
                setattr(
                    config_values, key, os.path.join(tmp, getattr(config_values, key))
                )
            for dirname in should_exist_directories:
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)

            for filename in should_exist_files:
                if not os.path.isfile(filename):
                    touch(filename)

            with open(config_values.obs_config, "w", encoding="utf-8") as fh:
                for o in obs:
                    fh.write(str(o))
                    fh.write("\n")

            def make_executable(filename):
                current_mode = os.stat(filename).st_mode
                os.chmod(filename, current_mode | stat.S_IEXEC)

            for filename in should_be_executable_files:
                make_executable(filename)

            for job_file, executable_file in should_exist_job_configs:
                path = Path(job_file).parent
                if not os.path.isdir(path / "script"):
                    os.mkdir(path / "script")
                touch(path / "script" / executable_file)
                make_executable(path / "script" / executable_file)
                Path(job_file).write_text(
                    f"EXECUTABLE script/{executable_file}\nMIN_ARG 0\nMAX_ARG 1\n",
                    encoding="utf-8",
                )

            summary_basename = os.path.splitext(
                os.path.basename(config_values.refcase)
            )[0]
            with contextlib.suppress(FileExistsError):
                os.mkdir("./refcase")
            config_values.refcase_smspec.to_file(f"./refcase/{summary_basename}.SMSPEC")
            config_values.refcase_unsmry.to_file(f"./refcase/{summary_basename}.UNSMRY")
            config_values.egrid.to_file(config_values.grid_file)
            with open(config_values.time_map[0], "w", encoding="utf-8") as fh:
                fh.write(config_values.time_map[1])

            if config_file_name is not None:
                to_config_file(config_file_name, config_values)
                assume(config_values.data_file != config_file_name)
                assume(config_values.runpath_file != config_file_name)

            yield config_values

    return generate_files_and_dict


def to_config_file(filename, config_values):
    config_dict = config_values.to_config_dict(filename, os.getcwd(), all_defines=False)
    with open(file=filename, mode="w+", encoding="utf-8") as config:
        config.write(
            f"{ConfigKeys.RUNPATH_FILE} {config_dict[ConfigKeys.RUNPATH_FILE]}\n"
        )
        # keys whose values are lists of tuples of the form (KEY, VALUE)
        tuple_value_keywords = [
            ConfigKeys.SETENV,
            ConfigKeys.RUN_TEMPLATE,
            ConfigKeys.DATA_KW,
            ConfigKeys.DEFINE,
        ]
        for keyword, keyword_value in config_dict.items():
            if keyword in tuple_value_keywords:
                config.writelines(
                    f"{keyword} {tuple_key} {tuple_value}\n"
                    for tuple_key, tuple_value in keyword_value
                )
            elif keyword == ConfigKeys.FORWARD_MODEL:
                config.writelines(
                    f"{keyword} {job_name}"
                    f"({', '.join(f'{a}={b}' for a, b in job_args)})\n"
                    for job_name, job_args in keyword_value
                )
            elif keyword == ConfigKeys.FIELD:
                for line in keyword_value:
                    config.write(keyword)
                    config.write(" ")
                    config.write(line[0])
                    config.write(" ")
                    config.write(line[1])
                    config.write(" ")
                    config.write(line[2])
                    config.write(" ")
                    config.write(" ".join(f"{k}:{v}" for k, v in line[3].items()))
                    config.write("\n")
            elif keyword == ConfigKeys.GEN_DATA:
                for line in keyword_value:
                    config.write(keyword)
                    config.write(" ")
                    config.write(line[0])
                    config.write(" ")
                    config.write(" ".join(f"{k}:{v}" for k, v in line[1].items()))
                    config.write("\n")
            elif keyword == ConfigKeys.INSTALL_JOB_DIRECTORY:
                config.writelines(
                    f"{keyword} {install_dir}\n" for install_dir in keyword_value
                )
            elif keyword == ConfigKeys.ANALYSIS_SET_VAR:
                config.writelines(
                    f"{keyword} {statement[0]} {statement[1]} {statement[2]}\n"
                    for statement in keyword_value
                )
            elif keyword == ConfigKeys.QUEUE_OPTION:
                config.writelines(
                    f"{keyword} {setting[0]} {setting[1]}"
                    + (f" {setting[2]}\n" if len(setting) == 3 else "\n")
                    for setting in keyword_value
                )
            elif keyword in {ConfigKeys.TIME_MAP, ConfigKeys.OBS_CONFIG}:
                config.write(f"{keyword} {keyword_value[0]}\n")
            elif keyword == ConfigKeys.INSTALL_JOB:
                config.writelines(
                    f"{keyword} {fm[0]} {fm[1][0]}\n" for fm in keyword_value
                )
            else:
                config.write(f"{keyword} {keyword_value}\n")
