import contextlib
import datetime
import glob
import os
import os.path
import shutil
import stat
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import assume, note
from py import path as py_path

from ert._c_wrappers.enkf import ConfigKeys
from ert._c_wrappers.enkf.config.enkf_config_node import FIELD_FUNCTION_NAMES
from ert._c_wrappers.job_queue import QueueDriverEnum

from .egrid_generator import egrids

words = st.text(
    min_size=4, max_size=8, alphabet=st.characters(min_codepoint=65, max_codepoint=90)
)


def touch(filename):
    with open(file=filename, mode="w", encoding="utf-8") as fh:
        fh.write(" ")


file_names = words
format_file_names = st.builds(lambda file_name: file_name + "-%d", file_names)


@st.composite
def directory_names(draw):
    return "dir" + draw(words)


@st.composite
def report_steps(draw):
    rep_steps = draw(
        st.lists(st.integers(min_value=0, max_value=100), min_size=4, unique=True)
    )
    return ",".join(str(step) for step in sorted(rep_steps))


transforms = st.sampled_from(FIELD_FUNCTION_NAMES)
small_floats = st.floats(min_value=1.0, max_value=10.0, allow_nan=False)
positives = st.integers(min_value=1, max_value=10000)
queue_systems = st.sampled_from(["LSF", "LOCAL", "TORQUE", "SLURM"])


def valid_queue_options(queue_system):
    valids = [ConfigKeys.MAX_RUNNING]
    if queue_system == QueueDriverEnum.LSF_DRIVER:
        valids += [
            "LSF_RESOURCE",
            "LSF_SERVER",
            "LSF_QUEUE",
            "LSF_LOGIN_SHELL",
            "LSF_RSH_CMD",
            "BSUB_CMD",
            "BJOBS_CMD",
            "BKILL_CMD",
            "BHIST_CMD",
            "BJOBS_TIMEOUT",
            "DEBUG_OUTPUT",
            "EXCLUDE_HOST",
            "PROJECT_CODE",
            "SUBMIT_SLEEP",
        ]
    elif queue_system == QueueDriverEnum.SLURM_DRIVER:
        valids += [
            ConfigKeys.SLURM_SBATCH_OPTION,
            ConfigKeys.SLURM_SCANCEL_OPTION,
            ConfigKeys.SLURM_SCONTROL_OPTION,
            ConfigKeys.SLURM_MEMORY_OPTION,
            ConfigKeys.SLURM_MEMORY_PER_CPU_OPTION,
            ConfigKeys.SLURM_EXCLUDE_HOST_OPTION,
            ConfigKeys.SLURM_INCLUDE_HOST_OPTION,
            ConfigKeys.SLURM_SQUEUE_TIMEOUT_OPTION,
            ConfigKeys.SLURM_MAX_RUNTIME_OPTION,
        ]
    elif queue_system == QueueDriverEnum.TORQUE_DRIVER:
        valids += [
            "QSUB_CMD",
            "QSTAT_CMD",
            "QDEL_CMD",
            "QUEUE",
            "NUM_CPUS_PER_NODE",
            "NUM_NODES",
            "KEEP_QSUB_OUTPUT",
            "CLUSTER_LABEL",
            "JOB_PREFIX",
            "DEBUG_OUTPUT",
            "SUBMIT_SLEEP",
        ]
    return valids


def valid_queue_values(option_name):
    if option_name in [
        "QSUB_CMD",
        "QSTAT_CMD",
        "QDEL_CMD",
        "QUEUE",
        "CLUSTER_LABEL",
        "JOB_PREFIX",
        "DEBUG_OUTPUT",
        "LSF_RESOURCE",
        "LSF_SERVER",
        "LSF_QUEUE",
        "LSF_LOGIN_SHELL",
        "LSF_RSH_CMD",
        "BSUB_CMD",
        "BJOBS_CMD",
        "BKILL_CMD",
        "BHIST_CMD",
        "BJOBS_TIMEOUT",
        "EXCLUDE_HOST",
        "PROJECT_CODE",
        ConfigKeys.SLURM_SBATCH_OPTION,
        ConfigKeys.SLURM_SCANCEL_OPTION,
        ConfigKeys.SLURM_SCONTROL_OPTION,
        ConfigKeys.SLURM_MEMORY_OPTION,
        ConfigKeys.SLURM_MEMORY_PER_CPU_OPTION,
        ConfigKeys.SLURM_EXCLUDE_HOST_OPTION,
        ConfigKeys.SLURM_INCLUDE_HOST_OPTION,
    ]:
        return words
    if option_name in [
        "SUBMIT_SLEEP",
        ConfigKeys.SLURM_SQUEUE_TIMEOUT_OPTION,
    ]:
        return st.builds(str, small_floats)
    if option_name in [
        "NUM_CPUS_PER_NODE",
        "NUM_NODES",
        "MAX_RUNNING",
        ConfigKeys.SLURM_MAX_RUNTIME_OPTION,
    ]:
        return st.builds(str, positives)
    if option_name in [
        "KEEP_QSUB_OUTPUT",
        "DEBUG_OUTPUT",
    ]:
        return st.builds(str, st.booleans())
    else:
        raise ValueError(f"unknown option {option_name}")


@st.composite
def queue_options(draw, systems):
    queue_system = draw(systems)
    name = draw(st.sampled_from(valid_queue_options(queue_system)))
    do_set = draw(st.booleans())
    if do_set:
        return [queue_system, name, draw(valid_queue_values(name))]
    else:
        # Missing VALUE means unset
        return [queue_system, name]


def default_ext_job_names():
    return (
        st.sampled_from(
            [
                "DELETE_FILE",
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
def random_ext_job_names(draw, some_words, some_file_names):
    return draw(
        st.tuples(
            some_words,
            st.just(draw(some_file_names) + "job_config"),
        )
    )


def defines(config_dict, config_files, cwd):
    """
    We combine default defines that are magically populated in the config
    content path using the predefined keyword mechanism with random key values.
    When initializing from dict, unlike when reading from file, defines are not
    resolved so we must make sure a define key is not present in any file names
    etc. Therefore all file_names are upper case and define keys are prefixed
    with the lower case 'key'.
    """
    config_file_name = os.path.basename(config_files)
    return [
        ("<CONFIG_PATH>", cwd),
        ("<CONFIG_FILE_BASE>", config_file_name.split(".")[0]),
        ("<DATE>", datetime.date.today().isoformat()),
        ("<CWD>", cwd),
        ("<CONFIG_FILE>", config_file_name),
        ("<RUNPATH_FILE>", os.path.join(cwd, config_dict[ConfigKeys.RUNPATH_FILE])),
        ("<NUM_CPU>", str(config_dict[ConfigKeys.NUM_CPU])),
    ]


def small_list(*arg, max_size=5, **kw_args):
    return st.lists(*arg, **kw_args, max_size=max_size)


def generate_config(draw):
    queue_system = draw(queue_systems)
    config_dict = draw(
        st.fixed_dictionaries(
            {
                ConfigKeys.NUM_REALIZATIONS: positives,
                ConfigKeys.ECLBASE: st.just(draw(words) + "%d"),
                ConfigKeys.RUNPATH_FILE: st.just(draw(file_names) + "runpath"),
                ConfigKeys.RUN_TEMPLATE: small_list(
                    st.builds(lambda fil: [fil + ".templ", fil], file_names)
                ),
                ConfigKeys.ALPHA_KEY: small_floats,
                ConfigKeys.ITER_CASE: words,
                ConfigKeys.ITER_COUNT: positives,
                ConfigKeys.ITER_RETRY_COUNT: positives,
                ConfigKeys.UPDATE_LOG_PATH: directory_names(),
                ConfigKeys.STD_CUTOFF_KEY: small_floats,
                ConfigKeys.MAX_RUNTIME: positives,
                ConfigKeys.MIN_REALIZATIONS: positives,
                ConfigKeys.DEFINE_KEY: small_list(
                    st.tuples(st.just(f"<key-{draw(words)}>"), words)
                ),
                ConfigKeys.STOP_LONG_RUNNING: st.booleans(),
                ConfigKeys.DATA_KW_KEY: small_list(
                    st.tuples(st.just(f"<{draw(words)}>"), words)
                ),
                ConfigKeys.DATA_FILE: st.just(draw(file_names) + ".DATA"),
                ConfigKeys.GRID: st.just(draw(words) + ".EGRID"),
                ConfigKeys.JOB_SCRIPT: st.just(draw(file_names) + "job_script"),
                ConfigKeys.JOBNAME: st.just("JOBNAME-" + draw(words)),
                ConfigKeys.RUNPATH: st.just("runpath-" + draw(format_file_names)),
                ConfigKeys.ENSPATH: st.just(draw(words) + ".enspath"),
                ConfigKeys.TIME_MAP: st.just(draw(file_names) + ".timemap"),
                ConfigKeys.OBS_CONFIG: st.just("obs-config-" + draw(file_names)),
                ConfigKeys.HISTORY_SOURCE: st.just("REFCASE_SIMULATED"),
                ConfigKeys.REFCASE: st.just("refcase/" + draw(file_names)),
                ConfigKeys.GEN_KW_EXPORT_NAME: st.just(
                    "gen-kw-export-name-" + draw(file_names)
                ),
                ConfigKeys.FIELD_KEY: small_list(
                    st.tuples(
                        st.just("FIELD-" + draw(words)),
                        st.just("PARAMETER"),
                        file_names,
                        st.just(f"FORWARD_INIT:{draw(st.booleans())}"),
                        st.just(f"INIT_TRANSFORM:{draw(transforms)}"),
                        st.just(f"OUTPUT_TRANSFORM:{draw(transforms)}"),
                        st.just(f"MIN:{draw(small_floats)}"),
                        st.just(f"MAX:{draw(small_floats)}"),
                        st.just(f"INIT_FILES:{draw(file_names)}"),
                    ),
                    unique_by=lambda element: element[0],
                ),
                ConfigKeys.GEN_DATA: small_list(
                    st.tuples(
                        st.just(f"GEN_DATA-{draw(words)}"),
                        st.just(f"{ConfigKeys.RESULT_FILE}:{draw(format_file_names)}"),
                        st.just(f"{ConfigKeys.INPUT_FORMAT}:ASCII"),
                        st.just(f"{ConfigKeys.REPORT_STEPS}:{draw(report_steps())}"),
                    ),
                    unique_by=lambda tup: tup[0],
                ),
                ConfigKeys.MAX_SUBMIT: positives,
                ConfigKeys.NUM_CPU: positives,
                ConfigKeys.QUEUE_SYSTEM: st.just(queue_system),
                ConfigKeys.QUEUE_OPTION: small_list(
                    queue_options(st.just(queue_system))
                ),
                ConfigKeys.ANALYSIS_COPY: small_list(
                    st.tuples(
                        st.just("STD_ENKF"),
                        words,
                    )
                ),
                ConfigKeys.ANALYSIS_SET_VAR: small_list(
                    st.tuples(
                        st.just("STD_ENKF"),
                        st.just("ENKF_NCOMP"),
                        st.integers(),
                    )
                ),
                ConfigKeys.ANALYSIS_SELECT: st.sampled_from(["STD_ENKF", "IES_ENKF"]),
                ConfigKeys.INSTALL_JOB: small_list(
                    random_ext_job_names(words, file_names)
                ),
                ConfigKeys.INSTALL_JOB_DIRECTORY: small_list(directory_names()),
                ConfigKeys.LICENSE_PATH: directory_names(),
                ConfigKeys.RANDOM_SEED: words,
                ConfigKeys.SETENV: small_list(st.tuples(words, words)),
            }
        )
    )

    installed_jobs = config_dict[ConfigKeys.INSTALL_JOB]
    config_dict[ConfigKeys.FORWARD_MODEL] = (
        draw(small_list(job(installed_jobs))) if installed_jobs else []
    )
    config_dict[ConfigKeys.SIMULATION_JOB] = (
        draw(small_list(sim_job(installed_jobs))) if installed_jobs else []
    )
    return config_dict


def job(installed_jobs):
    possible_job_names = st.sampled_from([job_name for job_name, _ in installed_jobs])
    arg = st.builds(lambda arg, value: f"<{arg}>={value}", words, words)
    args = st.builds(",".join, small_list(arg))
    return st.builds(lambda name, args: [name, args], possible_job_names, args)


def sim_job(installed_jobs):
    possible_job_names = [job_name for job_name, _ in installed_jobs]
    args = small_list(st.builds(lambda arg, value: f"<{arg}>={value}", words, words))
    x = st.builds(
        lambda job_name, args: [job_name] + (args),
        st.sampled_from(possible_job_names),
        args,
    )
    return x


@st.composite
def config_generators(draw):
    config_dict = generate_config(draw)

    should_exist_files = [
        job_path for _, job_path in config_dict[ConfigKeys.INSTALL_JOB]
    ]
    should_exist_files.extend(
        [
            config_dict[ConfigKeys.DATA_FILE],
            config_dict[ConfigKeys.JOB_SCRIPT],
            config_dict[ConfigKeys.TIME_MAP],
            config_dict[ConfigKeys.OBS_CONFIG],
        ]
    )
    should_exist_files.extend(
        [src for src, target in config_dict[ConfigKeys.RUN_TEMPLATE]]
    )

    should_be_executable_files = [config_dict[ConfigKeys.JOB_SCRIPT]]

    should_exist_directories = config_dict[ConfigKeys.INSTALL_JOB_DIRECTORY]

    def generate_job_config(job_path):
        return job_path, draw(file_names) + ".exe"

    should_exist_job_configs = [
        generate_job_config(job_path)
        for _, job_path in config_dict[ConfigKeys.INSTALL_JOB]
    ] + [
        generate_job_config(job_dir + "/" + draw(file_names))
        for job_dir in config_dict[ConfigKeys.INSTALL_JOB_DIRECTORY]
    ]

    should_exist_refcase = (
        config_dict[ConfigKeys.REFCASE] if ConfigKeys.REFCASE in config_dict else None
    )

    egrid = draw(egrids)

    # Context manager is returned from the generator and given to test function
    # Run this function to get dict, and as a side effect all required files
    # (not config) will then be written to a temp dir which is returned in a
    # tuple together with the dict as (dict, path). If given a config_file_name
    # will write the config to that file.
    @contextlib.contextmanager
    def generate_files_and_dict(tmp_path_factory, config_file_name=None):
        tmp = py_path.local(tmp_path_factory.mktemp("config_dict"))
        note(f"Using tmp dir: {str(tmp)}")
        with tmp.as_cwd():
            config_dict[ConfigKeys.JOB_SCRIPT] = os.path.abspath(
                config_dict[ConfigKeys.JOB_SCRIPT]
            )

            for dirname in should_exist_directories:
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)

            for filename in should_exist_files:
                if not os.path.isfile(filename):
                    touch(filename)

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

            if should_exist_refcase is not None:
                dest_basename = os.path.splitext(
                    os.path.basename(should_exist_refcase)
                )[0]
                refcase_src_files_glob = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../test-data/configuration_tests/input/"
                    "refcase/SNAKE_OIL_FIELD*",
                )
                with contextlib.suppress(FileExistsError):
                    os.mkdir("./refcase")
                refcase_src_files = glob.glob(refcase_src_files_glob)
                for refcase_src_file in refcase_src_files:
                    dest = os.path.basename(refcase_src_file).replace(
                        "SNAKE_OIL_FIELD", dest_basename
                    )
                    shutil.copy(refcase_src_file, "./refcase/" + dest)

            egrid.to_file(config_dict[ConfigKeys.GRID])

            # Make all paths absolute, as that should be the result after parsing
            cwd = os.getcwd()
            keys_that_should_be_absolute = [
                ConfigKeys.RUNPATH,
                ConfigKeys.RUNPATH_FILE,
                ConfigKeys.ENSPATH,
                ConfigKeys.TIME_MAP,
                ConfigKeys.OBS_CONFIG,
                ConfigKeys.DATA_FILE,
            ]
            for key in keys_that_should_be_absolute:
                config_dict[key] = os.path.join(cwd, config_dict[key])

            def make_run_template_abs(template):
                src, target = template
                return [os.path.join(cwd, src), target]

            config_dict[ConfigKeys.RUN_TEMPLATE] = list(
                map(
                    make_run_template_abs,
                    config_dict.get(ConfigKeys.RUN_TEMPLATE, []),
                )
            )

            if config_file_name is not None:
                to_config_file(config_file_name, config_dict)
                assume(config_dict[ConfigKeys.DATA_FILE] != config_file_name)
                assume(config_dict[ConfigKeys.RUNPATH_FILE] != config_file_name)

            yield config_dict

    return generate_files_and_dict


def to_config_file(filename, config_dict):  # pylint: disable=too-many-branches
    predefines = defines(config_dict, filename, os.getcwd())
    predefines.extend(config_dict[ConfigKeys.DEFINE_KEY])
    config_dict[ConfigKeys.DEFINE_KEY] = predefines

    with open(file=filename, mode="w+", encoding="utf-8") as config:
        config.write(
            f"{ConfigKeys.RUNPATH_FILE} {config_dict[ConfigKeys.RUNPATH_FILE]}\n"
        )
        # keys whose values are lists of tuples of the form (KEY, VALUE)
        tuple_value_keywords = [
            ConfigKeys.SETENV,
            ConfigKeys.RUN_TEMPLATE,
            ConfigKeys.DATA_KW_KEY,
            ConfigKeys.DEFINE_KEY,
            ConfigKeys.INSTALL_JOB,
        ]
        for keyword, keyword_value in config_dict.items():
            if keyword in tuple_value_keywords:
                for tuple_key, tuple_value in keyword_value:
                    config.write(f"{keyword} {tuple_key} {tuple_value}\n")
            elif keyword == ConfigKeys.SIMULATION_JOB:
                for job_config in keyword_value:
                    job_name = job_config[0]
                    job_args = " ".join(job_config[1:])
                    config.write(f"{keyword} {job_name} {job_args}\n")
            elif keyword == ConfigKeys.FORWARD_MODEL:
                for job_name, job_args in keyword_value:
                    config.write(f"{keyword} {job_name}({job_args})\n")
            elif keyword == ConfigKeys.FIELD_KEY:
                # keyword_value is a list of dicts, each defining a field
                for field_vals in keyword_value:
                    config.write(" ".join([keyword, *field_vals]) + "\n")
            elif keyword == ConfigKeys.GEN_DATA:
                # keyword_value is a list of dicts, each defining a field
                for gen_data_entry in keyword_value:
                    config.write(" ".join([keyword, *gen_data_entry]) + "\n")
            elif keyword == ConfigKeys.INSTALL_JOB_DIRECTORY:
                for install_dir in keyword_value:
                    config.write(f"{keyword} {install_dir}\n")
            elif keyword == ConfigKeys.ANALYSIS_COPY:
                for statement in keyword_value:
                    config.write(f"{keyword} {statement[0]} {statement[1]}\n")
            elif keyword == ConfigKeys.ANALYSIS_SET_VAR:
                for statement in keyword_value:
                    config.write(
                        f"{keyword} {statement[0]} {statement[1]} {statement[2]}\n"
                    )
            elif keyword == ConfigKeys.QUEUE_OPTION:
                for setting in keyword_value:
                    config.write(
                        f"{keyword} {setting[0]} {setting[1]}"
                        + (f" {setting[2]}\n" if len(setting) == 3 else "\n")
                    )
            else:
                config.write(f"{keyword} {keyword_value}\n")
