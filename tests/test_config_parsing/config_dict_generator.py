import contextlib
import datetime
import glob
import os
import os.path
import shutil
import stat
from pathlib import Path

import hypothesis.strategies as st
import py
from hypothesis import assume, note

from ert._c_wrappers.enkf import ConfigKeys
from ert._c_wrappers.enkf.enums import GenDataFileType
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


transforms = st.sampled_from(
    [
        "DENORMALIZE_PERMX",
        "NORMALIZE_PORO",
        "LN",
        "LOG10",
        "EXP0",
        "LOG",
        "EXP",
        "TRUNC_POW10",
        "LN0",
        "DENORMALIZE_PORO",
        "NORMALIZE_PERMZ",
        "NORMALIZE_PERMX",
        "DENORMALIZE_PERMZ",
        "POW10",
    ]
)


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


def generate_config(draw):
    queue_system = draw(queue_systems)
    config_dict = draw(
        st.fixed_dictionaries(
            {
                ConfigKeys.NUM_REALIZATIONS: positives,
                ConfigKeys.ECLBASE: st.just(draw(words) + "%d"),
                ConfigKeys.RUNPATH_FILE: st.just(draw(file_names) + "runpath"),
                ConfigKeys.ALPHA_KEY: small_floats,
                ConfigKeys.ITER_CASE: words,
                ConfigKeys.ITER_COUNT: positives,
                ConfigKeys.ITER_RETRY_COUNT: positives,
                ConfigKeys.RERUN_KEY: st.booleans(),
                ConfigKeys.RERUN_START_KEY: positives,
                ConfigKeys.UPDATE_LOG_PATH: directory_names(),
                ConfigKeys.STD_CUTOFF_KEY: small_floats,
                ConfigKeys.MAX_RUNTIME: positives,
                ConfigKeys.MIN_REALIZATIONS: positives,
                ConfigKeys.DEFINE_KEY: st.lists(
                    st.tuples(st.just(f"<key-{draw(words)}>"), words)
                ),
                ConfigKeys.DATA_KW_KEY: st.lists(
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
                ConfigKeys.DATAROOT: st.just("dataroot"),
                ConfigKeys.HISTORY_SOURCE: st.just("REFCASE_SIMULATED"),
                ConfigKeys.REFCASE: st.just("refcase/" + draw(file_names)),
                ConfigKeys.GEN_KW_EXPORT_NAME: st.just(
                    "gen-kw-export-name-" + draw(file_names)
                ),
                ConfigKeys.FIELD_KEY: st.lists(
                    st.fixed_dictionaries(
                        {
                            ConfigKeys.NAME: st.just("FIELD-" + draw(words)),
                            ConfigKeys.VAR_TYPE: st.just("PARAMETER"),
                            ConfigKeys.OUT_FILE: file_names,
                            # ConfigKeys.ENKF_INFILE: file_names, only used in general
                            ConfigKeys.FORWARD_INIT: st.booleans(),
                            ConfigKeys.INIT_TRANSFORM: transforms,
                            ConfigKeys.OUTPUT_TRANSFORM: transforms,
                            # ConfigKeys.INPUT_TRANSFORM: func, only used in general
                            ConfigKeys.MIN_KEY: small_floats,
                            ConfigKeys.MAX_KEY: small_floats,
                            ConfigKeys.INIT_FILES: file_names,
                        }
                    ),
                    unique_by=lambda field_dict: field_dict[ConfigKeys.NAME],
                ),
                ConfigKeys.GEN_DATA: st.lists(
                    st.fixed_dictionaries(
                        {
                            ConfigKeys.NAME: st.just("GEN_DATA-" + draw(words)),
                            ConfigKeys.RESULT_FILE: format_file_names,
                            ConfigKeys.INPUT_FORMAT: st.just(GenDataFileType.ASCII),
                            ConfigKeys.REPORT_STEPS: st.lists(
                                st.integers(min_value=0, max_value=100),
                                min_size=2,
                                unique=True,
                            ),
                        }
                    ),
                    unique_by=lambda field_dict: field_dict[ConfigKeys.NAME],
                ),
                ConfigKeys.MAX_SUBMIT: positives,
                ConfigKeys.NUM_CPU: positives,
                ConfigKeys.QUEUE_SYSTEM: st.just(queue_system),
                ConfigKeys.QUEUE_OPTION: st.lists(queue_options(st.just(queue_system))),
                ConfigKeys.ANALYSIS_COPY: st.lists(
                    st.fixed_dictionaries(
                        {
                            ConfigKeys.SRC_NAME: st.just("STD_ENKF"),
                            ConfigKeys.DST_NAME: words,
                        }
                    )
                ),
                ConfigKeys.ANALYSIS_SET_VAR: st.lists(
                    st.fixed_dictionaries(
                        {
                            ConfigKeys.MODULE_NAME: st.just("STD_ENKF"),
                            ConfigKeys.VAR_NAME: st.just("ENKF_NCOMP"),
                            ConfigKeys.VALUE: st.integers(),
                        }
                    )
                ),
                ConfigKeys.ANALYSIS_SELECT: st.just("STD_ENKF"),
                ConfigKeys.INSTALL_JOB: st.lists(
                    random_ext_job_names(words, file_names)
                ),
                ConfigKeys.INSTALL_JOB_DIRECTORY: st.lists(directory_names()),
                ConfigKeys.LICENSE_PATH: directory_names(),
                ConfigKeys.RANDOM_SEED: words,
                ConfigKeys.SETENV: st.lists(st.tuples(words, words)),
            }
        )
    )

    config_dict[ConfigKeys.FORWARD_MODEL] = draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    ConfigKeys.NAME: st.sampled_from(
                        [
                            job_name
                            for job_name, _ in config_dict[ConfigKeys.INSTALL_JOB]
                        ]
                    ),
                    ConfigKeys.ARGLIST: st.just(
                        ",".join(
                            draw(st.lists(st.just(f"<{draw(words)}>={draw(words)}")))
                        )
                    ),
                }
            ),
        )
        if config_dict[ConfigKeys.INSTALL_JOB]
        else st.just([])
    )
    return config_dict


@st.composite
def config_generators(draw):
    config_dict = generate_config(draw)

    should_exist_files = [
        job_path for _, job_path in config_dict[ConfigKeys.INSTALL_JOB]
    ]
    should_exist_files.append(config_dict[ConfigKeys.DATA_FILE])
    should_exist_files.append(config_dict[ConfigKeys.JOB_SCRIPT])
    should_exist_files.append(config_dict[ConfigKeys.TIME_MAP])
    should_exist_files.append(config_dict[ConfigKeys.OBS_CONFIG])

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
        tmp = py.path.local(tmp_path_factory.mktemp("config_dict"))
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
            config_dict[ConfigKeys.RUNPATH] = os.path.join(
                cwd, config_dict[ConfigKeys.RUNPATH]
            )
            config_dict[ConfigKeys.ENSPATH] = os.path.join(
                cwd, config_dict[ConfigKeys.ENSPATH]
            )
            config_dict[ConfigKeys.TIME_MAP] = os.path.join(
                cwd, config_dict[ConfigKeys.TIME_MAP]
            )
            config_dict[ConfigKeys.OBS_CONFIG] = os.path.join(
                cwd, config_dict[ConfigKeys.OBS_CONFIG]
            )
            config_dict[ConfigKeys.DATAROOT] = os.path.join(
                cwd, config_dict[ConfigKeys.DATAROOT]
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
        for keyword, keyword_value in config_dict.items():
            if keyword == ConfigKeys.DATA_KW_KEY:
                for define_key, define_value in keyword_value:
                    config.write(f"{keyword} {define_key} {define_value}\n")
            elif keyword == ConfigKeys.FORWARD_MODEL:
                for forward_model_job in keyword_value:
                    job_name = forward_model_job[ConfigKeys.NAME]
                    job_args = forward_model_job[ConfigKeys.ARGLIST]
                    config.write(f"{keyword} {job_name}({job_args})\n")
            elif keyword == ConfigKeys.FIELD_KEY:
                # keyword_value is a list of dicts, each defining a field
                for field_dict in keyword_value:
                    config.write(
                        " ".join(
                            [
                                keyword,
                                field_dict.get(ConfigKeys.NAME, ""),
                                field_dict.get(ConfigKeys.VAR_TYPE, ""),
                                field_dict.get(ConfigKeys.OUT_FILE, ""),
                                f"INIT_FILES:{field_dict.get(ConfigKeys.INIT_FILES)}"
                                if ConfigKeys.INIT_FILES in field_dict
                                else "",
                                f"MIN:{field_dict.get(ConfigKeys.MIN_KEY)}"
                                if ConfigKeys.MIN_KEY in field_dict
                                else "",
                                f"MAX:{field_dict.get(ConfigKeys.MAX_KEY)}"
                                if ConfigKeys.MAX_KEY in field_dict
                                else "",
                                (
                                    "OUTPUT_TRANSFORM:"
                                    f"{field_dict.get(ConfigKeys.OUTPUT_TRANSFORM)}"
                                )
                                if ConfigKeys.OUTPUT_TRANSFORM in field_dict
                                else "",
                                (
                                    "INIT_TRANSFORM:"
                                    f"{field_dict.get(ConfigKeys.INIT_TRANSFORM)}"
                                )
                                if ConfigKeys.INIT_TRANSFORM in field_dict
                                else "",
                                (
                                    "FORWARD_INIT:"
                                    f"{field_dict.get(ConfigKeys.FORWARD_INIT)}"
                                )
                                if ConfigKeys.FORWARD_INIT in field_dict
                                else "",
                                field_dict.get(ConfigKeys.ENKF_INFILE, ""),
                                field_dict.get(ConfigKeys.INPUT_TRANSFORM, ""),
                            ]
                        )
                        + "\n"
                    )
            elif keyword == ConfigKeys.GEN_DATA:
                # keyword_value is a list of dicts, each defining a field
                for field_dict in keyword_value:
                    report_steps_as_string = ",".join(
                        map(str, field_dict.get(ConfigKeys.REPORT_STEPS))
                    )
                    config.write(
                        " ".join(
                            [
                                keyword,
                                field_dict.get(ConfigKeys.NAME, ""),
                                f"RESULT_FILE:{field_dict.get(ConfigKeys.RESULT_FILE)}",
                                (
                                    "INPUT_FORMAT:"
                                    f"{field_dict.get(ConfigKeys.INPUT_FORMAT)}"
                                ),
                                f"REPORT_STEPS:{report_steps_as_string}",
                            ]
                        )
                        + "\n"
                    )
            elif keyword == ConfigKeys.INSTALL_JOB_DIRECTORY:
                for install_dir in keyword_value:
                    config.write(f"{keyword} {install_dir}\n")
            elif keyword == ConfigKeys.SETENV:
                for key, value in keyword_value:
                    config.write(f"{keyword} {key} {value}\n")
            elif keyword == ConfigKeys.ANALYSIS_COPY:
                for statement in keyword_value:
                    config.write(
                        f"{keyword}"
                        f" {statement[ConfigKeys.SRC_NAME]}"
                        f" {statement[ConfigKeys.DST_NAME]}\n"
                    )
            elif keyword == ConfigKeys.ANALYSIS_SET_VAR:
                for statement in keyword_value:
                    config.write(
                        f"{keyword} {statement[ConfigKeys.MODULE_NAME]}"
                        f" {statement[ConfigKeys.VAR_NAME]}"
                        f" {statement[ConfigKeys.VALUE]}\n"
                    )
            elif keyword == ConfigKeys.DEFINE_KEY:
                for define_key, define_value in keyword_value:
                    config.write(f"{keyword} {define_key} {define_value}\n")
            elif keyword == ConfigKeys.INSTALL_JOB:
                for job_name, job_path in keyword_value:
                    config.write(f"{keyword}" f" {job_name} {job_path}\n")
            elif keyword == ConfigKeys.QUEUE_OPTION:
                for setting in keyword_value:
                    config.write(
                        f"{keyword} {setting[0]} {setting[1]}"
                        + (f" {setting[2]}\n" if len(setting) == 3 else "\n")
                    )
            elif keyword == ConfigKeys.QUEUE_SYSTEM:
                config.write(f"{keyword}" f" {keyword_value}\n")
            else:
                config.write(f"{keyword} {keyword_value}\n")
