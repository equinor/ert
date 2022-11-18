import contextlib
import datetime
import glob
import os
import os.path
import shutil
import stat
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import assume

from ert._c_wrappers.enkf import ConfigKeys
from ert._c_wrappers.enkf.enums import GenDataFileType
from ert._c_wrappers.job_queue import QueueDriverEnum
from ert._c_wrappers.sched import HistorySourceEnum

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

queue_systems = st.sampled_from(
    [
        QueueDriverEnum.LSF_DRIVER,
        QueueDriverEnum.LOCAL_DRIVER,
        QueueDriverEnum.TORQUE_DRIVER,
        QueueDriverEnum.SLURM_DRIVER,
    ]
)


def queue_name(queue_driver):
    if queue_driver == QueueDriverEnum.LSF_DRIVER:
        return "LSF"
    if queue_driver == QueueDriverEnum.LOCAL_DRIVER:
        return "LOCAL"
    if queue_driver == QueueDriverEnum.TORQUE_DRIVER:
        return "TORQUE"
    if queue_driver == QueueDriverEnum.SLURM_DRIVER:
        return "SLURM"
    raise ValueError(f"unexpected driver `{queue_driver}`")


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
        return {
            ConfigKeys.DRIVER_NAME: queue_system,
            ConfigKeys.OPTION: name,
            ConfigKeys.VALUE: draw(valid_queue_values(name)),
        }
    else:
        # Missing VALUE means unset
        return {
            ConfigKeys.DRIVER_NAME: queue_system,
            ConfigKeys.OPTION: name,
        }


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


@st.composite
def defines(draw, config_files, cwds):
    """
    We combine default defines that are magically populated in the config
    content path using the predefined keyword mechanism with random key values.
    When initializing from dict, unlike when reading from file, defines are not
    resolved so we must make sure a define key is not present in any file names
    etc. Therefore all file_names are upper case and define keys are prefixed
    with the lower case 'key'.
    """
    config_file_name = os.path.basename(draw(config_files))
    pre_defined_kw_map = draw(
        st.fixed_dictionaries(
            {
                "<DATE>": st.just(datetime.date.today().isoformat()),
                "<CWD>": cwds,
                "<CONFIG_PATH>": cwds,
                "<CONFIG_FILE>": st.just(config_file_name),
                "<CONFIG_FILE_BASE>": st.just(config_file_name.split(".")[0]),
            }
        )
    )
    random_defines = draw(st.dictionaries(st.just("key-" + draw(words)), words))
    return {**random_defines, **pre_defined_kw_map}


@st.composite
def config_dicts(draw):
    queue_system = draw(queue_systems)
    config_file_name = st.just(draw(file_names) + ".ert")
    cwd = os.getcwd()
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
                ConfigKeys.DEFINE_KEY: defines(config_file_name, st.just(cwd)),
                ConfigKeys.DATA_KW_KEY: st.dictionaries(words, words),
                ConfigKeys.DATA_FILE: st.just(draw(file_names) + ".DATA"),
                ConfigKeys.GRID: st.just(draw(words) + ".EGRID"),
                ConfigKeys.JOB_SCRIPT: st.just(draw(file_names) + "job_script"),
                ConfigKeys.MAX_RESAMPLE: positives,
                ConfigKeys.JOBNAME: st.just("JOBNAME-" + draw(words)),
                ConfigKeys.RUNPATH: st.just("runpath-" + draw(format_file_names)),
                ConfigKeys.ENSPATH: st.just(draw(words) + ".enspath"),
                ConfigKeys.TIME_MAP: st.just(draw(file_names) + ".timemap"),
                ConfigKeys.OBS_CONFIG: st.just("obs-config-" + draw(file_names)),
                ConfigKeys.DATAROOT: st.just("."),
                ConfigKeys.HISTORY_SOURCE: st.just(
                    HistorySourceEnum.from_string("REFCASE_SIMULATED")
                ),
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
                ConfigKeys.SETENV: st.lists(
                    st.fixed_dictionaries(
                        {ConfigKeys.NAME: words, ConfigKeys.VALUE: words}
                    )
                ),
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

    should_exist_files = [
        job_path for _, job_path in config_dict[ConfigKeys.INSTALL_JOB]
    ]
    should_exist_files.append(config_dict[ConfigKeys.DATA_FILE])
    should_exist_files.append(config_dict[ConfigKeys.JOB_SCRIPT])
    should_exist_files.append(config_dict[ConfigKeys.TIME_MAP])
    should_exist_files.append(config_dict[ConfigKeys.OBS_CONFIG])

    should_be_executable_files = [
        job_path for _, job_path in config_dict[ConfigKeys.INSTALL_JOB]
    ]
    should_be_executable_files.append(config_dict[ConfigKeys.JOB_SCRIPT])

    config_dict[ConfigKeys.JOB_SCRIPT] = os.path.abspath(
        config_dict[ConfigKeys.JOB_SCRIPT]
    )

    for filename in should_exist_files:
        if not os.path.isfile(filename):
            touch(filename)

    if ConfigKeys.REFCASE in config_dict:
        dest_basename = os.path.splitext(
            os.path.basename(config_dict[ConfigKeys.REFCASE])
        )[0]
        refcase_src_files_glob = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../test-data/configuration_tests/input/refcase/SNAKE_OIL_FIELD*",
        )
        with contextlib.suppress(FileExistsError):
            os.mkdir("./refcase")
        refcase_src_files = glob.glob(refcase_src_files_glob)
        for refcase_src_file in refcase_src_files:
            dest = os.path.basename(refcase_src_file).replace(
                "SNAKE_OIL_FIELD", dest_basename
            )
            shutil.copy(refcase_src_file, "./refcase/" + dest)

    should_be_executable_files = [config_dict[ConfigKeys.JOB_SCRIPT]]

    if (
        len(config_dict[ConfigKeys.INSTALL_JOB]) == 0
        and os.getenv("ERT_SITE_CONFIG", None) is not None
    ):
        assume(len(config_dict[ConfigKeys.FORWARD_MODEL]) == 0)

    for _, job_path in config_dict[ConfigKeys.INSTALL_JOB]:
        executable_file = draw(file_names) + ".exe"
        touch(executable_file)
        should_be_executable_files.append(executable_file)
        Path(job_path).write_text(
            f"EXECUTABLE {executable_file}\nMIN_ARG 0\nMAX_ARG 1\n", encoding="utf-8"
        )

    should_exist_directories = config_dict[ConfigKeys.INSTALL_JOB_DIRECTORY]
    for dirname in should_exist_directories:
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

    for filename in should_be_executable_files:
        current_mode = os.stat(filename).st_mode
        os.chmod(filename, current_mode | stat.S_IEXEC)

    draw(egrids).to_file(config_dict[ConfigKeys.GRID])

    assume(config_dict[ConfigKeys.DATA_FILE] != config_file_name)
    assume(config_dict[ConfigKeys.RUNPATH_FILE] != config_file_name)

    return config_dict


def to_config_file(filename, config_dict):  # pylint: disable=too-many-branches
    with open(file=filename, mode="w+", encoding="utf-8") as config:
        config.write(
            f"{ConfigKeys.RUNPATH_FILE} {config_dict[ConfigKeys.RUNPATH_FILE]}\n"
        )
        for keyword, keyword_value in config_dict.items():
            if keyword == ConfigKeys.DATA_KW_KEY:
                for define_key, define_value in keyword_value.items():
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
                for statement in keyword_value:
                    config.write(
                        f"{keyword}"
                        f" {statement[ConfigKeys.NAME]} {statement[ConfigKeys.VALUE]}\n"
                    )
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
                for define_key, define_value in keyword_value.items():
                    config.write(f"{keyword} {define_key} {define_value}\n")
            elif keyword == ConfigKeys.INSTALL_JOB:
                for job_name, job_path in keyword_value:
                    config.write(f"{keyword}" f" {job_name} {job_path}\n")
            elif keyword == ConfigKeys.QUEUE_OPTION:
                for setting in keyword_value:
                    config.write(
                        f"{keyword}"
                        + f" {queue_name(setting[ConfigKeys.DRIVER_NAME])}"
                        + f" {setting[ConfigKeys.OPTION]}"
                        + (
                            f" {setting[ConfigKeys.VALUE]}\n"
                            if ConfigKeys.VALUE in setting
                            else "\n"
                        )
                    )
            elif keyword == ConfigKeys.QUEUE_SYSTEM:
                config.write(f"{keyword}" f" {keyword_value.name[:-7]}\n")
            else:
                config.write(f"{keyword} {keyword_value}\n")
