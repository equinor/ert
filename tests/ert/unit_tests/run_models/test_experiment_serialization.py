import shutil
import string
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from ert.config import (
    Field,
    ForwardModelStep,
    GenDataConfig,
    HookRuntime,
    ModelConfig,
    QueueSystem,
    SummaryConfig,
    SurfaceConfig,
    Workflow,
)
from ert.config.gen_kw_config import GenKwConfig, TransformFunctionDefinition
from ert.config.parsing import SchemaItemType
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueConfig,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.field_utils import FieldFileFormat
from ert.run_models import EnsembleExperiment
from ert.substitutions import Substitutions


def realistic_text(min_size=1, max_size=8):
    safe_chars = string.ascii_letters + string.digits + "_-"
    return st.text(alphabet=safe_chars, min_size=min_size, max_size=max_size)


optional_nonempty_str = st.one_of(st.none(), realistic_text())


# === Strategy for base fields in QueueOptions ===
@st.composite
def base_queue_fields(draw):
    return {
        "max_running": draw(st.integers(min_value=0, max_value=100)),
        "submit_sleep": draw(st.floats(min_value=0.0, max_value=10.0)),
        "num_cpu": draw(st.integers(min_value=1, max_value=64)),
        "realization_memory": draw(st.integers(min_value=0, max_value=1_000_000)),
        "job_script": shutil.which("fm_dispatch.py") or "fm_dispatch.py",
        "project_code": draw(optional_nonempty_str),
        "activate_script": draw(optional_nonempty_str),
    }


# === Strategies for each queue type ===


@st.composite
def lsf_queue_options_strategy(draw):
    base = draw(base_queue_fields())
    return LsfQueueOptions(
        name=QueueSystem.LSF,
        bhist_cmd=draw(optional_nonempty_str),
        bjobs_cmd=draw(optional_nonempty_str),
        bkill_cmd=draw(optional_nonempty_str),
        bsub_cmd=draw(optional_nonempty_str),
        exclude_host=draw(optional_nonempty_str),
        lsf_queue=draw(optional_nonempty_str),
        lsf_resource=draw(optional_nonempty_str),
        **base,
    )


@st.composite
def torque_queue_options_strategy(draw):
    base = draw(base_queue_fields())
    return TorqueQueueOptions(
        name=QueueSystem.TORQUE,
        qsub_cmd=draw(optional_nonempty_str),
        qstat_cmd=draw(optional_nonempty_str),
        qdel_cmd=draw(optional_nonempty_str),
        queue=draw(optional_nonempty_str),
        cluster_label=draw(optional_nonempty_str),
        job_prefix=draw(optional_nonempty_str),
        keep_qsub_output=draw(st.booleans()),
        **base,
    )


@st.composite
def slurm_queue_options_strategy(draw):
    base = draw(base_queue_fields())
    return SlurmQueueOptions(
        name=QueueSystem.SLURM,
        sbatch_cmd=draw(optional_nonempty_str),
        squeue_cmd=draw(optional_nonempty_str),
        scancel_cmd=draw(optional_nonempty_str),
        queue=draw(optional_nonempty_str),
        cluster_label=draw(optional_nonempty_str),
        job_prefix=draw(optional_nonempty_str),
        keep_sbatch_output=draw(st.booleans()),
        **base,
    )


@st.composite
def local_queue_options_strategy(draw):
    base = draw(base_queue_fields())
    return LocalQueueOptions(name=QueueSystem.LOCAL, **base)


# === Unified strategy ===


queue_options_strategy = st.one_of(
    lsf_queue_options_strategy(),
    torque_queue_options_strategy(),
    slurm_queue_options_strategy(),
    local_queue_options_strategy(),
)


@st.composite
def queue_config_strategy(draw):
    queue_options = draw(queue_options_strategy)

    return QueueConfig(
        max_submit=draw(st.integers(min_value=1, max_value=4)),
        queue_system=queue_options.name,
        queue_options=queue_options,
        stop_long_running=draw(st.booleans()),
        max_runtime=50000,
    )


@st.composite
def forward_model_step_strategy(draw, substitutions: Substitutions):
    def optional_file():
        safe_chars = string.ascii_letters + string.digits
        filename_base = draw(st.text(alphabet=safe_chars, min_size=1, max_size=5))
        return draw(st.one_of(st.none(), st.just(f"{filename_base}.txt")))

    name = draw(st.text(min_size=1, max_size=15))
    executable = draw(st.text(min_size=1, max_size=30))

    stdin_file = optional_file()
    stdout_file = optional_file()
    stderr_file = optional_file()
    start_file = optional_file()
    target_file = optional_file()
    error_file = optional_file()

    max_running_minutes = draw(
        st.one_of(st.none(), st.integers(min_value=1, max_value=10_000))
    )
    min_arg = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=5)))
    max_arg = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=10)))

    arglist = draw(st.lists(realistic_text()))
    required_keywords = draw(st.lists(realistic_text()))

    arg_types = draw(
        st.lists(
            st.sampled_from(list(SchemaItemType)),
            min_size=0,
            max_size=5,
        )
    )

    environment = draw(
        st.dictionaries(
            realistic_text(),
            st.one_of(realistic_text(), st.integers()),
            max_size=5,
        )
    )

    default_mapping = draw(
        st.dictionaries(
            realistic_text(),
            st.one_of(realistic_text(), st.integers()),
            max_size=5,
        )
    )

    return ForwardModelStep(
        name=name,
        executable=executable,
        stdin_file=stdin_file,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        start_file=start_file,
        target_file=target_file,
        error_file=error_file,
        max_running_minutes=max_running_minutes,
        min_arg=min_arg,
        max_arg=max_arg,
        arglist=arglist,
        required_keywords=required_keywords,
        arg_types=arg_types,
        environment=environment,
        default_mapping=default_mapping,
        private_args=substitutions,
    )


@st.composite
def workflow_strategy(draw):
    src_file = draw(realistic_text().map(lambda s: f"{s}.wf.json"))

    # Placeholder: _WorkflowJob is mocked as a string identifier
    job = draw(realistic_text())
    job_args = draw(
        st.one_of(
            st.none(),
            st.integers(),
            realistic_text(),
            st.lists(realistic_text(), max_size=3),
            st.dictionaries(realistic_text(), st.integers()),
        )
    )

    cmd_list = [(job, job_args)]  # could expand to more items if needed

    return Workflow(src_file=src_file, cmd_list=cmd_list)


@st.composite
def hooked_workflows_strategy(draw):
    keys = draw(
        st.lists(
            st.sampled_from(list(HookRuntime)), unique=True, min_size=1, max_size=3
        )
    )
    result = defaultdict(list)

    for key in keys:
        workflows = draw(st.lists(workflow_strategy(), min_size=1, max_size=3))
        result[key] = workflows

    return result


@st.composite
def baserunmodel_args(draw):
    storage_path = draw(realistic_text())
    runpath_file = Path(tempfile.mktemp())
    user_config_file = Path(tempfile.mktemp())
    log_path = Path(tempfile.mktemp())

    n_realizations = draw(st.integers(min_value=1, max_value=200))

    env_vars = draw(st.dictionaries(realistic_text(), realistic_text(), max_size=5))
    env_pr_fm_step = draw(
        st.dictionaries(
            realistic_text(),
            st.dictionaries(
                realistic_text(), st.one_of(st.integers(), st.text(), st.floats())
            ),
            max_size=3,
        )
    )

    # Ensure at least one True in the list of exactly n_realizations length
    true_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_realizations - 1),
            min_size=1,
            max_size=n_realizations,
            unique=True,
        )
    )
    active_realizations = [i in true_indices for i in range(n_realizations)]

    random_seed = draw(st.integers(min_value=0))
    start_iteration = draw(st.integers(min_value=0))
    minimum_required_realizations = draw(
        st.integers(min_value=0, max_value=n_realizations)
    )

    runpath_config = ModelConfig(
        num_realizations=draw(st.integers(min_value=1, max_value=n_realizations)),
        runpath_format_string="simulations/realization-<IENS>/iter-<ITER>",
        jobname_format_string="<CONFIG_FILE>-<IENS>",
        eclbase_format_string="ECLBASE<IENS>",
        gen_kw_export_name="parameters",
    )

    queue_config = draw(queue_config_strategy())

    substitutions_dict = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.text(min_size=1, max_size=20),
            max_size=5,
        )
    )
    substitutions = Substitutions(substitutions_dict)

    forward_model_steps = draw(
        st.lists(forward_model_step_strategy(substitutions), min_size=1, max_size=5)
    )

    hooked_workflows = draw(hooked_workflows_strategy())

    return {
        "storage_path": storage_path,
        "runpath_file": runpath_file,
        "user_config_file": user_config_file,
        "env_vars": env_vars,
        "env_pr_fm_step": env_pr_fm_step,
        "active_realizations": active_realizations,
        "log_path": log_path,
        "random_seed": random_seed,
        "start_iteration": start_iteration,
        "minimum_required_realizations": minimum_required_realizations,
        "runpath_config": runpath_config,
        "queue_config": queue_config,
        "forward_model_steps": forward_model_steps,
        "substitutions": substitutions,
        "hooked_workflows": hooked_workflows,
    }


@st.composite
def ensemble_experiment_strategy(draw):
    ensemble_name = draw(realistic_text())
    experiment_name = draw(realistic_text())
    design_matrix = None
    ert_templates = []

    return {
        "ensemble_name": ensemble_name,
        "experiment_name": experiment_name,
        "design_matrix": design_matrix,
        "ert_templates": ert_templates,
    }


@st.composite
def transform_function_definition_strategy(draw):
    name = draw(realistic_text())
    param_name = draw(realistic_text())
    values = draw(
        st.lists(
            st.one_of(
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                realistic_text(),
            ),
            max_size=5,
        )
    )
    return TransformFunctionDefinition(name=name, param_name=param_name, values=values)


@st.composite
def gen_kw_config_strategy(draw):
    transform_fns = draw(st.lists(transform_function_definition_strategy(), max_size=3))

    return GenKwConfig(
        name=draw(realistic_text()),
        forward_init=draw(st.booleans()),
        update=draw(st.booleans()),
        transform_function_definitions=transform_fns,
    )


@st.composite
def surface_config_strategy(draw):
    return SurfaceConfig(
        name=draw(realistic_text()),
        forward_init=draw(st.booleans()),
        update=draw(st.booleans()),
        ncol=draw(st.integers(min_value=1, max_value=1000)),
        nrow=draw(st.integers(min_value=1, max_value=1000)),
        xori=draw(st.floats(allow_nan=False, allow_infinity=False)),
        yori=draw(st.floats(allow_nan=False, allow_infinity=False)),
        xinc=draw(st.floats(min_value=0.001, allow_nan=False, allow_infinity=False)),
        yinc=draw(st.floats(min_value=0.001, allow_nan=False, allow_infinity=False)),
        rotation=draw(st.floats(min_value=0.0, max_value=360.0)),
        yflip=draw(st.integers(min_value=0, max_value=1)),
        forward_init_file=draw(realistic_text().map(lambda s: f"{s}.txt")),
        output_file=Path(draw(realistic_text().map(lambda s: f"{s}.dat"))),
        base_surface_path=draw(realistic_text()),
    )


@st.composite
def field_config_strategy(draw):
    file_formats = list(FieldFileFormat)  # Assuming it's an Enum

    return Field(
        name=draw(realistic_text()),
        forward_init=draw(st.booleans()),
        update=draw(st.booleans()),
        nx=draw(st.integers(min_value=1, max_value=100)),
        ny=draw(st.integers(min_value=1, max_value=100)),
        nz=draw(st.integers(min_value=1, max_value=100)),
        file_format=draw(st.sampled_from(file_formats)),
        output_transformation=draw(optional_nonempty_str),
        input_transformation=draw(optional_nonempty_str),
        truncation_min=draw(
            st.one_of(st.none(), st.floats(min_value=-1e6, max_value=1e6))
        ),
        truncation_max=draw(
            st.one_of(st.none(), st.floats(min_value=-1e6, max_value=1e6))
        ),
        forward_init_file=draw(realistic_text().map(lambda s: f"{s}.init")),
        output_file=Path(draw(realistic_text().map(lambda s: f"{s}.dat"))),
        grid_file=draw(realistic_text().map(lambda s: f"{s}.grid")),
        mask_file=draw(
            st.one_of(st.none(), realistic_text().map(lambda s: Path(f"{s}.mask")))
        ),
    )


@st.composite
def parameter_config_strategy(draw):
    return draw(
        st.one_of(
            surface_config_strategy(),
            field_config_strategy(),
            gen_kw_config_strategy(),  # ✅ now fully enabled
        )
    )


@st.composite
def gen_data_config_strategy(draw):
    input_files = draw(st.lists(realistic_text().map(lambda s: f"{s}.dat"), max_size=3))
    keys = draw(st.lists(realistic_text(), max_size=5))
    report_steps_list = draw(
        st.lists(
            st.one_of(
                st.none(), st.lists(st.integers(min_value=0, max_value=100), max_size=5)
            ),
            max_size=3,
        )
    )

    return GenDataConfig(
        input_files=input_files,
        keys=keys,
        report_steps_list=report_steps_list,
        has_finalized_keys=True,
    )


@st.composite
def summary_config_strategy(draw):
    input_files = draw(
        st.lists(realistic_text().map(lambda s: f"{s}.SMSPEC"), max_size=3)
    )
    keys = draw(st.lists(realistic_text(), max_size=5))

    refcase = draw(
        st.one_of(
            st.none(),
            st.lists(realistic_text(), max_size=5),
            st.sets(
                st.datetimes(
                    min_value=datetime(2000, 1, 1), max_value=datetime(2030, 12, 31)
                ),
                max_size=5,
            ),
        )
    )

    return SummaryConfig(
        input_files=input_files, keys=keys, refcase=refcase, has_finalized_keys=False
    )


@st.composite
def response_config_list_strategy(draw):
    configs = []

    # Randomly decide whether to include each (at most one of each type)
    if draw(st.booleans()):
        configs.append(draw(gen_data_config_strategy()))
    if draw(st.booleans()):
        configs.append(draw(summary_config_strategy()))

    return configs


@given(baserunmodel_args(), ensemble_experiment_strategy())
def test_ensemble_experiment(
    baserunmodel_args: dict[str, Any], ensemble_experiment_args: dict[str, Any]
) -> None:
    runmodel = EnsembleExperiment(**baserunmodel_args, **ensemble_experiment_args)
    serialized = runmodel.model_dump_json()
    runmodel2 = EnsembleExperiment.model_validate_json(serialized)
    assert runmodel == runmodel2
