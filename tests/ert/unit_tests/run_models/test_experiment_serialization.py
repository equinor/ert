import os
import queue
import shutil
import string
import tempfile
import unittest
import uuid
import warnings
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, note, settings
from hypothesis import strategies as st
from pytest import MonkeyPatch, TempPathFactory

from ert.config import (
    ConfigWarning,
    ErtConfig,
    ESSettings,
    ExecutableWorkflow,
    Field,
    ForwardModelStep,
    GenDataConfig,
    HookRuntime,
    ModelConfig,
    ObservationSettings,
    OutlierSettings,
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
from ert.mode_definitions import (
    ENIF_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    EVALUATE_ENSEMBLE_MODE,
    MANUAL_UPDATE_MODE,
)
from ert.run_models import (
    EnsembleExperiment,
    EnsembleInformationFilter,
    EnsembleSmoother,
    MultipleDataAssimilation,
    create_model,
)
from ert.storage import open_storage


def realistic_text(min_size=1, max_size=4):
    safe_chars = string.ascii_letters + string.digits + "_-"
    return st.text(alphabet=safe_chars, min_size=min_size, max_size=max_size)


optional_nonempty_str = st.one_of(st.none(), realistic_text())


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
        sbatch=draw(realistic_text(1, 8)),
        squeue=draw(realistic_text(1, 8)),
        scancel=draw(realistic_text(1, 8)),
        **base,
    )


@st.composite
def local_queue_options_strategy(draw):
    base = draw(base_queue_fields())
    return LocalQueueOptions(name=QueueSystem.LOCAL, **base)


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
def forward_model_step_strategy(draw, substitutions: dict[str, str]):
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

    job = ExecutableWorkflow(
        executable=draw(realistic_text()),
        name=draw(realistic_text()),
        min_args=draw(st.integers(min_value=1, max_value=4)),
        max_args=draw(st.integers(min_value=1, max_value=4)),
        arg_types=draw(st.lists(st.sampled_from(SchemaItemType))),
        stop_on_fail=draw(st.booleans()),
    )

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
def runmodel_args(draw):
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

    substitutions = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.text(min_size=1, max_size=20),
            max_size=5,
        )
    )

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
        "status_queue": queue.SimpleQueue(),  # runtime only
    }


@st.composite
def initial_ensemble_runmodel_strategy(
    draw, min_params: int = 1, max_params: int = 200
):
    response_configs = []

    if draw(st.booleans()):
        response_configs.append(draw(gen_data_config_strategy()))
    if draw(st.booleans()):
        response_configs.append(draw(summary_config_strategy()))

    return {
        "target_ensemble": draw(realistic_text()),
        "experiment_name": draw(realistic_text()),
        "design_matrix": None,
        "ert_templates": [],
        "parameter_configuration": draw(
            st.lists(
                st.one_of(
                    surface_config_strategy(),
                    field_config_strategy(),
                    gen_kw_config_strategy(),
                ),
                min_size=min_params,
                max_size=max_params,
                unique_by=lambda config: config.name,
            )
        ),
        "response_configuration": response_configs,
        "observations": None,
    }


@st.composite
def update_runmodel_strategy(draw):
    return {
        "target_ensemble": draw(realistic_text()),
        "analysis_settings": ESSettings(
            enkf_truncation=draw(st.floats(min_value=1e-10, max_value=1.0)),
            inversion=draw(st.sampled_from(["EXACT", "SUBSPACE"])),
            localization=draw(st.booleans()),
            localization_correlation_threshold=draw(
                st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))
            ),
        ),
        "update_settings": ObservationSettings(
            outlier_settings=OutlierSettings(
                alpha=draw(
                    st.floats(min_value=0.0, allow_nan=False, allow_infinity=False)
                ),
                std_cutoff=draw(
                    st.floats(min_value=1e-12, allow_nan=False, allow_infinity=False)
                ),
            ),
            auto_scale_observations=[["*"]],
        ),
    }


@st.composite
def multidass_strategy(_):
    # Note: this does not test restart runs, it may be
    # better to test that separately
    return {
        "restart_run": False,
        "prior_ensemble_id": None,
        "weights": MultipleDataAssimilation.default_weights,
    }


@st.composite
def transform_function_definition_strategy(draw):
    name = draw(realistic_text())
    param_name = "NORMAL"
    values = [0, 1]
    return TransformFunctionDefinition(name=name, param_name=param_name, values=values)


@st.composite
def gen_kw_config_strategy(draw):
    transform_fns = draw(
        st.lists(
            transform_function_definition_strategy(),
            max_size=3,
            unique_by=lambda config: config.name,
        )
    )

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
def gen_data_config_strategy(draw):
    num_keys = draw(st.integers(min_value=1, max_value=100))
    input_files = draw(
        st.lists(
            realistic_text().map(lambda s: f"{s}.dat"),
            max_size=num_keys,
            min_size=num_keys,
        )
    )
    keys = draw(st.lists(realistic_text(), max_size=num_keys, min_size=num_keys))
    report_steps_list = draw(
        st.lists(
            st.one_of(
                st.none(), st.lists(st.integers(min_value=0, max_value=100), max_size=5)
            ),
            min_size=num_keys,
            max_size=num_keys,
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
    num_keys = draw(st.integers(min_value=1, max_value=100))
    input_files = draw(
        st.lists(
            realistic_text().map(lambda s: f"{s}.SMSPEC"),
            max_size=num_keys,
            min_size=num_keys,
        )
    )
    keys = draw(st.lists(realistic_text(), max_size=num_keys, min_size=num_keys))

    return SummaryConfig(input_files=input_files, keys=keys, has_finalized_keys=False)


_not_yet_serializable_args = {
    # Should not be needed, will be replaced by endpoint
    "status_queue": queue.SimpleQueue(),
    "observations": None,  # Should just be serialized
}


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    runmodel_args(),
    initial_ensemble_runmodel_strategy(),
)
def test_that_deserializing_ensemble_experiment_is_the_inverse_of_serializing(
    tmp_path_factory: TempPathFactory,
    baserunmodel_args: dict[str, Any],
    ensemble_experiment_args: dict[str, Any],
) -> None:
    tmp_path = tmp_path_factory.mktemp("deserializing_ensemble_experiment")
    note(f"Running in directory {tmp_path}")
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        warnings.simplefilter("ignore", category=ConfigWarning)
        runmodel = EnsembleExperiment(
            **(
                baserunmodel_args
                | ensemble_experiment_args
                | _not_yet_serializable_args
            )
        )
        runmodel._storage.close()

        runmodel_from_serialized = EnsembleExperiment.model_validate(
            runmodel.model_dump() | _not_yet_serializable_args
        )

        assert runmodel_from_serialized.model_dump() == runmodel.model_dump()


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    runmodel_args(), initial_ensemble_runmodel_strategy(), update_runmodel_strategy()
)
def test_that_deserializing_ensemble_smoother_is_the_inverse_of_serializing(
    tmp_path_factory: TempPathFactory,
    baserunmodel_args: dict[str, Any],
    initial_ensemble_args: dict[str, Any],
    update_runmodel_args: dict[str, Any],
) -> None:
    tmp_path = tmp_path_factory.mktemp("deserializing_ensemble_smoother")
    note(f"Running in directory {tmp_path}")
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        runmodel = EnsembleSmoother(
            **(baserunmodel_args | initial_ensemble_args | update_runmodel_args)
        )
        runmodel._storage.close()

        runmodel_from_serialized = EnsembleSmoother.model_validate(
            runmodel.model_dump() | _not_yet_serializable_args
        )

        assert runmodel_from_serialized.model_dump() == runmodel.model_dump()


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    runmodel_args(), initial_ensemble_runmodel_strategy(), update_runmodel_strategy()
)
def test_that_deserializing_ensemble_information_filter_is_the_inverse_of_serializing(
    tmp_path_factory: TempPathFactory,
    baserunmodel_args: dict[str, Any],
    initial_ensemble_args: dict[str, Any],
    update_runmodel_args: dict[str, Any],
) -> None:
    tmp_path = tmp_path_factory.mktemp("deserializing_eif")
    note(f"Running in directory {tmp_path}")
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        runmodel = EnsembleInformationFilter(
            **(baserunmodel_args | initial_ensemble_args | update_runmodel_args)
        )
        runmodel._storage.close()

        runmodel_from_serialized = EnsembleInformationFilter.model_validate(
            runmodel.model_dump() | _not_yet_serializable_args
        )

        assert runmodel_from_serialized.model_dump() == runmodel.model_dump()


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@given(
    runmodel_args(),
    initial_ensemble_runmodel_strategy(),
    update_runmodel_strategy(),
    multidass_strategy(),
)
def test_that_deserializing_esmda_is_the_inverse_of_serializing(
    tmp_path_factory: TempPathFactory,
    baserunmodel_args: dict[str, Any],
    initial_ensemble_args: dict[str, Any],
    update_runmodel_args: dict[str, Any],
    multidass_args: dict[str, Any],
) -> None:
    tmp_path = tmp_path_factory.mktemp("deserializing_eif")
    note(f"Running in directory {tmp_path}")
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)

        runmodel = MultipleDataAssimilation(
            **(
                baserunmodel_args
                | initial_ensemble_args
                | update_runmodel_args
                | multidass_args
            )
        )
        runmodel._storage.close()

        runmodel_from_serialized = MultipleDataAssimilation.model_validate(
            runmodel.model_dump() | _not_yet_serializable_args
        )

    assert runmodel_from_serialized.model_dump() == runmodel.model_dump()


def _create_and_verify_runmodel_snapshot(config, snapshot, cli_args, case):
    runmodel = create_model(config, cli_args, queue.SimpleQueue())

    # Override these to avoid user time/env-specifics in snapshots
    runmodel.queue_config.queue_options.job_script = "fm_dispatch.py"
    runmodel.queue_config.queue_options.activate_script = "activate"
    runmodel.substitutions["<DATE>"] = "2000-01-01"

    serialized = runmodel.model_dump_json(indent=2) + "\n"

    # Trim off cwd to make snapshots general
    cwd = str(os.getcwd())
    serialized = serialized.replace(cwd, f"test-data/ert/{case}")

    snapshot.assert_match(
        serialized,
        f"{Path(config.user_config_file).stem}.json",
    )


cases_to_test = [
    "heat_equation/config.ert",
    "poly_example/poly.ert",
    "snake_oil/snake_oil.ert",
]

# Main motivation for these tests: Track and make visible how the
# ERT config (currently) maps to JSON, which should make it clear
# whenever we make changes to these, and also make refactor opportunities
# more apparent.


# These were intentionally split up to be more
# readable/accessible wrt single test case failures
@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.parametrize("case", cases_to_test)
def test_that_dumped_manual_update_matches_snapshot(
    case, copy_case, snapshot, change_to_tmpdir
):
    config_dir, config_file = case.split("/")
    copy_case(config_dir)

    config = ErtConfig.from_file(config_file)
    config.random_seed = 1  # Ensure deterministic
    with (
        open_storage(config.ens_path, mode="w") as storage,
        unittest.mock.patch(  # Ensure deterministic uuid generation
            "ert.storage.local_storage.uuid4", return_value=uuid.UUID(int=0)
        ),
    ):
        prior = storage.create_experiment(name="my_experiment").create_ensemble(
            ensemble_size=config.runpath_config.num_realizations,
            name="dummy",
            iteration=0,
        )

    _create_and_verify_runmodel_snapshot(
        config,
        snapshot,
        cli_args=Namespace(
            mode=MANUAL_UPDATE_MODE,
            realizations="1,2",
            ensemble_id=str(prior.id),
            target_ensemble="posterior<ITER>",
        ),
        case=f"{config_dir}.{config_file}",
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.parametrize("case", cases_to_test)
def test_that_dumped_evaluate_ensemble_matches_snapshot(
    case, copy_case, snapshot, change_to_tmpdir
):
    config_dir, config_file = case.split("/")
    copy_case(config_dir)

    config = ErtConfig.from_file(config_file)
    config.random_seed = 1  # Ensure deterministic
    with (
        open_storage(config.ens_path, mode="w") as storage,
        unittest.mock.patch(  # Ensure deterministic uuid generation
            "ert.storage.local_storage.uuid4", return_value=uuid.UUID(int=0)
        ),
    ):
        prior = storage.create_experiment(name="my_experiment").create_ensemble(
            ensemble_size=config.runpath_config.num_realizations,
            name="dummy",
            iteration=0,
        )

    _create_and_verify_runmodel_snapshot(
        config,
        snapshot,
        cli_args=Namespace(
            mode=EVALUATE_ENSEMBLE_MODE,
            realizations="1,2,3",
            ensemble_id=str(prior.id),
        ),
        case=f"{config_dir}.{config_file}",
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.parametrize("case", cases_to_test)
def test_that_dumped_ensemble_experiment_matches_snapshot(
    case, copy_case, snapshot, change_to_tmpdir
):
    config_dir, config_file = case.split("/")
    copy_case(config_dir)

    config = ErtConfig.from_file(config_file)
    config.random_seed = 1  # Ensure deterministic

    _create_and_verify_runmodel_snapshot(
        config,
        snapshot,
        cli_args=Namespace(
            mode=ENSEMBLE_EXPERIMENT_MODE,
            realizations="1,2",
            experiment_name="default_name",
            current_ensemble="the_experiment",
        ),
        case=f"{config_dir}.{config_file}",
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.parametrize("case", cases_to_test)
def test_that_dumped_ensemble_smoother_matches_snapshot(
    case, copy_case, snapshot, change_to_tmpdir
):
    config_dir, config_file = case.split("/")
    copy_case(config_dir)

    config = ErtConfig.from_file(config_file)
    config.random_seed = 1  # Ensure deterministic

    _create_and_verify_runmodel_snapshot(
        config,
        snapshot,
        cli_args=Namespace(
            mode=ENSEMBLE_SMOOTHER_MODE,
            realizations="1,2",
            target_ensemble="helloworld<ITER>",
            experiment_name="dummy",
        ),
        case=f"{config_dir}.{config_file}",
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.parametrize("case", cases_to_test)
def test_that_dumped_enif_matches_snapshot(case, copy_case, snapshot, change_to_tmpdir):
    config_dir, config_file = case.split("/")
    copy_case(config_dir)

    config = ErtConfig.from_file(config_file)
    config.random_seed = 1  # Ensure deterministic

    _create_and_verify_runmodel_snapshot(
        config,
        snapshot,
        cli_args=Namespace(
            mode=ENIF_MODE,
            realizations="1,2",
            target_ensemble="helloworld<ITER>",
            experiment_name="dummy",
        ),
        case=f"{config_dir}.{config_file}",
    )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.parametrize("case", cases_to_test)
def test_that_dumped_esmda_matches_snapshot(
    case, copy_case, snapshot, change_to_tmpdir
):
    config_dir, config_file = case.split("/")
    copy_case(config_dir)

    config = ErtConfig.from_file(config_file)
    config.random_seed = 1  # Ensure deterministic

    _create_and_verify_runmodel_snapshot(
        config,
        snapshot,
        cli_args=Namespace(
            mode=ES_MDA_MODE,
            realizations="1,2",
            target_ensemble="iter-<ITER>",
            weights="4, 2, 1",
            restart_run=False,
            prior_ensemble_id=None,
            experiment_name="es-mda",
        ),
        case=f"{config_dir}.{config_file}",
    )
