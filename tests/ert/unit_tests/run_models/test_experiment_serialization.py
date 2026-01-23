import os
import queue
import string
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

from ert.base_model_context import use_runtime_plugins
from ert.config import (
    ConfigWarning,
    ErtConfig,
    ErtScriptWorkflow,
    ESSettings,
    ExecutableWorkflow,
    ExternalErtScript,
    Field,
    ForwardModelStep,
    GenDataConfig,
    GenKwConfig,
    HookRuntime,
    ModelConfig,
    ObservationSettings,
    OutlierSettings,
    SummaryConfig,
    SurfaceConfig,
    Workflow,
)
from ert.config.parsing import SchemaItemType
from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    QueueConfig,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.mode_definitions import (
    ENIF_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    EVALUATE_ENSEMBLE_MODE,
    MANUAL_UPDATE_MODE,
)
from ert.plugins import ErtRuntimePlugins
from ert.run_models import (
    EnsembleExperiment,
    EnsembleInformationFilter,
    EnsembleSmoother,
    MultipleDataAssimilation,
    create_model,
)
from ert.run_models.multiple_data_assimilation import MultipleDataAssimilationConfig
from ert.storage import open_storage


def realistic_text(min_size=1, max_size=4):
    safe_chars = string.ascii_letters + string.digits + "_-"
    return st.text(alphabet=safe_chars, min_size=min_size, max_size=max_size)


optional_nonempty_str = st.one_of(st.none(), realistic_text())


base_queue_fields = {
    "max_running": st.integers(min_value=0, max_value=100),
    "submit_sleep": st.floats(min_value=0.0, max_value=10.0),
    "num_cpu": st.integers(min_value=1, max_value=64),
    "realization_memory": st.integers(min_value=0, max_value=1_000_000),
    "project_code": optional_nonempty_str,
    "activate_script": optional_nonempty_str,
}


lsf_queue_options = st.builds(
    LsfQueueOptions,
    bhist_cmd=optional_nonempty_str,
    bjobs_cmd=optional_nonempty_str,
    bkill_cmd=optional_nonempty_str,
    bsub_cmd=optional_nonempty_str,
    exclude_host=optional_nonempty_str,
    lsf_queue=optional_nonempty_str,
    lsf_resource=optional_nonempty_str,
    **base_queue_fields,
)


torque_queue_options = st.builds(
    TorqueQueueOptions,
    qsub_cmd=optional_nonempty_str,
    qstat_cmd=optional_nonempty_str,
    qdel_cmd=optional_nonempty_str,
    queue=optional_nonempty_str,
    cluster_label=optional_nonempty_str,
    job_prefix=optional_nonempty_str,
    **base_queue_fields,
)


slurm_queue_options = st.builds(
    SlurmQueueOptions,
    sbatch=realistic_text(1, 8),
    squeue=realistic_text(1, 8),
    scancel=realistic_text(1, 8),
    **base_queue_fields,
)


local_queue_options = st.builds(LocalQueueOptions, **base_queue_fields)


queue_options = st.one_of(
    lsf_queue_options,
    torque_queue_options,
    slurm_queue_options,
    local_queue_options,
)


@st.composite
def queue_configs(draw):
    queue_option = draw(queue_options)

    return QueueConfig(
        max_submit=draw(st.integers(min_value=1, max_value=4)),
        queue_system=queue_option.name,
        queue_options=queue_option,
        stop_long_running=draw(st.booleans()),
        max_runtime=50000,
    )


@st.composite
def optional_file(draw):
    safe_chars = string.ascii_letters + string.digits
    filename_base = draw(st.text(alphabet=safe_chars, min_size=1, max_size=5))
    return draw(st.one_of(st.none(), st.just(f"{filename_base}.txt")))


def forward_model_steps(substitutions):
    return st.builds(
        ForwardModelStep,
        stdin_file=optional_file(),
        stdout_file=optional_file(),
        stderr_file=optional_file(),
        start_file=optional_file(),
        target_file=optional_file(),
        error_file=optional_file(),
        max_running_minutes=st.one_of(
            st.none(), st.integers(min_value=1, max_value=10_000)
        ),
        min_arg=st.one_of(st.none(), st.integers(min_value=0, max_value=5)),
        max_arg=st.one_of(st.none(), st.integers(min_value=0, max_value=10)),
        default_mapping=st.dictionaries(
            realistic_text(),
            realistic_text(),
            max_size=5,
        ),
        private_args=substitutions,
    )


workflow_jobs = st.one_of(
    st.builds(
        ExecutableWorkflow,
        min_args=st.integers(min_value=1, max_value=4),
        max_args=st.integers(min_value=1, max_value=4),
        arg_types=st.lists(st.sampled_from(SchemaItemType)),
    ),
    st.builds(
        ErtScriptWorkflow,
        min_args=st.just(1),
        max_args=st.just(1),
        arg_types=st.just([SchemaItemType.STRING]),
        ert_script=st.just(ExternalErtScript),
    ),
)

workflows = st.builds(
    Workflow,
    src_file=realistic_text().map(lambda s: f"{s}.wf.json"),
    cmd_list=st.lists(
        st.tuples(
            workflow_jobs,
            st.one_of(
                st.none(),
                st.integers(),
                realistic_text(),
                st.lists(realistic_text(), max_size=3),
                st.dictionaries(realistic_text(), st.integers()),
            ),
        )
    ),
)


@st.composite
def hooked_workflows(draw):
    keys = draw(
        st.lists(st.sampled_from(HookRuntime), unique=True, min_size=1, max_size=3)
    )
    result = defaultdict(list)

    for key in keys:
        result[key] = draw(st.lists(workflows, min_size=1, max_size=3))

    return result


def runmodel_args(draw, tmp_path_factory):
    storage_path = draw(realistic_text())
    tmp_path = tmp_path_factory.mktemp("deserializing_ensemble_experiment")
    (runpath_file := tmp_path / "runpath_file").touch()
    (user_config_file := tmp_path / "config.ert").touch()
    (log_path := tmp_path / "log_path").mkdir()

    n_realizations = draw(st.integers(min_value=1, max_value=200))

    env_vars = draw(st.dictionaries(realistic_text(), realistic_text(), max_size=5))
    env_pr_fm_step = draw(
        st.dictionaries(
            realistic_text(),
            st.dictionaries(
                realistic_text(),
                st.one_of(
                    st.integers(),
                    st.text(),
                    # We do not support nan/infinity here
                    st.floats(allow_infinity=False, allow_nan=False),
                ),
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

    substitutions = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.text(min_size=1, max_size=20),
            max_size=5,
        )
    )

    forward_model_step_list = draw(
        st.lists(forward_model_steps(st.just(substitutions)), min_size=1, max_size=5)
    )

    hooked_workflows_dict = draw(hooked_workflows())
    installed_ertscripts = {}
    for _, workflows in hooked_workflows_dict.items():
        for workflow in workflows:
            for cmd, _ in workflow.cmd_list:
                if isinstance(cmd, ErtScriptWorkflow):
                    installed_ertscripts[cmd.name] = cmd

    overridden_env_vars_in_plugins = (
        {
            k: env_vars[k] + "_PLUGIN"
            for k in draw(st.sets(st.sampled_from(list(env_vars))))
        }
        if len(env_vars) > 0
        else {}
    )
    env_vars_in_plugins = draw(
        st.dictionaries(realistic_text(), realistic_text(), max_size=5)
    )

    runtime_plugins = ErtRuntimePlugins(
        installed_forward_model_steps={},
        installed_workflow_jobs=installed_ertscripts,
        queue_options=None,
        activate_script="",
        environment_variables=env_vars_in_plugins | overridden_env_vars_in_plugins,
        env_pr_fm_step={},
        help_links={},
    )

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
        "queue_config": draw(queue_configs()),
        "forward_model_steps": forward_model_step_list,
        "substitutions": substitutions,
        "hooked_workflows": hooked_workflows_dict,
    }, runtime_plugins


@st.composite
def initial_ensemble_runmodels(draw, min_params: int = 1, max_params: int = 200):
    response_configs = []

    if draw(st.booleans()):
        response_configs.append(draw(gen_data_configs()))
    if draw(st.booleans()):
        response_configs.append(draw(summary_configs()))

    return {
        "target_ensemble": draw(realistic_text()),
        "experiment_name": draw(realistic_text()),
        "design_matrix": None,
        "ert_templates": [],
        "parameter_configuration": draw(
            st.lists(
                st.one_of(
                    surface_configs,
                    field_configs,
                    gen_kw_configs,
                ),
                min_size=min_params,
                max_size=max_params,
                unique_by=lambda config: config.name,
            )
        ),
        "response_configuration": response_configs,
        "observations": [],
    }


@st.composite
def update_runmodels(draw):
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
def multidass(_):
    # Note: this does not test restart runs, it may be
    # better to test that separately
    return {
        "restart_run": False,
        "prior_ensemble_id": None,
        "weights": MultipleDataAssimilationConfig.default_weights,
    }


distribution_strategy = st.fixed_dictionaries(
    {
        "name": st.sampled_from(["normal", "lognormal"]),
        "mean": st.floats(min_value=-100, max_value=100),
        "std": st.floats(min_value=0.001, max_value=10),
    }
)

gen_kw_configs = st.builds(
    GenKwConfig,
    name=st.text(min_size=1, max_size=20),
    distribution=distribution_strategy,
)


surface_configs = st.builds(
    SurfaceConfig,
    ncol=st.integers(min_value=1, max_value=1000),
    nrow=st.integers(min_value=1, max_value=1000),
    xori=st.floats(allow_nan=False, allow_infinity=False),
    yori=st.floats(allow_nan=False, allow_infinity=False),
    xinc=st.floats(min_value=0.001, allow_nan=False, allow_infinity=False),
    yinc=st.floats(min_value=0.001, allow_nan=False, allow_infinity=False),
    rotation=st.floats(min_value=0.0, max_value=360.0),
    yflip=st.integers(min_value=0, max_value=1),
    forward_init_file=realistic_text().map(lambda s: f"{s}.txt"),
    output_file=realistic_text().map(lambda s: Path(f"{s}.dat")),
    base_surface_path=realistic_text(),
)

field_configs = st.builds(
    Field,
    nx=st.integers(min_value=1, max_value=100),
    ny=st.integers(min_value=1, max_value=100),
    nz=st.integers(min_value=1, max_value=100),
    output_transformation=optional_nonempty_str,
    input_transformation=optional_nonempty_str,
    truncation_min=st.one_of(st.none(), st.floats(min_value=-1e6, max_value=1e6)),
    truncation_max=st.one_of(st.none(), st.floats(min_value=-1e6, max_value=1e6)),
    forward_init_file=realistic_text().map(lambda s: f"{s}.init"),
    output_file=realistic_text().map(lambda s: Path(f"{s}.dat")),
    grid_file=realistic_text().map(lambda s: f"{s}.grid"),
    mask_file=st.one_of(st.none(), realistic_text().map(lambda s: Path(f"{s}.mask"))),
)


@st.composite
def gen_data_configs(draw):
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
def summary_configs(draw):
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
}


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    initial_ensemble_runmodels(),
    st.data(),
)
def test_that_deserializing_ensemble_experiment_is_the_inverse_of_serializing(
    tmp_path_factory: pytest.TempPathFactory,
    ensemble_experiment_args: dict[str, Any],
    data,
) -> None:
    tmp_path = tmp_path_factory.mktemp("deserializing_ensemble_experiment")
    baserunmodel_args, runtime_plugins = runmodel_args(data.draw, tmp_path_factory)
    note(f"Running in directory {tmp_path}")
    with (
        pytest.MonkeyPatch.context() as patch,
        use_runtime_plugins(runtime_plugins),
    ):
        patch.chdir(tmp_path)
        warnings.simplefilter("ignore", category=ConfigWarning)
        runmodel = EnsembleExperiment(
            **(
                baserunmodel_args
                | ensemble_experiment_args
                | {"status_queue": queue.SimpleQueue()}
            )
        )
        runmodel._storage.close()

        runmodel_from_serialized = EnsembleExperiment.model_validate(
            runmodel.model_dump(mode="json") | {"status_queue": queue.SimpleQueue()}
        )

        assert (
            runmodel_from_serialized.env_vars
            == runtime_plugins.environment_variables | baserunmodel_args["env_vars"]
        )
        assert runmodel_from_serialized.model_dump(mode="json") == runmodel.model_dump(
            mode="json"
        )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(initial_ensemble_runmodels(), update_runmodels(), st.data())
def test_that_deserializing_ensemble_smoother_is_the_inverse_of_serializing(
    tmp_path_factory: pytest.TempPathFactory,
    initial_ensemble_args: dict[str, Any],
    update_runmodel_args: dict[str, Any],
    data,
) -> None:
    tmp_path = tmp_path_factory.mktemp("deserializing_ensemble_smoother")
    baserunmodel_args, runtime_plugins = runmodel_args(data.draw, tmp_path_factory)
    note(f"Running in directory {tmp_path}")
    with pytest.MonkeyPatch.context() as patch, use_runtime_plugins(runtime_plugins):
        patch.chdir(tmp_path)
        runmodel = EnsembleSmoother(
            **(
                baserunmodel_args
                | initial_ensemble_args
                | update_runmodel_args
                | {"status_queue": queue.SimpleQueue()}
            ),
        )
        runmodel._storage.close()

        runmodel_from_serialized = EnsembleSmoother.model_validate(
            runmodel.model_dump(mode="json") | {"status_queue": queue.SimpleQueue()}
        )

        assert (
            runmodel_from_serialized.env_vars
            == runtime_plugins.environment_variables | baserunmodel_args["env_vars"]
        )
        assert runmodel_from_serialized.model_dump(mode="json") == runmodel.model_dump(
            mode="json"
        )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(initial_ensemble_runmodels(), update_runmodels(), st.data())
def test_that_deserializing_ensemble_information_filter_is_the_inverse_of_serializing(
    tmp_path_factory: pytest.TempPathFactory,
    initial_ensemble_args: dict[str, Any],
    update_runmodel_args: dict[str, Any],
    data,
) -> None:
    tmp_path = tmp_path_factory.mktemp("deserializing_eif")
    baserunmodel_args, runtime_plugins = runmodel_args(data.draw, tmp_path_factory)
    note(f"Running in directory {tmp_path}")
    with pytest.MonkeyPatch.context() as patch, use_runtime_plugins(runtime_plugins):
        patch.chdir(tmp_path)
        runmodel = EnsembleInformationFilter(
            **(
                baserunmodel_args
                | initial_ensemble_args
                | update_runmodel_args
                | {"status_queue": queue.SimpleQueue()}
            ),
        )
        runmodel._storage.close()

        runmodel_from_serialized = EnsembleInformationFilter.model_validate(
            runmodel.model_dump(mode="json") | {"status_queue": queue.SimpleQueue()}
        )

        assert (
            runmodel_from_serialized.env_vars
            == runtime_plugins.environment_variables | baserunmodel_args["env_vars"]
        )
        assert runmodel_from_serialized.model_dump(mode="json") == runmodel.model_dump(
            mode="json"
        )


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@given(initial_ensemble_runmodels(), update_runmodels(), multidass(), st.data())
def test_that_deserializing_esmda_is_the_inverse_of_serializing(
    tmp_path_factory: pytest.TempPathFactory,
    initial_ensemble_args: dict[str, Any],
    update_runmodel_args: dict[str, Any],
    multidass_args: dict[str, Any],
    data,
) -> None:
    tmp_path = tmp_path_factory.mktemp("deserializing_eif")
    baserunmodel_args, runtime_plugins = runmodel_args(data.draw, tmp_path_factory)
    note(f"Running in directory {tmp_path}")

    with pytest.MonkeyPatch.context() as patch, use_runtime_plugins(runtime_plugins):
        patch.chdir(tmp_path)

        runmodel = MultipleDataAssimilation(
            **(
                baserunmodel_args
                | initial_ensemble_args
                | update_runmodel_args
                | multidass_args
            ),
            status_queue=queue.SimpleQueue(),
        )
        runmodel._storage.close()

        runmodel_from_serialized = MultipleDataAssimilation.model_validate(
            runmodel.model_dump() | {"status_queue": queue.SimpleQueue()}
        )

    assert runmodel_from_serialized.model_dump(mode="json") == runmodel.model_dump(
        mode="json"
    )


def _create_and_verify_runmodel_snapshot(config, snapshot, cli_args, case):
    runmodel = create_model(config, cli_args, queue.SimpleQueue())

    # Override these to avoid user time/env-specifics in snapshots
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


@pytest.fixture
def executable_workflow_job():
    return ExecutableWorkflow(
        name="exec_wf_name",
        type="user_installed_executable",
        min_args=4,
        max_args=3,
        arg_types=[
            SchemaItemType.BOOL,
            SchemaItemType.ISODATE,
            SchemaItemType.BYTESIZE,
            SchemaItemType.INT,
            SchemaItemType.INT,
            SchemaItemType.INVALID,
        ],
        stop_on_fail=False,
        executable=None,
    )


@pytest.fixture
def ertscript_workflow_job():
    return ErtScriptWorkflow(
        name="the_ertscript_wf_name",
        type="site_installed",
        min_args=1,
        max_args=1,
        arg_types=[SchemaItemType.STRING],
        stop_on_fail=False,
        ert_script=ExternalErtScript,
        category="other",
    )


def test_that_executable_wf_job_serializes_entire_wfjob(executable_workflow_job):
    with use_runtime_plugins(ErtRuntimePlugins(installed_workflow_jobs={})):
        serialized = executable_workflow_job.model_dump(mode="json")
        deserialized = ExecutableWorkflow.model_validate(serialized)
        assert deserialized == executable_workflow_job


def test_that_workflow_with_executable_wf_job_serializes_entire_wfjob(
    executable_workflow_job,
):
    workflow = Workflow(
        src_file="Ox.wf.json",
        cmd_list=[
            (
                executable_workflow_job,
                "R8oZ",
            ),
        ],
    )

    with use_runtime_plugins(ErtRuntimePlugins(installed_workflow_jobs={})):
        serialized = workflow.model_dump(mode="json")
        deserialized = Workflow.model_validate(serialized)
        assert deserialized == workflow


def test_that_ertscript_wf_job_serializes_ertscript_by_name(ertscript_workflow_job):
    with use_runtime_plugins(
        ErtRuntimePlugins(
            installed_workflow_jobs={
                ertscript_workflow_job.name: ertscript_workflow_job
            }
        )
    ):
        serialized_job = ertscript_workflow_job.model_dump(mode="json")
        deserialized_workflow_job = ErtScriptWorkflow.model_validate(serialized_job)
        assert deserialized_workflow_job == ertscript_workflow_job


def test_that_ertscript_wf_job_deserialization_raises_error_if_uninstalled(
    ertscript_workflow_job,
):
    serialized_job = ertscript_workflow_job.model_dump(mode="json")
    with (
        use_runtime_plugins(
            ErtRuntimePlugins(
                installed_workflow_jobs={
                    f"{ertscript_workflow_job.name}ff": ertscript_workflow_job
                }
            )
        ),
        pytest.raises(
            KeyError,
            match=f"Expected workflow job {ertscript_workflow_job.name}"
            " to be installed",
        ),
    ):
        ErtScriptWorkflow.model_validate(serialized_job)


def test_that_workflow_with_ertscript_serializes_ertscript_by_name(
    ertscript_workflow_job,
):
    workflow = Workflow(
        src_file="Ox.wf.json",
        cmd_list=[
            (ertscript_workflow_job, "hello"),
        ],
    )

    with use_runtime_plugins(
        ErtRuntimePlugins(
            installed_workflow_jobs={
                ertscript_workflow_job.name: ertscript_workflow_job
            }
        )
    ):
        serialized = workflow.model_dump(mode="json")
        deserialized = Workflow.model_validate(serialized)
        assert deserialized == workflow
