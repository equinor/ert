import stat
from pathlib import Path
from shutil import which
from textwrap import dedent

import pytest
import yaml

from ert.base_model_context import use_runtime_plugins
from ert.config import ConfigWarning, SummaryConfig
from ert.plugins import get_site_plugins
from ert.run_models.everest_run_model import EverestRunModel, _get_workflow_jobs
from everest.config import EverestConfig, InstallDataConfig
from tests.everest.utils import everest_config_with_defaults


def test_default_installed_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = [
        "template_render",
        "make_directory",
        "copy_directory",
        "copy_file",
        "move_file",
        "symlink",
    ]
    ever_config = everest_config_with_defaults(
        **yaml.safe_load(
            dedent(f"""
    model: {{"realizations": [0]}}
    forward_model: {jobs}
    """)
        )
    )

    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=site_plugins)

    assert [fm.name for fm in runmodel.forward_model_steps[1:]] == jobs


@pytest.mark.filterwarnings(
    "ignore:Config contains a SUMMARY key but no forward model steps"
)
@pytest.mark.parametrize(
    "config_yaml",
    [
        dedent("""
    wells: [{ name: test}]
    """),
        dedent("""
    controls:
      - name: my_control
        type: well_control
        initial_guess: 0.1
        perturbation_magnitude: 0.01
        variables:
          - name: test
            min: 0
            max: 0.1
    """),
    ],
)
def test_combined_wells_everest_to_ert(tmp_path, monkeypatch, config_yaml):
    monkeypatch.chdir(tmp_path)
    Path("my_file").touch()
    Path("my_executable").touch(mode=stat.S_IEXEC)
    ever_config = everest_config_with_defaults(
        **yaml.safe_load(
            config_yaml
            + dedent("""
    model: {"realizations": [0]}
    definitions: {eclbase: my_test_case}
    install_jobs:
      - name: nothing
        executable: my_executable
    forward_model:
      - job: nothing
        results:
          file_name: something
          type: summary
    """)
        )
    )

    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        runmodel = EverestRunModel.create(
            ever_config, "some_exp_name", "batch", runtime_plugins=site_plugins
        )
    smry_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )
    assert "WOPR:*" in smry_config.keys


@pytest.mark.parametrize(
    "source, target, symlink, cmd",
    [
        ["source_file", "target_file", True, "symlink"],
        ["source_file", "target_file", False, "copy_file"],
        ["source_folder", "target_folder", False, "copy_directory"],
    ],
)
def test_install_data_no_init(tmp_path, source, target, symlink, cmd, monkeypatch):
    """
    Configure the everest config with the install_data section and check that the
    correct ert forward models are created
    """
    monkeypatch.chdir(tmp_path)
    Path("source_file").touch()
    Path.mkdir("source_folder")
    ever_config = everest_config_with_defaults(
        model={"realizations": [0]},
        install_data=[{"source": source, "target": target, "link": symlink}],
    )

    errors = EverestConfig.lint_config_dict(ever_config.to_dict())
    assert len(errors) == 0

    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=site_plugins)

    matching_fm_step = next(fm for fm in runmodel.forward_model_steps if fm.name == cmd)
    assert matching_fm_step.arglist == [f"./{source}", target]


def test_workflow_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workflow_jobs = [{"name": "test", "executable": which("echo")}]
    ever_config = everest_config_with_defaults(
        install_workflow_jobs=workflow_jobs, model={"realizations": [0]}
    )
    workflow_jobs = _get_workflow_jobs(ever_config)
    jobs = workflow_jobs.get("test")
    assert jobs.executable == which("echo")


def test_workflow_job_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    echo = which("echo")
    workflow_jobs = [
        {"name": "test", "executable": which("true")},
        {"name": "test", "executable": echo},
    ]
    ever_config = everest_config_with_defaults(
        install_workflow_jobs=workflow_jobs, model={"realizations": [0]}
    )
    with pytest.warns(
        ConfigWarning,
        match=f"Duplicate workflow job with name 'test', overriding it with {echo!r}.",
    ):
        workflow_jobs = _get_workflow_jobs(ever_config)
    jobs = workflow_jobs.get("test")
    assert jobs.executable == echo


def test_workflows_deprecated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("TEST").write_text("EXECUTABLE echo", encoding="utf-8")
    workflow_jobs = [{"name": "my_test", "source": "TEST"}]
    workflow = {"pre_simulation": ["my_test"]}
    with pytest.warns(
        ConfigWarning, match="`install_workflow_jobs: source` is deprecated"
    ):
        ever_config = everest_config_with_defaults(
            workflows=workflow,
            model={"realizations": [0]},
            install_workflow_jobs=workflow_jobs,
        )
    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=site_plugins)
        assert (
            runmodel.hooked_workflows.popitem()[1][0].cmd_list[0][0].executable
            == "echo"
        )


def test_workflows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executable = which("echo")
    workflow_jobs = [{"name": "my_test", "executable": executable}]
    workflow = {"pre_simulation": ["my_test"]}
    ever_config = everest_config_with_defaults(
        workflows=workflow,
        model={"realizations": [0]},
        install_workflow_jobs=workflow_jobs,
    )
    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=site_plugins)
        assert (
            runmodel.hooked_workflows.popitem()[1][0].cmd_list[0][0].executable
            == executable
        )


def test_user_config_jobs_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing_job = "copy_file"
    ever_config = everest_config_with_defaults(model={"realizations": [0]})
    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        runmodel = EverestRunModel.create(ever_config, runtime_plugins=site_plugins)

    assert runmodel.forward_model_steps[0].name == existing_job
    runmodel._storage.close()
    echo = which("echo")

    ever_config_new = everest_config_with_defaults(
        model={"realizations": [0]},
        install_jobs=[{"name": existing_job, "executable": echo}],
    )
    with use_runtime_plugins(site_plugins):
        runmodel_new = EverestRunModel.create(
            ever_config_new, runtime_plugins=site_plugins
        )

    only_fm_step = runmodel_new.forward_model_steps[0]
    assert only_fm_step.executable == echo
    assert only_fm_step.name == existing_job


def test_passthrough_explicit_summary_keys(change_to_tmpdir):
    custom_sum_keys = [
        "GOIR:PRODUC",
        "GOIT:INJECT",
        "GOIT:PRODUC",
        "GWPR:INJECT",
        "GWPR:PRODUC",
        "GWPT:INJECT",
        "GWPT:PRODUC",
        "GWIR:INJECT",
    ]

    config = everest_config_with_defaults(
        forward_model=[
            {
                "job": "eclipse100 eclipse/model/EgG.DATA --version 2020.2",
                "results": {
                    "file_name": "eclipse/model/EGG",
                    "type": "summary",
                    "keys": custom_sum_keys,
                },
            }
        ]
    )
    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        runmodel = EverestRunModel.create(
            config, "some_exp_name", "batch", runtime_plugins=site_plugins
        )
    smry_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )

    assert set(custom_sum_keys).issubset(set(smry_config.keys))


def test_that_export_keywords_are_turned_into_summary_config_keys(
    monkeypatch, tmp_path, min_config
):
    monkeypatch.chdir(tmp_path)
    extra_sum_keys = [
        "GOIR:PRODUC",
        "GOIT:INJECT",
        "GOIT:PRODUC",
        "GWPR:INJECT",
        "GWPR:PRODUC",
        "GWPT:INJECT",
        "GWPT:PRODUC",
        "GWIR:INJECT",
    ]

    min_config["export"] = {"keywords": extra_sum_keys}
    min_config["forward_model"] = [
        {
            "job": "eclipse100 CASE.DATA",
            "results": {"file_name": "CASE", "type": "summary"},
        }
    ]
    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        config = EverestConfig(**min_config)
        runmodel = EverestRunModel.create(
            config, "exp", "batch", runtime_plugins=site_plugins
        )
    summary_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )
    assert set(extra_sum_keys).issubset(summary_config.keys)


def test_that_summary_keys_are_passed_through_forward_model_results(
    monkeypatch, tmp_path, min_config
):
    monkeypatch.chdir(tmp_path)
    min_config["forward_model"] = [
        {
            "job": "eclipse100 CASE.DATA",
            "results": {
                "file_name": "CASE",
                "type": "summary",
                "keys": ["one", "two", "three"],
            },
        }
    ]

    site_plugins = get_site_plugins()
    with use_runtime_plugins(site_plugins):
        config = EverestConfig(**min_config)
        runmodel = EverestRunModel.create(
            config, "exp", "batch", runtime_plugins=site_plugins
        )

    summary_config = next(
        r for r in runmodel.response_configuration if isinstance(r, SummaryConfig)
    )

    assert {"one", "two", "three"}.issubset(summary_config.keys)


def test_that_install_data_raises_error_on_missing_copy_file(tmp_path):
    source_file = tmp_path / "some_file.json"
    source_file.write_text('{"mock_key": "mock_value"}')

    config = InstallDataConfig(source=str(source_file), target="the_output.json")

    with pytest.raises(KeyError, match=r"ERT forward model: copy_file to be installed"):
        config.to_ert_forward_model_step(
            config_directory=str(tmp_path),
            output_directory=str(tmp_path / "output"),
            model_realizations=[0],
            installed_fm_steps={},
        )


def test_that_install_data_raises_error_on_missing_copy_directory(tmp_path):
    config_directory = tmp_path / "config_dir"
    source_directory = config_directory / "<REALIZATION_ID>"
    realizations = [0, 1, 2]

    for realization in realizations:
        realization_dir = source_directory.with_name(
            source_directory.name.replace("<REALIZATION_ID>", str(realization))
        )
        realization_dir.mkdir(parents=True)

    config = InstallDataConfig(
        source=str(source_directory), target="target_dir", link=False
    )

    with pytest.raises(
        KeyError, match=r"ERT forward model: copy_directory to be installed"
    ):
        config.to_ert_forward_model_step(
            config_directory=str(config_directory),
            output_directory=str(tmp_path / "output"),
            model_realizations=realizations,
            installed_fm_steps={},
        )


def test_that_install_data_raises_error_on_missing_symlink(tmp_path):
    source_file = tmp_path / "source_file.json"
    source_file.write_text('{"mock_key": "mock_value"}')

    config = InstallDataConfig(
        source=str(source_file),
        target="linked_file.json",
        link=True,
    )

    with pytest.raises(KeyError, match=r"ERT forward model: symlink to be installed"):
        config.to_ert_forward_model_step(
            config_directory=str(tmp_path),
            output_directory=str(tmp_path / "output"),
            model_realizations=[0],
            installed_fm_steps={},
        )
