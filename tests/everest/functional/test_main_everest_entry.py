import os
from textwrap import dedent

import pytest
from ruamel.yaml import YAML
from seba_sqlite.snapshot import SebaSnapshot
from tests.everest.utils import (
    capture_streams,
    skipif_no_everest_models,
)

from everest import __version__ as everest_version
from everest.bin.main import start_everest
from everest.config import EverestConfig, ServerConfig
from everest.detached import ServerStatus, everserver_status

WELL_ORDER = "everest/model/config.yml"

pytestmark = pytest.mark.xdist_group(name="starts_everest")


@pytest.mark.integration_test
def test_everest_entry_docs():
    """Test calling everest with --docs

    Note that the correctness of the information printed out is checked by
    other tests. Here we just check that the entry point triggers the
    correct execution paths in the applcation
    """
    with (
        capture_streams() as (out, err),
        pytest.raises(SystemExit),
    ):  # there is a call to sys.exit
        start_everest(["everest", "--docs"])
    lines = [line.strip() for line in out.getvalue().split("\n")]
    assert "wells (optional)" in lines
    assert "controls (required)" in lines
    assert "objective_functions (required)" in lines
    assert "definitions (optional)" in lines
    assert not err.getvalue()


@pytest.mark.integration_test
def test_everest_entry_manual():
    """Test calling everest with --manual"""
    with capture_streams() as (out, err), pytest.raises(SystemExit):
        start_everest(["everest", "--manual"])
    lines = [line.strip() for line in out.getvalue().split("\n")]
    assert "wells (optional)" in lines
    assert "controls (required)" in lines
    assert "objective_functions (required)" in lines
    assert "definitions (optional)" in lines

    doc_lines = [line for line in lines if line.startswith("| Documentation:")]
    assert doc_lines != []
    assert not err.getvalue()


@pytest.mark.integration_test
def test_everest_entry_version():
    """Test calling everest with --version"""
    with capture_streams() as (out, err), pytest.raises(SystemExit):
        start_everest(["everest", "--version"])

    channels = [err.getvalue(), out.getvalue()]
    assert any(everest_version in channel for channel in channels)


@pytest.mark.integration_test
def test_everest_main_entry_bad_command():
    # Setup command line arguments for the test
    with capture_streams() as (_, err), pytest.raises(SystemExit):
        start_everest(["everest", "bad_command"])
    lines = [line.strip() for line in err.getvalue().split("\n")]
    # Check everest run fails and correct err msg is displayed
    assert "The most commonly used everest commands are:" in lines
    assert "Run everest <command> --help for more information on a command" in lines


@pytest.mark.flaky(reruns=5)
@pytest.mark.fails_on_macos_github_workflow
@pytest.mark.integration_test
def test_everest_entry_run(cached_example):
    _, config_file, _ = cached_example("math_func/config_minimal.yml")
    # Setup command line arguments
    with capture_streams():
        start_everest(["everest", "run", config_file])

    config = EverestConfig.load_file(config_file)
    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.completed

    snapshot = SebaSnapshot(config.optimization_output_dir).get_snapshot()

    best_settings = snapshot.optimization_data[-1]
    assert best_settings.controls["point_x"] == pytest.approx(0.5, abs=0.05)
    assert best_settings.controls["point_y"] == pytest.approx(0.5, abs=0.05)
    assert best_settings.controls["point_z"] == pytest.approx(0.5, abs=0.05)

    assert best_settings.objective_value == pytest.approx(0.0, abs=0.0005)

    with capture_streams():
        start_everest(["everest", "monitor", config_file])

    config = EverestConfig.load_file(config_file)
    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.completed


@pytest.mark.integration_test
def test_everest_entry_monitor_no_run(cached_example):
    _, config_file, _ = cached_example("math_func/config_minimal.yml")
    with capture_streams():
        start_everest(["everest", "monitor", config_file])

    config = EverestConfig.load_file(config_file)
    status = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )

    assert status["status"] == ServerStatus.never_run


@pytest.mark.integration_test
def test_everest_main_export_entry(cached_example):
    # Setup command line arguments
    _, config_file, _ = cached_example("math_func/config_minimal.yml")
    with capture_streams():
        start_everest(["everest", "export", config_file])
    assert os.path.exists(os.path.join("everest_output", "config_minimal.csv"))


@pytest.mark.integration_test
def test_everest_main_lint_entry(cached_example):
    # Setup command line arguments
    _, config_file, _ = cached_example("math_func/config_minimal.yml")
    with capture_streams() as (out, err):
        start_everest(["everest", "lint", config_file])
    assert "config_minimal.yml is valid" in out.getvalue()

    # Make the config invalid
    with open(config_file, encoding="utf-8") as f:
        raw_config = YAML(typ="safe", pure=True).load(f)
    raw_config["controls"][0]["initial_guess"] = "invalid"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml = YAML(typ="safe", pure=True)
        yaml.indent = 2
        yaml.default_flow_style = False
        yaml.dump(raw_config, f)

    with capture_streams() as (out, err), pytest.raises(SystemExit):
        start_everest(["everest", "lint", config_file])

    type_ = "(type=float_parsing)"
    validation_msg = dedent(
        f"""Loading config file <config_minimal.yml> failed with:
Found 1 validation error:

controls -> 0 -> initial_guess
    * Input should be a valid number, unable to parse string as a number {type_}
"""
    )
    assert validation_msg in err.getvalue()


@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.integration_test
def test_everest_main_configdump_entry(copy_egg_test_data_to_tmp):
    # Setup command line arguments
    with capture_streams() as (out, _):
        start_everest(["everest", "render", WELL_ORDER])
    yaml = YAML(typ="safe", pure=True)
    render_dict = yaml.load(out.getvalue())

    # Test whether the config file is correctly rendered with jinja
    data_file = (
        "everest/model/../../eclipse/include/realizations/"
        "realization-<GEO_ID>/eclipse/model/EGG.DATA"
    )
    assert render_dict["definitions"]["data_file"] == os.path.join(
        os.getcwd(), data_file
    )
