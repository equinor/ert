import os
from textwrap import dedent

import pytest
from ruamel.yaml import YAML
from seba_sqlite.snapshot import SebaSnapshot

from everest import __version__ as everest_version
from everest.bin.main import start_everest
from everest.config import EverestConfig
from everest.detached import (
    ServerStatus,
    context_stop_and_wait,
    everserver_status,
    wait_for_context,
)
from tests.utils import capture_streams, relpath, tmpdir

CONFIG_PATH = relpath("..", "examples", "math_func")
CONFIG_FILE_MINIMAL = "config_minimal.yml"

EGG_CONFIG_PATH = relpath("..", "examples", "egg")
WELL_ORDER = "everest/model/config.yml"

pytestmark = pytest.mark.xdist_group(name="starts_everest")


def test_everest_entry_docs():
    """Test calling everest with --docs

    Note that the correctness of the information printed out is checked by
    other tests. Here we just check that the entry point triggers the
    correct execution paths in the applcation
    """
    with capture_streams() as (out, err):
        with pytest.raises(SystemExit):  # there is a call to sys.exit
            start_everest(["everest", "--docs"])
    lines = [line.strip() for line in out.getvalue().split("\n")]
    assert "wells (optional)" in lines
    assert "controls (required)" in lines
    assert "objective_functions (required)" in lines
    assert "definitions (optional)" in lines
    assert err.getvalue() == ""


def test_everest_entry_manual():
    """Test calling everest with --manual"""
    with capture_streams() as (out, err):
        with pytest.raises(SystemExit):
            start_everest(["everest", "--manual"])
    lines = [line.strip() for line in out.getvalue().split("\n")]
    assert "wells (optional)" in lines
    assert "controls (required)" in lines
    assert "objective_functions (required)" in lines
    assert "definitions (optional)" in lines

    doc_lines = [line for line in lines if line.startswith("| Documentation:")]
    assert doc_lines != []
    assert err.getvalue() == ""


def test_everest_entry_version():
    """Test calling everest with --version"""
    with capture_streams() as (out, err):
        with pytest.raises(SystemExit):
            start_everest(["everest", "--version"])

    channels = [err.getvalue(), out.getvalue()]
    assert any(everest_version in channel for channel in channels)


def test_everest_main_entry_bad_command():
    # Setup command line arguments for the test
    with capture_streams() as (out, err):
        with pytest.raises(SystemExit):
            start_everest(["everest", "bad_command"])
    lines = [line.strip() for line in err.getvalue().split("\n")]
    # Check everest run fails and correct err msg is displayed
    assert "The most commonly used everest commands are:" in lines
    assert "Run everest <command> --help for more information on a command" in lines


@pytest.mark.flaky(reruns=5)
@tmpdir(CONFIG_PATH)
@pytest.mark.fails_on_macos_github_workflow
def test_everest_entry_run():
    wait_for_context()
    # Setup command line arguments
    with capture_streams() as (out, err):
        start_everest(["everest", "run", CONFIG_FILE_MINIMAL])

    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    status = everserver_status(config)

    assert status["status"] == ServerStatus.completed

    snapshot = SebaSnapshot(config.optimization_output_dir).get_snapshot()

    best_settings = snapshot.optimization_data[-1]
    assert best_settings.controls["point_0_x"] == pytest.approx(0.5, abs=0.05)
    assert best_settings.controls["point_0_y"] == pytest.approx(0.5, abs=0.05)
    assert best_settings.controls["point_0_z"] == pytest.approx(0.5, abs=0.05)

    assert best_settings.objective_value == pytest.approx(0.0, abs=0.0005)

    context_stop_and_wait()

    with capture_streams() as (out, err):
        start_everest(["everest", "monitor", CONFIG_FILE_MINIMAL])

    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    status = everserver_status(config)

    assert status["status"] == ServerStatus.completed

    context_stop_and_wait()


@tmpdir(CONFIG_PATH)
def test_everest_entry_monitor_no_run():
    with capture_streams() as (out, err):
        start_everest(["everest", "monitor", CONFIG_FILE_MINIMAL])

    config = EverestConfig.load_file(CONFIG_FILE_MINIMAL)
    status = everserver_status(config)

    assert status["status"] == ServerStatus.never_run

    context_stop_and_wait()


@tmpdir(CONFIG_PATH)
def test_everest_main_export_entry():
    # Setup command line arguments
    with capture_streams() as (out, err):
        start_everest(["everest", "export", CONFIG_FILE_MINIMAL])
    assert os.path.exists(os.path.join("everest_output", "config_minimal.csv"))


@tmpdir(CONFIG_PATH)
def test_everest_main_lint_entry():
    # Setup command line arguments
    with capture_streams() as (out, err):
        start_everest(["everest", "lint", CONFIG_FILE_MINIMAL])
    assert "config_minimal.yml is valid" in out.getvalue()

    # Make the config invalid
    with open(CONFIG_FILE_MINIMAL, encoding="utf-8") as f:
        raw_config = YAML(typ="safe", pure=True).load(f)
    raw_config["controls"][0]["initial_guess"] = "invalid"
    with open(CONFIG_FILE_MINIMAL, "w", encoding="utf-8") as f:
        yaml = YAML(typ="safe", pure=True)
        yaml.indent = 2
        yaml.default_flow_style = False
        yaml.dump(raw_config, f)

    with capture_streams() as (out, err):
        with pytest.raises(SystemExit):
            start_everest(["everest", "lint", CONFIG_FILE_MINIMAL])

    _type = "(type=float_parsing)"
    validation_msg = dedent(
        f"""Loading config file <config_minimal.yml> failed with:
Found  1 validation error:

controls -> 0 -> initial_guess
    * Input should be a valid number, unable to parse string as a number {_type}
"""
    )
    assert validation_msg in err.getvalue()


@tmpdir(EGG_CONFIG_PATH)
@pytest.mark.fails_on_macos_github_workflow
def test_everest_main_configdump_entry():
    # Setup command line arguments
    with capture_streams() as (out, err):
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
