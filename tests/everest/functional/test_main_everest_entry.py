import threading
import time
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from ruamel.yaml import YAML
from tests.everest.utils import (
    capture_streams,
    everest_config_with_defaults,
    get_optimal_result,
)

from ert.storage import ExperimentState
from everest import __version__ as everest_version
from everest.bin.main import start_everest
from everest.bin.utils import get_experiment_status
from everest.config import EverestConfig

CONFIG_FILE_ADVANCED = "config_advanced.yml"


@pytest.mark.xdist_group(name="starts_everest")
def test_everest_entry_version():
    """Test calling everest with --version"""
    with capture_streams() as (out, err), pytest.raises(SystemExit):
        start_everest(["everest", "--version"])

    channels = [err.getvalue(), out.getvalue()]
    assert any(everest_version in channel for channel in channels)


@pytest.mark.xdist_group(name="starts_everest")
def test_everest_main_entry_bad_command():
    # Setup command line arguments for the test
    with capture_streams() as (_, err), pytest.raises(SystemExit):
        start_everest(["everest", "bad_command"])
    lines = [line.strip() for line in err.getvalue().split("\n")]
    # Check everest run fails and correct err msg is displayed
    assert "Supported commands:" in lines
    assert "Run everest <command> --help for more information on a command" in lines


@pytest.mark.xdist_group("math_func/config_minimal.yml")
@pytest.mark.integration_test
def test_everest_entry_render(cached_example):
    _, config_file, _, _ = cached_example("math_func/config_minimal.yml")
    with capture_streams() as (out, _):
        start_everest(["everest", "render", config_file])
    assert YAML().load(out.getvalue()), "rendered output to stdout was not yaml"


@pytest.mark.skip_mac_ci
@pytest.mark.integration_test
@pytest.mark.xdist_group("math_func/config_minimal.yml")
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_everest_entry_run(cached_example):
    _, config_file, _, _ = cached_example("math_func/config_minimal.yml")

    # Ensure no interference with plugins which may set queue system
    config_content = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    config_content["simulator"] = {"queue_system": {"name": "local", "max_running": 2}}
    Path(config_file).write_text(
        yaml.dump(config_content, default_flow_style=False), encoding="utf-8"
    )

    # Setup command line arguments
    with capture_streams() as (out, _):
        start_everest(["everest", "run", config_file, "--skip-prompt"])

    assert (
        "EVEREST run finished with: Maximum number of batches reached" in out.getvalue()
    )

    config = EverestConfig.load_file(config_file)
    optimal = get_optimal_result(config.optimization_output_dir)
    assert optimal.controls["point.x"] == pytest.approx(0.5, abs=0.05)
    assert optimal.controls["point.y"] == pytest.approx(0.5, abs=0.05)
    assert optimal.controls["point.z"] == pytest.approx(0.5, abs=0.05)

    assert optimal.total_objective == pytest.approx(0.0, abs=0.0005)

    with capture_streams():
        start_everest(["everest", "monitor", config_file])

    config = EverestConfig.load_file(config_file)
    experiment_status = get_experiment_status(config.storage_dir)
    assert experiment_status.status == ExperimentState.completed


@pytest.mark.integration_test
@pytest.mark.xdist_group("math_func/config_minimal.yml")
def test_everest_entry_monitor_already_run(cached_example):
    _, config_file, _, _ = cached_example("math_func/config_minimal.yml")
    with capture_streams() as (out, _):
        start_everest(["everest", "monitor", config_file])
    assert "Optimization already completed." in out.getvalue()


@pytest.mark.integration_test
def test_everest_entry_monitor_not_run(change_to_tmpdir):
    everest_config_with_defaults().write_to_file("config.yml")
    with capture_streams() as (out, _):
        start_everest(["everest", "monitor", "config.yml"])
    assert "The optimization has not run yet." in out.getvalue()


@pytest.mark.xdist_group("math_func/config_minimal.yml")
@pytest.mark.integration_test
def test_everest_main_lint_entry(cached_example):
    # Setup command line arguments
    _, config_file, _, _ = cached_example("math_func/config_minimal.yml")
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

line: 2, column: 18. controls -> 0 -> initial_guess
    * Input should be a valid number, unable to parse string as a number {type_}
"""
    )
    assert validation_msg in err.getvalue()


@pytest.mark.skip_mac_ci
@pytest.mark.flaky(reruns=3)
@pytest.mark.timeout(60)
@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
def test_that_keyboard_interrupt_stops_optimization_with_a_graceful_shutdown(
    capsys, setup_minimal_everest_case
):
    with setup_minimal_everest_case(forward_model_sleep_time=15) as config_path:
        config = EverestConfig.load_file(config_path)

        def wait_and_kill():
            while True:
                status = get_experiment_status(config.storage_dir)
                if status and status.status == ExperimentState.running:
                    import _thread  # noqa: PLC0415

                    _thread.interrupt_main()
                    return
                time.sleep(1)

        thread = threading.Thread(target=wait_and_kill, args=())
        thread.start()

        with pytest.raises(SystemExit):
            start_everest(["everest", "run", "config.yml", "--skip-prompt"])

        out = capsys.readouterr().out

        assert (
            "The optimization will be run by an experiment server on this machine"
            in out
        )
        assert "KeyboardInterrupt" in out
        assert "The optimization will be stopped and the program will exit..." in out

        status = get_experiment_status(config.storage_dir)
        assert status.status == ExperimentState.stopped
