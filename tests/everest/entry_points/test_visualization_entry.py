from pathlib import Path
from unittest.mock import MagicMock, patch

from everest.bin.visualization_script import visualization_entry
from tests.everest.utils import capture_streams, everest_config_with_defaults


@patch("everest.bin.visualization_script.run_plotter_gui")
@patch("everest.bin.visualization_script.ErtServerController")
@patch(
    "everest.bin.visualization_script.EverestStorage.get_everest_experiment",
    side_effect=StopIteration,
)
@patch(
    "everest.bin.visualization_script.EverestStorage.check_for_deprecated_seba_storage"
)
@patch(
    "everest.bin.visualization_script.LocalStorage.check_migration_needed",
    return_value=False,
)
def test_that_visualization_entry_prints_error_when_storage_has_no_experiment(
    check_migration,
    check_deprecated,
    get_experiment,
    ert_server,
    run_plotter_gui_mock,
    change_to_tmpdir,
):
    Path("config.yml").touch()
    everest_config_with_defaults(config_path="./config.yml").write_to_file("config.yml")

    with capture_streams() as (out, _):
        visualization_entry(["config.yml"])

    assert "At least one batch needs to be initialized" in out.getvalue()
    run_plotter_gui_mock.assert_not_called()


@patch("everest.bin.visualization_script.run_plotter_gui")
@patch("everest.bin.visualization_script.ErtServerController")
@patch(
    "everest.bin.visualization_script.EverestStorage.get_everest_experiment",
    return_value=MagicMock(ensembles_with_function_results=[]),
)
@patch(
    "everest.bin.visualization_script.EverestStorage.check_for_deprecated_seba_storage"
)
@patch(
    "everest.bin.visualization_script.LocalStorage.check_migration_needed",
    return_value=False,
)
def test_that_visualization_entry_prints_error_when_no_function_results(
    check_migration,
    check_deprecated,
    get_experiment,
    ert_server,
    run_plotter_gui_mock,
    change_to_tmpdir,
):
    Path("config.yml").touch()
    everest_config_with_defaults(config_path="./config.yml").write_to_file("config.yml")

    with capture_streams() as (out, _):
        visualization_entry(["config.yml"])

    assert "No data found in storage" in out.getvalue()
    run_plotter_gui_mock.assert_not_called()


@patch("everest.bin.visualization_script.run_plotter_gui")
@patch("everest.bin.visualization_script.ErtServerController")
@patch(
    "everest.bin.visualization_script.EverestStorage.get_everest_experiment",
    return_value=MagicMock(ensembles_with_function_results=["some_ensemble"]),
)
@patch(
    "everest.bin.visualization_script.EverestStorage.check_for_deprecated_seba_storage"
)
@patch(
    "everest.bin.visualization_script.LocalStorage.check_migration_needed",
    return_value=False,
)
def test_that_visualization_entry_opens_gui_when_data_is_present(
    check_migration,
    check_deprecated,
    get_experiment,
    ert_server,
    run_plotter_gui_mock,
    change_to_tmpdir,
):
    Path("config.yml").touch()
    everest_config_with_defaults(config_path="./config.yml").write_to_file("config.yml")

    visualization_entry(["config.yml"])

    run_plotter_gui_mock.assert_called_once()
