from os.path import exists
from unittest.mock import PropertyMock, patch

from seba_sqlite.snapshot import SebaSnapshot

from everest.bin.config_branch_script import config_branch_entry
from everest.config import EverestConfig
from everest.config_file_loader import load_yaml
from everest.config_keys import ConfigKeys as CK
from tests.utils import relpath, tmpdir

CONFIG_PATH = relpath("..", "examples", "math_func")
CONFIG_FILE = "config_advanced.yml"
CACHED_SEBA_FOLDER = relpath("test_data", "cached_results_config_advanced")


# @patch.object(EverestConfig, "optimization_output_dir", new_callable=PropertyMock)
@patch.object(
    EverestConfig,
    "optimization_output_dir",
    new_callable=PropertyMock,
    return_value=CACHED_SEBA_FOLDER,
)
@tmpdir(CONFIG_PATH)
def test_config_branch_entry(get_opt_output_dir_mock):
    new_config_file_name = "new_restart_config.yml"
    batch_id = 1

    config_branch_entry([CONFIG_FILE, new_config_file_name, "-b", str(batch_id)])

    get_opt_output_dir_mock.assert_called_once()
    assert exists(new_config_file_name)

    old_config = load_yaml(CONFIG_FILE)
    old_controls = old_config[CK.CONTROLS]

    assert CK.INITIAL_GUESS in old_controls[0]

    new_config = load_yaml(new_config_file_name)
    new_controls = new_config[CK.CONTROLS]

    assert CK.INITIAL_GUESS not in new_controls[0]
    assert len(new_controls) == len(old_controls)
    assert len(new_controls[0][CK.VARIABLES]) == len(old_controls[0][CK.VARIABLES])

    opt_controls = {}

    snapshot = SebaSnapshot(CACHED_SEBA_FOLDER)
    for opt_data in snapshot._optimization_data():
        if opt_data.batch_id == batch_id:
            opt_controls = opt_data.controls

    new_controls_initial_guesses = set(
        [var[CK.INITIAL_GUESS] for var in new_controls[0][CK.VARIABLES]]
    )
    opt_control_val_for_batch_id = set([v for k, v in opt_controls.items()])

    assert new_controls_initial_guesses == opt_control_val_for_batch_id
