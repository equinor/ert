from pathlib import Path
from unittest.mock import patch

from everest.bin.visualization_script import visualization_entry
from everest.detached import ServerStatus
from tests.everest.utils import capture_streams


@patch(
    "everest.bin.visualization_script.everserver_status",
    return_value={"status": ServerStatus.completed},
)
def test_visualization_entry(_, cached_example):
    config_path, config_file, _ = cached_example("math_func/config_advanced.yml")
    with capture_streams() as (out, _):
        visualization_entry([str(Path(config_path) / config_file)])
    assert "No visualization plugin installed!" in out.getvalue()
