import sys
from unittest.mock import PropertyMock, patch

import pytest

from everest.bin.visualization_script import visualization_entry
from everest.config import EverestConfig
from everest.detached import ServerStatus
from tests.everest.utils import capture_streams, relpath

CONFIG_PATH = relpath(
    "..", "..", "test-data", "everest", "math_func", "config_advanced.yml"
)
CACHED_SEBA_FOLDER = relpath("test_data", "cached_results_config_advanced")


@patch.object(
    EverestConfig,
    "optimization_output_dir",
    new_callable=PropertyMock,
    return_value=CACHED_SEBA_FOLDER,
)
@patch(
    "everest.bin.visualization_script.everserver_status",
    return_value={"status": ServerStatus.completed},
)
@pytest.mark.skipif(sys.version_info.major < 3, reason="requires python3 or higher")
def test_visualization_entry(
    opt_dir_mock,
    server_status_mock,
):
    with capture_streams() as (out, _):
        visualization_entry([CONFIG_PATH])
    assert "No visualization plugin installed!" in out.getvalue()
