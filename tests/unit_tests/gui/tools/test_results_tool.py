from unittest.mock import MagicMock

import pytest

from ert.gui.tools.load_results import LoadResultsTool


@pytest.mark.parametrize(
    "run_path, expected",
    [
        ("valid_%d", True),
        ("valid_%d/iter_%d", True),
        ("invalid_%d_%d_%d", False),
        ("invalid", False),
    ],
)
def test_results_tool_valid_runpath(run_path, expected):
    facade_mock = MagicMock()
    facade_mock.run_path = run_path
    tool = LoadResultsTool(facade_mock)
    assert tool.is_valid_run_path() is expected
