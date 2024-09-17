from unittest.mock import Mock

import pytest

from everest.config import EverestConfig
from everest.queue_driver import queue_driver
from everest.queue_driver.queue_driver import _extract_queue_system


@pytest.mark.parametrize(
    "input_config,queue_system,expected_result",
    [
        (
            EverestConfig.with_defaults(**{}),
            "local",
            {"QUEUE_SYSTEM": "LOCAL", "QUEUE_OPTION": []},
        ),
        (
            EverestConfig.with_defaults(**{"simulator": {"queue_system": "lsf"}}),
            "lsf",
            {"QUEUE_SYSTEM": "LSF", "QUEUE_OPTION": []},
        ),
        (
            EverestConfig.with_defaults(**{"simulator": {"queue_system": "slurm"}}),
            "slurm",
            {"QUEUE_SYSTEM": "SLURM", "QUEUE_OPTION": []},
        ),
    ],
)
def test_extract_queue_system(monkeypatch, input_config, queue_system, expected_result):
    extract_options_mock = Mock(return_value=[])
    monkeypatch.setattr(
        queue_driver,
        "_extract_ert_queue_options_from_simulator_config",
        extract_options_mock,
    )
    ert_config = {}
    _extract_queue_system(input_config, ert_config)
    assert ert_config == expected_result
    extract_options_mock.assert_called_once_with(input_config.simulator, queue_system)
