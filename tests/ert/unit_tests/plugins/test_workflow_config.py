import logging
from unittest.mock import Mock

import pytest

import ert
import ert.config.workflow_job
from ert import ErtScript
from ert.plugins import workflow_config


def test_workflow_config_duplicate_log_message(caplog, monkeypatch):
    def get_mock_config():
        workflow_mock = Mock(spec=ert.config.workflow_job.ErtScriptWorkflow)
        workflow_mock.name = "same_name"
        return workflow_mock

    config = workflow_config.WorkflowConfigs()

    # Create duplicate workflows
    workflows = [get_mock_config(), get_mock_config()]

    monkeypatch.setattr(config, "_workflows", workflows)
    with caplog.at_level(logging.INFO):
        config.get_workflows()
    assert "Duplicate workflow name: same_name, skipping" in caplog.text


@pytest.mark.parametrize(
    "name, expected", [(None, "default_name"), ("some_name", "some_name")]
)
def test_workflow_config_init_name(monkeypatch, name, expected):
    mock_func = ErtScript
    mock_func.__name__ = "default_name"

    monkeypatch.setattr(ert.config.workflow_job.WorkflowJob, "__post_init__", Mock())
    workflow = ert.config.workflow_job.ErtScriptWorkflow(mock_func, name=name)

    assert workflow.name == expected
    assert workflow.ert_script == mock_func
