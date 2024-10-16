import inspect
import logging
import os
from unittest.mock import Mock

import pytest

from ert.plugins import workflow_config


@pytest.mark.parametrize(
    "workflows, expected",
    [
        ([], {}),
        ([("a", "/a/path")], {"a": "/a/path"}),
        (
            [("a", "/a/path"), ("b", "/another/path")],
            {"a": "/a/path", "b": "/another/path"},
        ),
    ],
)
def test_workflow_configs(workflows, expected, monkeypatch):
    def get_mock_config(name, config_path):
        workflow_mock = Mock(spec=workflow_config.WorkflowConfig)
        workflow_mock.name = name
        workflow_mock.config_path = config_path
        return workflow_mock

    config = workflow_config.WorkflowConfigs()

    workflows = [get_mock_config(name, config_path) for name, config_path in workflows]

    monkeypatch.setattr(config, "_workflows", workflows)
    assert config.get_workflows() == expected


def test_workflow_config_duplicate_log_message(caplog, monkeypatch):
    def get_mock_config():
        workflow_mock = Mock(spec=workflow_config.WorkflowConfig)
        workflow_mock.name = "same_name"
        workflow_mock.config_path = "/duplicate/path"
        workflow_mock.function_dir = "func_dir"
        return workflow_mock

    config = workflow_config.WorkflowConfigs()

    # Create duplicate workflows
    workflows = [get_mock_config(), get_mock_config()]

    monkeypatch.setattr(config, "_workflows", workflows)
    with caplog.at_level(logging.INFO):
        config.get_workflows()
    assert "Duplicate workflow name: same_name, skipping func_dir" in caplog.text


@pytest.mark.parametrize(
    "name, expected", [(None, "default_name"), ("some_name", "some_name")]
)
def test_workflow_config_init_name(tmpdir, monkeypatch, name, expected):
    mock_func = Mock()
    mock_func.__name__ = "default_name"

    inspect_mock = Mock(return_value="/some/path")
    monkeypatch.setattr(inspect, "getfile", inspect_mock)
    workflow = workflow_config.WorkflowConfig(mock_func, tmpdir.strpath, name=name)

    assert workflow.name == expected


def test_workflow_config_init_path(tmpdir, monkeypatch):
    mock_func = Mock()
    mock_func.__name__ = "default_name"

    inspect_mock = Mock(return_value="/some/path")
    monkeypatch.setattr(inspect, "getfile", inspect_mock)
    workflow = workflow_config.WorkflowConfig(mock_func, tmpdir.strpath)

    assert workflow.function_dir == "/some/path"


def test_workflow_config_write_workflow_config(tmpdir, monkeypatch):
    expected_config = "INTERNAL      True\nSCRIPT        /some/path"
    mock_func = Mock()
    mock_func.__name__ = "default_name"

    inspect_mock = Mock(return_value="/some/path")
    monkeypatch.setattr(inspect, "getfile", inspect_mock)
    workflow_config.WorkflowConfig(mock_func, tmpdir.strpath)

    with tmpdir.as_cwd():
        assert os.path.isfile("DEFAULT_NAME")

        with open("DEFAULT_NAME", encoding="utf-8") as fin:
            content = fin.read()

        assert content == expected_config
