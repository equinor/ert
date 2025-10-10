import argparse
import logging
from unittest.mock import Mock

import pytest

from ert import ErtScript
from ert.config import workflow_config
from ert.config.workflow_config import LegacyWorkflowConfigs


def test_workflow_config_duplicate_log_message(caplog, monkeypatch):
    def get_mock_config():
        workflow_mock = Mock()
        workflow_mock.name = "same_name"
        return workflow_mock

    config = workflow_config.WorkflowConfigs()

    # Create duplicate workflows
    workflows = [get_mock_config(), get_mock_config()]

    monkeypatch.setattr(config, "_workflows", workflows)
    with caplog.at_level(logging.INFO):
        config.get_workflows()
    assert "Duplicate workflow name: same_name, skipping" in caplog.text


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_config_can_set_name_through_dunder_field():
    class MockedErtScript(ErtScript):
        pass

    configs = LegacyWorkflowConfigs()
    _ = configs.add_workflow(ert_script=MockedErtScript)

    assert configs.get_workflows()["MockedErtScript"].ert_script is MockedErtScript


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_config_can_set_name_through_parameter():
    class MockedErtScript(ErtScript):
        pass

    configs = LegacyWorkflowConfigs()
    _ = configs.add_workflow(ert_script=MockedErtScript, name="parameter_name")

    assert configs.get_workflows()["parameter_name"].ert_script is MockedErtScript


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_config_can_set_name_through_add_workflow_return_value():
    configs = LegacyWorkflowConfigs()

    class MockedErtScript(ErtScript):
        pass

    workflow = configs.add_workflow(ert_script=MockedErtScript)
    workflow.name = "name"

    assert configs.get_workflows()["name"].ert_script is MockedErtScript


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_config_description_defaults_to_its_docstring():
    class MockedErtScript(ErtScript):
        """description"""

    configs = LegacyWorkflowConfigs()
    _ = configs.add_workflow(ert_script=MockedErtScript)

    assert configs.get_workflows()["MockedErtScript"].description == "description"


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_config_can_set_description_through_parameter():
    class MockedErtScript(ErtScript):
        pass

    configs = LegacyWorkflowConfigs()
    _ = configs.add_workflow(ert_script=MockedErtScript, description="description")

    assert configs.get_workflows()["MockedErtScript"].description == "description"


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_config_can_set_description_through_add_workflow_return_value():
    configs = LegacyWorkflowConfigs()
    workflow = configs.add_workflow(ert_script=ErtScript)
    workflow.description = "description"

    assert configs.get_workflows()["ErtScript"].description == "description"


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_config_can_set_examples_through_parameter():
    configs = LegacyWorkflowConfigs()
    workflow = configs.add_workflow(ert_script=ErtScript, examples="examples")
    workflow.name = "name"

    assert configs.get_workflows()["name"].examples == "examples"


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_config_can_set_examples_through_add_workflow_return_value():
    configs = LegacyWorkflowConfigs()
    workflow = configs.add_workflow(ert_script=ErtScript)
    workflow.examples = "examples"

    assert configs.get_workflows()["ErtScript"].examples == "examples"


@pytest.mark.filterwarnings("ignore:Use of legacy_ertscript_workflow is deprecated")
def test_legacy_workflow_configs_sets_parser_through_add_workflow_return_value():
    configs = LegacyWorkflowConfigs()
    workflow = configs.add_workflow(ert_script=ErtScript, name="name")

    def create_parser():
        return argparse.ArgumentParser()

    workflow.parser = create_parser
    assert configs.parsers[workflow.name] is create_parser
