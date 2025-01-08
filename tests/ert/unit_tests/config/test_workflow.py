import os

import pytest

from ert.config import ConfigValidationError, Workflow


@pytest.mark.usefixtures("use_tmpdir")
def test_reading_non_existent_workflow_raises_config_error():
    with pytest.raises(ConfigValidationError, match="No such file or directory"):
        Workflow.from_file("/tmp/does_not_exist", None, {})
    os.mkdir("is_a_directory")
    with pytest.raises(ConfigValidationError, match="Is a directory"):
        Workflow.from_file("is_a_directory", None, {})
