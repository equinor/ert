import pytest
from pydantic import ValidationError

from everest.config.forward_model_config import ForwardModelStepConfig


def test_forward_model_step_config_valid():
    config = ForwardModelStepConfig(
        job="example_job",
        results={"type": "summary", "file_name": "output.txt"},
    )
    assert config.results.type == "summary"


def test_forward_model_step_config_missing_type():
    expected_substring = (
        "Missing required field 'type' in 'results'. This field is needed to "
        "determine the correct result schema (e.g., 'gen_data' or 'summary')."
        " Please include a 'type' key in the 'results' section."
    )

    with pytest.raises(ValidationError) as exc_info:
        ForwardModelStepConfig(job="example_job", results={"file_name": "output.txt"})
    assert expected_substring in str(exc_info.value)
