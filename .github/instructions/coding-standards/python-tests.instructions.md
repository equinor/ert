---
applyTo: '**/*.py'
description: 'Python test code authoring conventions'
---

# Python Test Instructions

Conventions for Python test code.

## Test Framework

* Use `pytest` for writing and running tests.
* Tests should be self-contained, no shared mutable state, fixtures for clean environments, deterministic, fast.
* Test observable behavior, not implementation details; a test should survive a refactor that preserves behavior.
* Assertions should produce clear failure messages; compare concrete values so pytest can show the diff (`assert result == 42`, not `assert is_valid(result)`) and add a message when the comparison alone does not explain the failure.

## Mocking Libraries

| Library                        | Usage                                                  |
|--------------------------------|--------------------------------------------------------|
| pytest-mock (`mocker` fixture) | Preferred for new projects and test migrations         |
| monkeypatch                    | Acceptable for simple attribute/environment patching   |
| unittest.mock (direct import)  | Existing projects only; migrate to mocker when editing |

### When to Use mocker vs monkeypatch

* `mocker.patch()` — replacing functions, methods, classes, or module attributes with controlled return values or side effects; verifying call counts and arguments.
* `monkeypatch.setattr()` — simple attribute overrides (constants, config values, environment variables) where return tracking is not needed.
* Direct `MagicMock()` import — acceptable for constructing pure test data stubs (mock objects used as constructor arguments, not as spy/assert targets).

## Test Naming

Rationale: `pytest --collect-only tests/` output should be self-explanatory.

Rely on file/directory name for context rather than repeating it in the test name.

Test method format: `test_that_<behavior_or_invariant>`, or `test_when_<condition>_<outcome>` when describing a conditional sequence.

Good examples from this repository:

```text
test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior
test_that_posterior_generalized_variance_increases_in_cutoff
test_that_setenv_does_not_expand_envvar
test_that_new_line_can_be_escaped
test_that_unknown_queue_option_gives_error_message
test_that_config_path_substitution_is_the_name_of_the_configs_directory
test_when_forward_model_contains_multiple_steps_just_one_checksum_status_is_given
```

Poor examples (vague or not behavior-focused):

```text
test_color_always
test_legends
test_result_success
test_print_progress
test_bad_user_config_file_error_message
```

Prefer one assertion per test. Related assertions validating the same behavior are acceptable. Do not verify logger mocks.

Use `@pytest.mark.parametrize` for data-driven tests with multiple input/output combinations.
If setting multiple cases in `@pytest.mark.parametrize`, use `id=` for meaningful case names.

Test names including `works`, `correctly`, `as_expected`, `are_handled`, `handles`, `success`, `failure` are explicitly banned.
Replace them with the explicit condition or outcome, e.g. `test_that_double_comments_are_handled` → `test_that_double_comments_are_ignored`.

## Test Organization

* File naming mirrors module under test with `test_` prefix (for example, `_read_summary.py` → `test_read_summary.py`).
* Fixtures in `conftest.py` when shared across multiple test files.
* Class-based grouping optional; use when tests share setup logic.
* Group test methods by behavior, alphabetically within groups.
* Common mock setup in fixtures or class-level setup; specific setup in individual tests.

## Test Categories

* `unit_tests` must be exceptionally fast/reliable; mark `slow`, `unreliable`, `high_utilization` otherwise.
* `ui_tests` test user-visible workflows (actions and resulting UI/CLI state); do not duplicate logic assertions already covered by unit tests.
* `performance_tests` guard runtime/memory.
* Fuzz/hypothesis coverage expected for data-integrity code (`ert.storage`, `ert.field_utils`, `ert.config._read_summary`).

## Hypothesis

Hypothesis is used for property-based testing, allowing you to define properties that should hold for a wide range of inputs.
The library will generate test cases to try and falsify these properties.

```python
# Before — example-based
def reverse(s):
    return s[::-1]

def test_reverse():
    assert reverse("abc") == "cba"
    assert reverse("") == ""
    assert reverse("racecar") == "racecar"
    assert reverse("a") == "a"


# After — property-based with hypothesis
from hypothesis import given, strategies as st

@given(st.text())
def test_that_reverse_is_self_inverse(s):
    assert reverse(reverse(s)) == s

@given(st.lists(st.integers()))
def test_that_reverse_preserves_length(xs):
    assert len(xs) == len(reverse(xs))
```

## pytest-mock Patterns

The `mocker` fixture from pytest-mock replaces direct `unittest.mock` usage. These patterns show each migration.

### mocker.patch() replacing @patch decorator

```python
# Before — unittest.mock
from unittest.mock import patch


@patch("myapp.service.fetch_data")
def test_that_process_returns_the_fetched_value(mock_fetch):
    mock_fetch.return_value = {"key": "value"}
    result = process()
    assert result == "value"


# After — pytest-mock
def test_that_process_returns_the_fetched_value(mocker):
    mock_fetch = mocker.patch("myapp.service.fetch_data", return_value={"key": "value"})
    result = process()
    assert result == "value"
    mock_fetch.assert_called_once()
```

### mocker.patch() replacing with patch() context manager

```python
# Before — unittest.mock
from unittest.mock import patch


def test_that_send_request_returns_the_endpoint_status_code():
    with patch("myapp.client.post") as mock_post:
        mock_post.return_value.status_code = 200
        response = send_request()
    assert response.status_code == 200


# After — pytest-mock
def test_that_send_request_returns_the_endpoint_status_code(mocker):
    mock_post = mocker.patch("myapp.client.post")
    mock_post.return_value.status_code = 200
    response = send_request()
    assert response.status_code == 200
```

### mocker.patch.dict() replacing @patch.dict

```python
# Before — unittest.mock
from unittest.mock import patch


@patch.dict("os.environ", {"API_KEY": "test-key"})
def test_that_api_key_is_read_from_the_environment():
    config = load_config()
    assert config.api_key == "test-key"


# After — pytest-mock
def test_that_api_key_is_read_from_the_environment(mocker):
    mocker.patch.dict("os.environ", {"API_KEY": "test-key"})
    config = load_config()
    assert config.api_key == "test-key"
```

### mocker.patch.object() replacing patch.object()

```python
# Before — unittest.mock
from unittest.mock import patch

from myapp.service import DataService


@patch.object(DataService, "connect")
def test_that_connect_returns_true_when_the_connection_is_established(mock_connect):
    mock_connect.return_value = True
    svc = DataService()
    assert svc.connect() is True


# After — pytest-mock
from myapp.service import DataService


def test_that_connect_returns_true_when_the_connection_is_established(mocker):
    mock_connect = mocker.patch.object(DataService, "connect", return_value=True)
    svc = DataService()
    assert svc.connect() is True
    mock_connect.assert_called_once()
```

### mocker.MagicMock() and mocker.AsyncMock() for spy targets

Use `mocker.MagicMock()` and `mocker.AsyncMock()` when constructing mock objects that serve as spy targets for call assertion:

```python
def test_that_handle_passes_the_request_to_the_processor(mocker):
    mock_processor = mocker.MagicMock()
    handler = RequestHandler(processor=mock_processor)
    handler.handle({"id": 1})
    mock_processor.process.assert_called_once_with({"id": 1})


async def test_that_async_handle_awaits_the_processor_with_the_request(mocker):
    mock_processor = mocker.AsyncMock()
    handler = AsyncRequestHandler(processor=mock_processor)
    await handler.handle({"id": 1})
    mock_processor.process.assert_awaited_once_with({"id": 1})
```

### Direct MagicMock() import for test data stubs

Direct `MagicMock()` import stays as-is when constructing pure test data stubs that are not spy/assert targets:

```python
from unittest.mock import MagicMock


def test_that_the_output_formatter_accepts_any_writer():
    stub_writer = MagicMock()
    stub_writer.encoding = "utf-8"
    formatter = OutputFormatter(writer=stub_writer)

    result = formatter.format("hello")

    assert result == "hello"
```

## Complete Example

A full test module using the mocker fixture and naming conventions:

```python
import pytest

from myapp.processor import DataProcessor
from myapp.service import DataService


@pytest.fixture()
def mock_service(mocker):
    return mocker.patch.object(DataService, "fetch", return_value={"status": "ok", "value": 42})


@pytest.fixture()
def processor():
    return DataProcessor(service=DataService())


def test_that_process_returns_the_fetched_value(processor, mock_service):
    result = processor.process()

    assert result == 42


def test_that_process_fetches_from_the_service_exactly_once(processor, mock_service):
    processor.process()

    mock_service.assert_called_once()


def test_that_a_failing_fetch_propagates_the_connection_error(processor, mocker):
    mocker.patch.object(DataService, "fetch", side_effect=ConnectionError("timeout"))

    with pytest.raises(ConnectionError, match="timeout"):
        processor.process()


@pytest.mark.parametrize(
    ("status", "value"),
    [
        pytest.param("ok", 42, id="ok_status_returns_the_fetched_value"),
        pytest.param("pending", 0, id="pending_status_returns_zero"),
    ],
)
def test_that_process_returns_a_value_depending_on_the_fetched_status(mocker, status, value):
    mocker.patch.object(DataService, "fetch", return_value={"status": status, "value": value})
    processor = DataProcessor(service=DataService())

    result = processor.process()

    assert result == value
```
