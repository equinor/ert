import json
import logging
import tempfile
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider

import ert.plugins.hook_implementations
from ert import plugin
from ert.plugins import ErtPluginManager
from tests.ert.unit_tests.plugins import dummy_plugins
from tests.ert.unit_tests.plugins.dummy_plugins import (
    DummyFMStep,
)


def test_no_plugins():
    pm = ErtPluginManager(plugins=[ert.plugins.hook_implementations])
    assert pm.get_help_links() == {"GitHub page": "https://github.com/equinor/ert"}
    assert pm.get_flow_config_path() is None
    assert pm.get_ecl100_config_path() is None
    assert pm.get_ecl300_config_path() is None
    assert pm.get_forward_model_configuration() == {}

    assert len(pm.forward_model_steps) > 0
    assert len(pm._get_config_workflow_jobs()) > 0

    assert pm._site_config_lines() == [
        "-- Content below originated from ert (site_config_lines)",
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_SYSTEM LOCAL",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
    ]


def test_with_plugins():
    pm = ErtPluginManager(plugins=[ert.plugins.hook_implementations, dummy_plugins])
    assert pm.get_help_links() == {
        "GitHub page": "https://github.com/equinor/ert",
        "test": "test",
        "test2": "test",
    }
    assert pm.get_flow_config_path() == "/dummy/path/flow_config.yml"
    assert pm.get_ecl100_config_path() == "/dummy/path/ecl100_config.yml"
    assert pm.get_ecl300_config_path() == "/dummy/path/ecl300_config.yml"
    assert pm.get_forward_model_configuration() == {"FLOW": {"mpipath": "/foo"}}

    assert pm.get_installable_jobs()["job1"] == "/dummy/path/job1"
    assert pm.get_installable_jobs()["job2"] == "/dummy/path/job2"
    assert pm._get_config_workflow_jobs()["wf_job1"] == "/dummy/path/wf_job1"
    assert pm._get_config_workflow_jobs()["wf_job2"] == "/dummy/path/wf_job2"

    assert pm._site_config_lines() == [
        "-- Content below originated from ert (site_config_lines)",
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_SYSTEM LOCAL",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
        "-- Content below originated from dummy (site_config_lines)",
        "JOB_SCRIPT job_dispatch_dummy.py",
        "QUEUE_OPTION LOCAL MAX_RUNNING 2",
    ]


def test_fm_config_with_empty_config():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {}

    assert (
        ErtPluginManager(plugins=[SomePlugin]).get_forward_model_configuration() == {}
    )


def test_fm_config_with_empty_config_for_step():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo": {}}

    assert (
        ErtPluginManager(plugins=[SomePlugin]).get_forward_model_configuration() == {}
    )


def test_fm_config_merges_data_for_step():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo": {"com": 3}}

    class OtherPlugin:
        @plugin(name="bar")
        def forward_model_configuration():
            return {"foo": {"bar": 2}}

    assert ErtPluginManager(
        plugins=[SomePlugin, OtherPlugin]
    ).get_forward_model_configuration() == {"foo": {"com": 3, "bar": 2}}


def test_fm_config_multiple_steps():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo100": {"com": 3}}

    class OtherPlugin:
        @plugin(name="bar")
        def forward_model_configuration():
            return {"foo200": {"bar": 2}}

    assert ErtPluginManager(
        plugins=[SomePlugin, OtherPlugin]
    ).get_forward_model_configuration() == {"foo100": {"com": 3}, "foo200": {"bar": 2}}


def test_fm_config_conflicting_config():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo100": {"com": "from_someplugin"}}

    class OtherPlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo100": {"com": "from_otherplugin"}}

    with pytest.raises(RuntimeError, match="Duplicate configuration"):
        ErtPluginManager(
            plugins=[SomePlugin, OtherPlugin]
        ).get_forward_model_configuration()


def test_fm_config_with_repeated_keys_different_fm_step():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo1": {"bar": "1"}}

    class OtherPlugin:
        @plugin(name="foo2")
        def forward_model_configuration():
            return {"foo2": {"bar": "2"}}

    assert ErtPluginManager(
        plugins=[SomePlugin, OtherPlugin]
    ).get_forward_model_configuration() == {"foo1": {"bar": "1"}, "foo2": {"bar": "2"}}


def test_fm_config_with_repeated_keys_with_different_case():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo": {"bar": "lower", "BAR": "higher"}}

    with pytest.raises(RuntimeError, match="Duplicate configuration"):
        ErtPluginManager(plugins=[SomePlugin]).get_forward_model_configuration()


def test_fm_config_with_wrong_type():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return 1

    with pytest.raises(TypeError, match="foo did not return a dict"):
        ErtPluginManager(plugins=[SomePlugin]).get_forward_model_configuration()


def test_fm_config_with_wrong_steptype():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {1: {"bar": "1"}}

    with pytest.raises(TypeError, match="foo did not provide dict"):
        ErtPluginManager(plugins=[SomePlugin]).get_forward_model_configuration()


def test_fm_config_with_wrong_subtype():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo100": 1}

    with pytest.raises(TypeError, match="foo did not provide dict"):
        ErtPluginManager(plugins=[SomePlugin]).get_forward_model_configuration()


def test_fm_config_with_wrong_keytype():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo100": {1: "bar"}}

    with pytest.raises(TypeError, match="foo did not provide dict"):
        ErtPluginManager(plugins=[SomePlugin]).get_forward_model_configuration()


def test_job_documentation():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    expected = {
        "job1": {
            "config_file": "/dummy/path/job1",
            "source_package": "dummy",
            "source_function_name": "installable_jobs",
            "description": "job description",
            "examples": "example 1 and example 2",
            "category": "test.category.for.job",
        },
        "job2": {
            "config_file": "/dummy/path/job2",
            "source_package": "dummy",
            "source_function_name": "installable_jobs",
        },
    }
    assert pm.get_documentation_for_jobs() == expected


def test_workflows_merge(monkeypatch, tmpdir):
    expected_result = {
        "wf_job1": "/dummy/path/wf_job1",
        "wf_job2": "/dummy/path/wf_job2",
        "some_func": str(tmpdir / "SOME_FUNC"),
    }
    tempfile_mock = Mock(return_value=tmpdir)
    monkeypatch.setattr(tempfile, "mkdtemp", tempfile_mock)
    pm = ErtPluginManager(plugins=[dummy_plugins])
    result = pm.get_installable_workflow_jobs()
    assert result == expected_result


def test_workflows_merge_duplicate(caplog):
    pm = ErtPluginManager(plugins=[dummy_plugins])

    dict_1 = {"some_job": "/a/path"}
    dict_2 = {"some_job": "/a/path"}

    with caplog.at_level(logging.INFO):
        result = pm._merge_internal_jobs(dict_1, dict_2)

    assert result == {"some_job": "/a/path"}

    assert (
        "Duplicate key: some_job in workflow hook implementations, "
        "config path 1: /a/path, config path 2: /a/path"
    ) in caplog.text


def test_add_logging_handle(tmpdir):
    with tmpdir.as_cwd():
        pm = ErtPluginManager(plugins=[dummy_plugins])
        pm.add_logging_handle_to_root(logging.getLogger())
        logging.critical("I should write this to spam.log")
        with open("spam.log", encoding="utf-8") as fin:
            result = fin.read()
        assert "I should write this to spam.log" in result


def test_add_span_processor():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    tracer_provider = TracerProvider()
    tracer = tracer_provider.get_tracer("ert.tests")
    pm.add_span_processor_to_trace_provider(tracer_provider)
    with tracer.start_as_current_span("span_1"):
        print("do_something")
        with tracer.start_as_current_span("span_2"):
            print("do_something_else")
    tracer_provider.force_flush()
    span_info = "[" + dummy_plugins.span_output.getvalue().replace("}\n{", "},{") + "]"
    span_info = json.loads(span_info)
    span_info = {span["name"]: span for span in span_info}
    assert span_info["span_2"]["parent_id"] == span_info["span_1"]["context"]["span_id"]


def test_that_forward_model_step_is_registered(tmpdir):
    with tmpdir.as_cwd():
        pm = ErtPluginManager(plugins=[dummy_plugins])
        assert pm.forward_model_steps == [DummyFMStep]
