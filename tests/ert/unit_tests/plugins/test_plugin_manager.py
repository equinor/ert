import json
import logging
from pathlib import Path

import pytest

import ert.plugins.hook_implementations
from ert.plugins import ErtPluginManager, ErtRuntimePlugins, plugin
from ert.trace import trace, tracer
from tests.ert.unit_tests.plugins import dummy_plugins
from tests.ert.unit_tests.plugins.dummy_plugins import PLUGIN_IP_ADDRESS, DummyFMStep


def test_no_plugins():
    pm = ErtPluginManager(plugins=[ert.plugins.hook_implementations])
    assert pm.get_help_links() == {"GitHub page": "https://github.com/equinor/ert"}
    assert pm.get_forward_model_configuration() == {}

    assert len(pm.forward_model_steps) > 0
    assert len(pm._get_config_workflow_jobs()) > 0


def test_with_plugins():
    pm = ErtPluginManager(plugins=[ert.plugins.hook_implementations, dummy_plugins])
    assert pm.get_help_links() == {
        "GitHub page": "https://github.com/equinor/ert",
        "test": "test",
        "test2": "test",
    }
    assert pm.get_forward_model_configuration() == {"FLOW": {"mpipath": "/foo"}}

    assert pm._get_config_workflow_jobs()["wf_job1"] == "dummy/path/wf_job1"
    assert pm._get_config_workflow_jobs()["wf_job2"] == "dummy/path/wf_job2"


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


def test_fm_config_with_empty_string_for_step():
    class SomePlugin:
        @plugin(name="foo")
        def forward_model_configuration():
            return {"foo": {"com": ""}}

    assert ErtPluginManager(plugins=[SomePlugin]).get_forward_model_configuration() == {
        "foo": {"com": ""}
    }


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


def test_workflows_merge():
    expected_result = {
        "wf_job1": "dummy/path/wf_job1",
        "wf_job2": "dummy/path/wf_job2",
    }
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
        logging.critical("I should write this to spam.log")  # noqa: LOG015
        assert "I should write this to spam.log" in Path("spam.log").read_text(
            encoding="utf-8"
        )


def test_add_span_processor():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    pm.add_span_processor_to_trace_provider()
    with tracer.start_as_current_span("span_1"):
        print("do_something")
        with tracer.start_as_current_span("span_2"):
            print("do_something_else")
    trace.get_tracer_provider().force_flush()
    span_info = "[" + dummy_plugins.span_output.getvalue().replace("}\n{", "},{") + "]"
    span_info = json.loads(span_info)
    span_info = {span["name"]: span for span in span_info}
    assert span_info["span_2"]["parent_id"] == span_info["span_1"]["context"]["span_id"]


def test_that_add_same_span_processor_twice_does_not_cause_duplicate_spans():
    pm = ErtPluginManager(plugins=[dummy_plugins])
    pm.add_span_processor_to_trace_provider()
    pm.add_span_processor_to_trace_provider()
    dummy_plugins.span_output.seek(0)
    dummy_plugins.span_output.truncate(0)
    with tracer.start_as_current_span("span_1"):
        print("do_something")
    trace.get_tracer_provider().force_flush()
    span_info = "[" + dummy_plugins.span_output.getvalue().replace("}\n{", "},{") + "]"
    span_info = json.loads(span_info)
    assert len(span_info) == 1


def test_that_forward_model_step_is_registered(tmpdir):
    with tmpdir.as_cwd():
        pm = ErtPluginManager(plugins=[dummy_plugins])
        assert pm.forward_model_steps == [DummyFMStep]


def test_that_plugin_manager_with_two_site_configurations_raises_error():
    class SiteOne:
        @plugin(name="foo")
        def site_configurations():
            return ErtRuntimePlugins(environment_variables={"a": "b"})

    class SiteTwo:
        @plugin(name="foo")
        def site_configurations():
            return ErtRuntimePlugins(environment_variables={"a": "c"})

    with pytest.raises(ValueError, match="Only one site configuration is allowed"):
        ErtPluginManager(plugins=[SiteOne, SiteTwo]).get_site_configurations()


def test_get_ip_address(monkeypatch, tmpdir):
    default_ip_address = "10.10.10.10"
    monkeypatch.setattr(
        "ert.shared.net_utils.get_ip_address", lambda: default_ip_address
    )
    with tmpdir.as_cwd():
        pm = ErtPluginManager(plugins=[dummy_plugins])
        default_plugin = ErtPluginManager(plugins=[])
        assert default_plugin.get_ip_address() != pm.get_ip_address()
        assert default_plugin.get_ip_address() == default_ip_address
        assert pm.get_ip_address() == PLUGIN_IP_ADDRESS
