import os
import textwrap
from contextlib import ExitStack as does_not_raise
from pathlib import Path

import pytest
from hypothesis import given, strategies

from ert.config import ConfigValidationError, ExecutableWorkflow, Workflow


@pytest.mark.usefixtures("use_tmpdir")
def test_reading_non_existent_workflow_raises_config_error():
    with pytest.raises(ConfigValidationError, match="No such file or directory"):
        Workflow.from_file("does_not_exist", None, {})
    os.mkdir("is_a_directory")
    with pytest.raises(ConfigValidationError, match="Is a directory"):
        Workflow.from_file("is_a_directory", None, {})


@pytest.mark.usefixtures("use_tmpdir")
def test_that_failure_in_parsing_workflow_gives_config_validation_error():
    Path("workflow").write_text("DEFINE\n", encoding="utf-8")
    with pytest.raises(
        ConfigValidationError, match=r"DEFINE must have .* arguments"
    ) as err:
        _ = Workflow.from_file("workflow", None, {})
    assert os.path.abspath(err.value.errors[0].filename) == os.path.abspath("workflow")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_substitution_happens_in_workflow():
    Path("workflow").write_text("JOB <A> <B>\n", encoding="utf-8")

    job = ExecutableWorkflow(
        executable="echo",
        name="JOB",
        min_args=None,
        max_args=None,
        arg_types=[],
    )
    wf = Workflow.from_file(
        "workflow",
        {
            "<A>": "a",
            "<B>": "b",
        },
        {"JOB": job},
    )
    assert wf.cmd_list == [(job, ["a", "b"])]


def get_workflow_job(name):
    return ExecutableWorkflow(
        executable="echo",
        name=name,
        min_args=None,
        max_args=None,
        arg_types=[],
    )


@pytest.mark.usefixtures("use_tmpdir")
@given(
    strategies.lists(
        strategies.sampled_from(
            [
                "foo",
                "bar",
                "baz",
            ]
        ),
        min_size=1,
        max_size=20,
    )
)
def test_that_multiple_workflow_jobs_are_ordered_correctly(order):
    Path("workflow").write_text("\n".join(order), encoding="utf-8")

    foo = get_workflow_job("foo")
    bar = get_workflow_job("bar")
    baz = get_workflow_job("baz")

    wf = Workflow.from_file(
        src_file="workflow",
        context=None,
        job_dict={
            "foo": foo,
            "bar": bar,
            "baz": baz,
        },
    )

    assert [x[0].name for x in wf.cmd_list] == order


@pytest.mark.usefixtures("use_tmpdir")
def test_that_redefine_in_workflow_overwrites_in_subsequent_lines():
    Path("workflow").write_text(
        textwrap.dedent("""
            DEFINE <A> 1
            foo <A>
            bar <A>
            DEFINE <A> 3
            foo <A>
            baz <A>"""),
        encoding="utf-8",
    )

    foo = get_workflow_job("foo")
    bar = get_workflow_job("bar")
    baz = get_workflow_job("baz")

    wf = Workflow.from_file(
        src_file="workflow",
        context=None,
        job_dict={
            "foo": foo,
            "bar": bar,
            "baz": baz,
        },
    )

    commands = [(name, args[0]) for (name, args) in wf.cmd_list]

    assert commands == [(foo, "1"), (bar, "1"), (foo, "3"), (baz, "3")]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_unknown_jobs_gives_error():
    Path("workflow").write_text(
        "boo <A>\nkingboo <A>",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigValidationError, match="Job with name: kingboo is not recognized"
    ):
        Workflow.from_file(
            src_file="workflow",
            context=None,
            job_dict={"boo": get_workflow_job("boo")},
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("config", "expectation"),
    [
        (
            "WORKFLOW",
            pytest.raises(ConfigValidationError, match="not have enough arguments"),
        ),
        (
            "WORKFLOW arg_1",
            does_not_raise(),
        ),
        (
            "WORKFLOW arg_1 arg_2",
            does_not_raise(),
        ),
        (
            "WORKFLOW arg_1 arg_2 arg_3",
            pytest.raises(ConfigValidationError, match="too many arguments"),
        ),
    ],
)
@pytest.mark.parametrize(("min_args", "max_args"), [(1, 2), (None, None)])
def test_args_validation(config, expectation, min_args, max_args):
    Path("workflow").write_text(config, encoding="utf-8")
    if min_args is None and max_args is None:
        expectation = does_not_raise()
    with expectation:
        Workflow.from_file(
            src_file="workflow",
            context=None,
            job_dict={
                "WORKFLOW": ExecutableWorkflow(
                    executable="echo",
                    name="WORKFLOW",
                    min_args=min_args,
                    max_args=max_args,
                    arg_types=[],
                ),
            },
        )
