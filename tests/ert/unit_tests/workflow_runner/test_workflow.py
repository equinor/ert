import os
from contextlib import ExitStack as does_not_raise

import pytest
from hypothesis import given, strategies

from ert.config import ConfigValidationError, Workflow, WorkflowJob
from ert.substitution_list import SubstitutionList
from ert.workflow_runner import WorkflowRunner

from .workflow_common import WorkflowCommon


def get_workflow_job(name):
    return WorkflowJob(
        name=name,
        internal=False,
        min_args=None,
        max_args=None,
        arg_types=[],
        executable=None,
        script=None,
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow():
    WorkflowCommon.createExternalDumpJob()

    dump_job = WorkflowJob.from_file("dump_job", name="DUMP")

    with pytest.raises(ConfigValidationError, match="Could not open config_file"):
        _ = WorkflowJob.from_file("knock_job", name="KNOCK")

    workflow = Workflow.from_file("dump_workflow", None, {"DUMP": dump_job})

    assert len(workflow) == 2

    job, args = workflow[0]
    assert args[0] == "dump1"
    assert args[1] == "dump_text_1"

    job, args = workflow[1]
    assert job.name == "DUMP"


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_run():
    WorkflowCommon.createExternalDumpJob()

    dump_job = WorkflowJob.from_file("dump_job", name="DUMP")

    context = SubstitutionList()
    context["<PARAM>"] = "text"

    workflow = Workflow.from_file("dump_workflow", context, {"DUMP": dump_job})

    assert len(workflow) == 2

    WorkflowRunner(workflow).run_blocking()

    with open("dump1", "r", encoding="utf-8") as f:
        assert f.read() == "dump_text_1"

    with open("dump2", "r", encoding="utf-8") as f:
        assert f.read() == "dump_text_2"


@pytest.mark.usefixtures("use_tmpdir")
def test_failing_workflow_run():
    with pytest.raises(ConfigValidationError, match="does not exist"):
        _ = Workflow.from_file("the_file_name.ert", None, {})


@pytest.mark.usefixtures("use_tmpdir")
def test_that_failure_in_parsing_workflow_gives_config_validation_error():
    with open("workflow", "w", encoding="utf-8") as f:
        f.write("DEFINE\n")
    with pytest.raises(
        ConfigValidationError, match="DEFINE must have .* arguments"
    ) as err:
        _ = Workflow.from_file("workflow", None, {})
    assert os.path.abspath(err.value.errors[0].filename) == os.path.abspath("workflow")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_substitution_happens_in_workflow():
    with open("workflow", "w", encoding="utf-8") as f:
        f.write("JOB <A> <B>\n")
    substlist = SubstitutionList()
    substlist["<A>"] = "a"
    substlist["<B>"] = "b"
    job = WorkflowJob(
        name="JOB",
        internal=False,
        min_args=None,
        max_args=None,
        arg_types=[],
        executable="echo",
        script=None,
    )
    wf = Workflow.from_file(
        "workflow",
        substlist,
        {"JOB": job},
    )
    assert wf.cmd_list == [(job, ["a", "b"])]


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
    with open("workflow", "w", encoding="utf-8") as f:
        f.write("\n".join(order))

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
def test_that_multiple_workflow_jobs_with_redefines_are_ordered_correctly():
    with open("workflow", "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "DEFINE <A> 1",
                    "foo <A>",
                    "bar <A>",
                    "DEFINE <A> 3",
                    "foo <A>",
                    "baz <A>",
                ]
            )
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
    with open("workflow", "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "boo <A>",
                    "kingboo <A>",
                ]
            )
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
    "config, expectation",
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
@pytest.mark.parametrize("min_args, max_args", [(1, 2), (None, None)])
def test_args_validation(config, expectation, min_args, max_args):
    with open("workflow", "w", encoding="utf-8") as f:
        f.write(config)
    if min_args is None and max_args is None:
        expectation = does_not_raise()
    with expectation:
        Workflow.from_file(
            src_file="workflow",
            context=None,
            job_dict={
                "WORKFLOW": WorkflowJob(
                    name="WORKFLOW",
                    internal=False,
                    min_args=min_args,
                    max_args=max_args,
                    arg_types=[],
                    executable=None,
                    script=None,
                ),
            },
        )
