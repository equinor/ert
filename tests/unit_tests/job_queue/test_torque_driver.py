from pathlib import Path
from typing import Optional

import pytest

from ert import _clib
from ert.job_queue import JobStatusType


@pytest.mark.usefixtures("use_tmpdir")
def test_job_create_submit_script():
    script_name = "qsub_script.sh"
    _clib.torque_driver.create_submit_script(
        script_name, "job_program.py", ["/tmp/jaja/", "number2arg"]
    )
    assert (
        Path(script_name).read_text(encoding="utf-8")
        == "#!/bin/sh\njob_program.py /tmp/jaja/ number2arg"
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "qstat_output, jobnr, expected_status",
    [
        (None, "", JobStatusType.JOB_QUEUE_STATUS_FAILURE),
        ("", "1234", JobStatusType.JOB_QUEUE_STATUS_FAILURE),
        ("Job Id: 1\njob_state = R", "1", JobStatusType.JOB_QUEUE_RUNNING),
        ("Job Id: 1.namespace\njob_state = R", "1", JobStatusType.JOB_QUEUE_RUNNING),
        ("Job Id: 11\njob_state = R", "1", JobStatusType.JOB_QUEUE_STATUS_FAILURE),
        ("Job Id: 1", "1", JobStatusType.JOB_QUEUE_STATUS_FAILURE),
        ("Job Id: 1\njob_state = E", "1", JobStatusType.JOB_QUEUE_DONE),
        ("Job Id: 1\njob_state = C", "1", JobStatusType.JOB_QUEUE_DONE),
        ("Job Id: 1\njob_state = H", "1", JobStatusType.JOB_QUEUE_PENDING),
        ("Job Id: 1\njob_state = Q", "1", JobStatusType.JOB_QUEUE_PENDING),
        ("Job Id: 1\njob_state = Æ", "1", JobStatusType.JOB_QUEUE_STATUS_FAILURE),
        (
            "Job Id: 1\njob_state = E\nExit_status = 1",
            "1",
            JobStatusType.JOB_QUEUE_EXIT,
        ),
        (
            "Job Id: 1\njob_state = C\nExit_status = 1",
            "1",
            JobStatusType.JOB_QUEUE_EXIT,
        ),
        (
            "Job Id: 1\njob_state = C\nJob Id: 2\njob_state = R",
            "2",
            JobStatusType.JOB_QUEUE_RUNNING,
        ),
    ],
)
def test_parse_status(
    qstat_output: Optional[str], jobnr: str, expected_status: JobStatusType
):
    if qstat_output is not None:
        Path("qstat.out").write_text(qstat_output, encoding="utf-8")
    assert _clib.torque_driver.parse_status("qstat.out", jobnr) == expected_status


@pytest.mark.parametrize(
    "num_nodes, cluster_label, num_cpus_per_node, "
    "memory_per_job, expected_resource_string",
    [
        pytest.param(1, "", 1, "", "nodes=1:ppn=1", id="defaults"),
        pytest.param(
            1, "fancynodes", 2, "", "nodes=1:fancynodes:ppn=2", id="clusterlabel"
        ),
        pytest.param(1, "", 2, "32gb", "nodes=1:ppn=2:mem=32gb", id="memory_per_job"),
        pytest.param(
            1, "", 2, "32pb", "nodes=1:ppn=2:mem=32pb", id="outrageous_memory_per_job"
        ),
        pytest.param(
            1,
            "bignodes",
            2,
            "32pb",
            "nodes=1:bignodes:ppn=2:mem=32pb",
            id="label_and_memory",
        ),
        # ERTs config parser will give ConfigValidationError on
        # memory strings not adhering to <integer><mb|gb>.
    ],
)
def test_build_resource_string(
    num_nodes,
    cluster_label,
    num_cpus_per_node,
    memory_per_job,
    expected_resource_string,
):
    assert (
        _clib.torque_driver.build_resource_string(
            num_nodes, cluster_label, num_cpus_per_node, memory_per_job
        )
        == expected_resource_string
    )
