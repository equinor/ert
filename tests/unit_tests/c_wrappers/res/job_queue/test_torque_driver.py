from pathlib import Path
from typing import Optional

import pytest

from ert import _clib
from ert.job_queue import JobStatusType


def test_job_create_submit_script(use_tmpdir):
    # pylint: disable=unused-argument
    script_name = "qsub_script.sh"
    _clib.torque_driver.create_submit_script(
        script_name, "job_program.py", ["/tmp/jaja/", "number2arg"]
    )
    assert (
        Path(script_name).read_text(encoding="utf-8")
        == "#!/bin/sh\njob_program.py /tmp/jaja/ number2arg"
    )


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
    qstat_output: Optional[str], jobnr: str, expected_status: JobStatusType, use_tmpdir
):
    # pylint: disable=unused-argument
    if qstat_output is not None:
        Path("qstat.out").write_text(qstat_output, encoding="utf-8")
    assert _clib.torque_driver.parse_status("qstat.out", jobnr) == expected_status
