from pathlib import Path
from typing import Optional

import pytest

from ert._c_wrappers.job_queue.job_status_type_enum import JobStatusType
from ert._clib import _torque_driver


def test_job_create_submit_script(use_tmpdir):
    # pylint: disable=unused-argument
    script_name = "qsub_script.sh"
    _torque_driver.torque_job_create_submit_script(
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
    ],
)
def test_parse_status(
    qstat_output: Optional[str], jobnr: str, expected_status: JobStatusType, use_tmpdir
):
    # pylint: disable=unused-argument
    if qstat_output is not None:
        Path("qstat.out").write_text(qstat_output, encoding="utf-8")
    assert (
        _torque_driver.torque_driver_parse_status("qstat.out", jobnr) == expected_status
    )
