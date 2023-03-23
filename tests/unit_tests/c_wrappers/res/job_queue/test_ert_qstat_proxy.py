import datetime
import enum
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import testpath

from ert import _clib

PROXYSCRIPT = _clib.torque_driver.DEFAULT_QSTAT_CMD

EXAMPLE_QSTAT_CONTENT = """
Job Id: 15399.s034-lcam
    Job_Name = DROGON-1
    Job_Owner = combert
    queue = hb120
    job_state = H
Job Id: 15400
    Job_Name = DROGON-2
    Job_Owner = barbert_15399
    queue = hb120
    job_state = R
Job Id: 15402.s034-lcam
    Job_Name = DROGON-3
    Job_Owner = foobert
    queue = hb120
    job_state = E
""".strip()

PROXYFILE_FOR_TESTS = "proxyfile"

MOCKED_QSTAT_BACKEND = (
    # NB: This mock does not support the job id as an argument.
    'import time; time.sleep(0.5); print("""'
    + EXAMPLE_QSTAT_CONTENT
    + '""")'
)
MOCKED_QSTAT_BACKEND_FAILS = "import sys; sys.exit(1)"
MOCKED_QSTAT_BACKEND_LOGGING = (
    "import uuid; open('log/' + str(uuid.uuid4()), 'w').write('.'); "
    + MOCKED_QSTAT_BACKEND
)
MOCKED_QSTAT_ECHO_ARGS = (
    "import sys; open('args', 'w').write(str(sys.argv[1:])); " + MOCKED_QSTAT_BACKEND
)


@pytest.mark.parametrize("jobid", [15399, 15400, 15402])
def test_recent_proxyfile_exists(tmpdir, jobid, monkeypatch):
    monkeypatch.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND):
        result = subprocess.run(
            [PROXYSCRIPT, str(jobid), PROXYFILE_FOR_TESTS],
            check=False,
            capture_output=True,
        )
    print(result)
    assert str(jobid) in str(result.stdout)
    if sys.platform.startswith("darwin"):
        # On Darwin, the proxy script falls back to the mocked backend which is
        # not feature complete for this test:
        assert len(result.stdout.splitlines()) == len(
            EXAMPLE_QSTAT_CONTENT.splitlines()
        )
    else:
        lines = result.stdout.splitlines()
        assert lines[0].decode("utf-8").startswith(f"Job Id: {jobid}")
        assert len(lines) >= 5


def test_proxyfile_not_exists(tmpdir, monkeypatch):
    """If there is no proxy file, the backend should be called"""
    monkeypatch.chdir(tmpdir)
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND):
        result = subprocess.run(
            [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
            check=True,
            capture_output=True,
        )
    if not sys.platform.startswith("darwin"):
        assert Path(PROXYFILE_FOR_TESTS).exists()
    assert "15399" in str(result.stdout)
    if sys.platform.startswith("darwin"):
        # (the mocked backend is not feature complete)
        assert len(result.stdout.splitlines()) == len(
            EXAMPLE_QSTAT_CONTENT.splitlines()
        )
    else:
        lines = result.stdout.splitlines()
        assert lines[0].decode("utf-8") == "Job Id: 15399.s034-lcam"
        assert len(lines) >= 5


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_missing_backend_script(tmpdir, monkeypatch):
    """If a cache file is there, we will use it, but if not, and there is no
    backend, we fail"""
    monkeypatch.chdir(tmpdir)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
            check=True,
        )
    # Try again with cached file present
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    subprocess.run(
        [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
        check=True,
    )


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_recent_proxyfile_locked(tmpdir, monkeypatch):
    """If the proxyfile is locked in the OS, and it is not too old, we should use it.

    NB: This test relies on the proxy script utilizing 'flock', which is possibly
    an implementation detail."""
    monkeypatch.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    with open(PROXYFILE_FOR_TESTS, encoding="utf-8") as proxy_fd:
        fcntl.flock(proxy_fd, fcntl.LOCK_EX)
        # Ensure that if we fall back to the backend, we fail the test:
        with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND_FAILS):
            subprocess.run(
                [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
                check=True,
                capture_output=False,
            )
        fcntl.flock(proxy_fd, fcntl.LOCK_UN)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_old_proxyfile_exists(tmpdir, monkeypatch):
    """If the proxyfile is there, but old, acquire the lock, fix the cache file,
    and return the correct result."""
    monkeypatch.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(
        EXAMPLE_QSTAT_CONTENT.replace("15399", "25399"),
        encoding="utf-8"
        # (if this proxyfile is used, it will fail the test)
    )
    # Manipulate mtime of the file so the script thinks it is old:
    eleven_seconds_ago = datetime.datetime.now() - datetime.timedelta(seconds=11)
    os.utime(
        PROXYFILE_FOR_TESTS,
        (eleven_seconds_ago.timestamp(), eleven_seconds_ago.timestamp()),
    )
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND):
        result = subprocess.run(
            [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
            check=True,
            capture_output=True,
        )
        print(result)
        assert Path(PROXYFILE_FOR_TESTS).exists()
        assert "15399" in str(result.stdout)
        assert len(result.stdout.splitlines()) >= 5


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_old_proxyfile_locked(tmpdir, monkeypatch):
    """If the proxyfile is locked in the OS, and it is too old to use, we fail hard

    NB: This rest relies on the proxy script utilizing 'flock', which is possibly
    an implementation detail."""
    monkeypatch.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    # Manipulate mtime of the file so the script thinks it is old:
    eleven_seconds_ago = datetime.datetime.now() - datetime.timedelta(seconds=11)
    os.utime(
        PROXYFILE_FOR_TESTS,
        (eleven_seconds_ago.timestamp(), eleven_seconds_ago.timestamp()),
    )

    with open(PROXYFILE_FOR_TESTS, encoding="utf-8") as proxy_fd:
        fcntl.flock(proxy_fd, fcntl.LOCK_EX)
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(
                [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
                check=True,
            )
        fcntl.flock(proxy_fd, fcntl.LOCK_UN)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_proxyfile_not_existing_but_locked(tmpdir, monkeypatch):
    """This is when another process has locked the output file for writing, but
    it has not finished yet (and the file is thus empty). The proxy should fail in
    this situation."""
    monkeypatch.chdir(tmpdir)
    with open(PROXYFILE_FOR_TESTS, "w", encoding="utf-8") as proxy_fd:
        fcntl.flock(proxy_fd, fcntl.LOCK_EX)
        assert os.stat(PROXYFILE_FOR_TESTS).st_size == 0
        with pytest.raises(subprocess.CalledProcessError):
            result = subprocess.run(
                [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
                check=True,
                capture_output=True,
            )
            print(str(result.stdout))
            print(str(result.stderr))
        fcntl.flock(proxy_fd, fcntl.LOCK_UN)


@pytest.mark.parametrize(
    "options, expected",
    [
        ("", []),
        ("-x", ["-x"]),
        ("--long-option", ["--long-option"]),
        ("--long1 --long2", ["--long1", "--long2"]),
        ("-x -f", ["-x", "-f"]),
        ("no_go", None),
    ],
)
@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_options_passed_through_proxy(tmpdir, options, expected, monkeypatch):
    monkeypatch.chdir(tmpdir)
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_ECHO_ARGS):
        # pylint: disable=subprocess-run-check
        # (the return value from subprocess is manually checked)
        result = subprocess.run(
            [PROXYSCRIPT, options, "15399", PROXYFILE_FOR_TESTS],
            capture_output=True,
        )
        if expected is None:
            assert result.returncode == 1
            assert result.stdout.strip().decode() == f"qstat: Unknown Job Id {options}"
            return
        assert result.returncode == 0

    cmdline = Path("args").read_text(encoding="utf-8").replace("'", '"')
    assert json.loads(cmdline) == expected
    # (the output from the mocked qstat happens to adhere to json syntax)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_many_concurrent_qstat_invocations(tmpdir, monkeypatch):
    """Run many qstat invocations simultaneously, with a mocked backend qstat
    script that logs how many times it is invoked, then we assert that it has
    not been called often.

    If this test has enough invocations (or the hardware is slow enough), this test
    will also pass the timeout in the script yielding a rerun of the backend. In that
    scenario there is a risk for a race condition where the cache file is blank when
    another process is reading from it. This test also assert that failures are only
    allowed to happen in a sequence from the second invocation (the first process
    succeeds because it calls the backend, the second one fails because there is no
    cache file and it is locked.)

    This test will dump something like
    01111111111000000000000000000000000000000000000 to stdout
    where each digit is the return code from the proxy script. Only one sequence of 1's
    is allowed after the start, later on there should be no failures (that would be
    errors from race conditions.)
    """
    starttime = time.time()
    invocations = 400
    sleeptime = 0.02  # seconds. Lower number increase probability of race conditions.
    # (the mocked qstat backend sleeps for 0.5 seconds to facilitate races)
    cache_timeout = 2  # This is CACHE_TIMEOUT in the shell script
    assert invocations * sleeptime > cache_timeout  # Ensure race conditions can happen

    monkeypatch.chdir(tmpdir)
    Path("log").mkdir()  # The mocked backend writes to this directory
    subprocesses = []
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND_LOGGING):
        for _ in range(invocations):
            # pylint: disable=consider-using-with
            # process.wait() is called below
            subprocesses.append(
                subprocess.Popen(
                    [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
                    stdout=subprocess.DEVNULL,
                )
            )
            # The asserts below in this test function depend on each subprocess
            # to finish in order of invocation:
            time.sleep(sleeptime)

        class CacheState(enum.Enum):
            # Only consecutive transitions are allowed.
            FIRST_INVOCATION = 0
            FIRST_HOLDS_FLOCK = 1
            CACHE_EXISTS = 2

        state = None
        for _, process in enumerate(subprocesses):
            process.wait()
            if state is None:
                if process.returncode == 0:
                    state = CacheState.FIRST_INVOCATION
                assert state is not None, "First invocation should not fail"

            elif state == CacheState.FIRST_INVOCATION:
                assert process.returncode == 1
                # The proxy should fail in this scenario, and ERTs queue
                # manager must retry later.
                state = CacheState.FIRST_HOLDS_FLOCK

            elif state == CacheState.FIRST_HOLDS_FLOCK:
                if process.returncode == 1:
                    # Continue waiting until the cache is ready
                    pass
                if process.returncode == 0:
                    state = CacheState.CACHE_EXISTS

            else:
                assert state == CacheState.CACHE_EXISTS
                assert (
                    process.returncode == 0
                ), "Check for race condition if AssertionError"

            print(process.returncode, end="")
        print("\n")

    # Allow a limited set of backend runs. We get more backend runs the
    # slower the iron.
    time_taken = time.time() - starttime
    backend_runs = len(list(Path("log").iterdir()))
    print(
        f"We got {backend_runs} backend runs from "
        f"{invocations} invocations in {time_taken:.2f} seconds."
    )

    # We require more than one backend run because there is race condition we need
    # to test for. Number of backend runs should then be relative to the time taken
    # to run the test (plus 3 for slack)
    assert 1 < backend_runs < int(time_taken / cache_timeout) + 3


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_optional_job_id_namespace(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    assert "15399.s034" in EXAMPLE_QSTAT_CONTENT
    result_job_with_namespace = subprocess.run(
        [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS], check=True, capture_output=True
    )
    lines = result_job_with_namespace.stdout.splitlines()
    assert lines[0].decode("utf-8") == "Job Id: 15399.s034-lcam"
    assert len(lines) >= 5

    assert "15400\n" in EXAMPLE_QSTAT_CONTENT
    result_job_without_namespace = subprocess.run(
        [PROXYSCRIPT, "15400", PROXYFILE_FOR_TESTS], check=True, capture_output=True
    )
    lines = result_job_without_namespace.stdout.splitlines()
    assert lines[0].decode("utf-8") == "Job Id: 15400"
    assert len(lines) >= 5


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_no_such_job_id(tmpdir, monkeypatch):
    """Ensure we replicate qstat's error behaviour, yielding error
    if a job id does not exist."""

    monkeypatch.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [PROXYSCRIPT, "10001", PROXYFILE_FOR_TESTS],
            check=True,
        )


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_proxy_fails_if_backend_fails(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    with pytest.raises(subprocess.CalledProcessError), testpath.MockCommand(
        "qstat", python=MOCKED_QSTAT_BACKEND_FAILS
    ):
        subprocess.run(
            [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
            check=True,
            capture_output=False,
        )


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_no_argument(tmpdir, monkeypatch):
    """qstat with no arguments lists all jobs. So should the proxy."""
    monkeypatch.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    result = subprocess.run(
        [PROXYSCRIPT, "", PROXYFILE_FOR_TESTS],
        check=True,
        capture_output=True,
    )
    assert len(result.stdout.splitlines()) == len(EXAMPLE_QSTAT_CONTENT.splitlines())
