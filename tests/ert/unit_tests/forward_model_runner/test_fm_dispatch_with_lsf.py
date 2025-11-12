import asyncio
import contextlib
import json
import shutil

import pytest
import zmq.asyncio

from ert.scheduler.event import FinishedEvent, StartedEvent
from ert.scheduler.lsf_driver import LsfDriver
from ert.shared import get_ip_address
from tests.ert.utils import MockZMQServer, wait_until


class TCPMockZMQServer(MockZMQServer):
    """MockZMQServer that uses TCP instead of IPC protocol."""

    def __init__(self, signal=0, port_range=(51820, 51840)) -> None:
        super().__init__(signal)
        # Need to use public IP on some clusters
        self.host = get_ip_address()
        self.port = port_range[0]
        self.min_port = port_range[0]
        self.max_port = port_range[1]
        self.port = None
        self._server_ready = asyncio.Event()

    async def __aenter__(self) -> "TCPMockZMQServer":
        await super().__aenter__()
        await self._server_ready.wait()
        return self

    async def mock_zmq_server(self):
        zmq_context = zmq.asyncio.Context()
        self.router_socket = zmq_context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 0)
        self.port = self.router_socket.bind_to_random_port(
            "tcp://*", min_port=self.min_port, max_port=self.max_port
        )
        self.uri = f"tcp://{self.host}:{self.port}"
        self._server_ready.set()
        self.handler_task = asyncio.create_task(self._handler())
        try:
            await self.handler_task
        finally:
            self.router_socket.close()
            zmq_context.term()


def create_jobs_json(dispatch_url: str, fm_steps: list[tuple[str, str]]) -> dict:
    """Create a jobs.json configuration for fm_dispatch."""
    return {
        "ens_id": "test_ensemble",
        "real_id": 0,
        "dispatch_url": dispatch_url,
        "jobList": [
            {
                "name": name,
                "executable": executable,
                "stdout": f"{name}.stdout",
                "stderr": f"{name}.stderr",
            }
            for name, executable in fm_steps
        ],
    }


@pytest.mark.integration_test
@pytest.mark.timeout(120)
async def test_that_job_submitted_job_sends_start_and_success_events_to_zmq_server(
    pytestconfig, tmp_path
):
    if not pytestconfig.getoption("lsf"):
        pytest.skip()

    assert not str(tmp_path).startswith("/tmp"), (
        "Please use --basetemp option to pytest, "
        "the real LSF cluster needs a shared disk"
    )

    """Verify that jobs submitted via drivers communicate with a ZMQ server."""
    test_dir = tmp_path / "test_fm_dispatch"
    test_dir.mkdir()

    fm_script = test_dir / "simple_step.sh"
    fm_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fm_script.chmod(0o755)
    runpath = test_dir / "realization-0"
    runpath.mkdir()

    async with TCPMockZMQServer() as zmq_server:
        jobs_json = create_jobs_json(
            zmq_server.uri, [("simple_step", str(fm_script.absolute()))]
        )
        (runpath / "jobs.json").write_text(json.dumps(jobs_json), encoding="utf-8")

        driver = LsfDriver()

        fm_dispatch_path = shutil.which("fm_dispatch.py") or "fm_dispatch.py"
        await driver.submit(
            0, fm_dispatch_path, str(runpath.absolute()), runpath=runpath
        )

        async def poll_until_done():
            while True:
                await driver.poll()
                await asyncio.sleep(0.1)

        polling_task = asyncio.create_task(poll_until_done())

        try:
            finished = None
            while finished is None:
                event = await asyncio.wait_for(driver.event_queue.get(), timeout=60)
                if isinstance(event, FinishedEvent):
                    finished = event
                elif isinstance(event, StartedEvent):
                    assert event.iens == 0

            assert finished.iens == 0
            assert finished.returncode == 0
        finally:
            polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await polling_task

        wait_until(lambda: len(zmq_server.messages) > 0, timeout=10)

        event_types = {json.loads(msg).get("event_type") for msg in zmq_server.messages}
        assert "forward_model_step.start" in event_types
        assert "forward_model_step.success" in event_types
