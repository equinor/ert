import asyncio
import json
import logging
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock

import pytest

from _ert.events import ForwardModelStepChecksum
from ert.ensemble_evaluator import EnsembleEvaluator, state
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.scheduler import job
from ert.scheduler.job import Job
from ert.scheduler.scheduler import Scheduler


@pytest.mark.slow
@pytest.mark.timeout(60)
async def test_scheduler_receives_checksum_and_waits_for_disk_sync(
    tmpdir, make_ensemble, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG)

    async def rename_and_wait():
        Path("real_0/job_test_file").rename("real_0/test")
        for _ in iter(
            lambda: "Waiting for disk synchronization" not in caplog.messages, False
        ):
            await asyncio.sleep(0.1)
        Path("real_0/test").rename("real_0/job_test_file")

    def create_manifest_file():
        with open("real_0/manifest.json", mode="w", encoding="utf-8") as f:
            json.dump({"file": "job_test_file"}, f)

    with tmpdir.as_cwd():
        ensemble = make_ensemble(monkeypatch, tmpdir, 1, 2)
        # rename_and_wait_task = asyncio.create_task(rename_and_wait())
        # Creating testing manifest file
        create_manifest_file()
        file_path = Path("real_0/job_test_file")
        file_path.write_text("test", encoding="utf-8")
        # Skip waiting for stdout/err in job
        mocked_stdouterr_parser = AsyncMock(
            return_value=Job.DEFAULT_FILE_VERIFICATION_TIMEOUT
        )
        monkeypatch.setattr(
            job, "log_warnings_from_forward_model", mocked_stdouterr_parser
        )

        config = EvaluatorServerConfig(use_token=False)

        event_queue = asyncio.Queue()

        evaluator = EnsembleEvaluator(ensemble, config, Event(), event_queue.put_nowait)

        async def mock_checksum_consumer(self, *args) -> None:
            event = await self._manifest_queue.get()
            assert isinstance(event, ForwardModelStepChecksum)
            self.checksum.update(event.checksums)
            self._manifest_queue.task_done()
            await rename_and_wait()

        monkeypatch.setattr(Scheduler, "_checksum_consumer", mock_checksum_consumer)
        monkeypatch.setattr(job, "DISK_SYNCHRONIZATION_POLLING_INTERVAL", 0.1)
        await asyncio.gather(evaluator.run_and_get_successful_realizations())
        assert "Waiting for disk synchronization" in caplog.messages
        assert f"File {file_path.absolute()} checksum successful." in caplog.messages
        assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
