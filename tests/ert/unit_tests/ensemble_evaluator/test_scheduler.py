import asyncio
import json
import logging
from pathlib import Path
from queue import SimpleQueue
from unittest.mock import AsyncMock

import pytest

from ert.ensemble_evaluator import EnsembleEvaluator, state
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.scheduler import job
from ert.scheduler.job import Job
from ert.scheduler.scheduler import Scheduler


@pytest.mark.integration_test
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

        # Creating testing manifest file
        create_manifest_file()
        file_path = Path("real_0/job_test_file")
        file_path.write_text("test")
        # Skip waiting for stdout/err in job
        mocked_stdouterr_parser = AsyncMock(
            return_value=Job.DEFAULT_FILE_VERIFICATION_TIMEOUT
        )
        monkeypatch.setattr(
            job, "log_warnings_from_forward_model", mocked_stdouterr_parser
        )
        # actual_md5sum = hashlib.md5(file_path.read_bytes()).hexdigest()
        config = EvaluatorServerConfig(use_token=False)

        event_queue = asyncio.Queue()

        evaluator = EnsembleEvaluator(
            ensemble, config, SimpleQueue(), event_queue.put_nowait
        )
        original_method = Scheduler._update_checksum

        async def mock_update_checksum(self, *args) -> None:
            await original_method(self, *args)
            await asyncio.wait_for(rename_and_wait(), timeout=5)

        monkeypatch.setattr(Scheduler, "_update_checksum", mock_update_checksum)

        await evaluator.run_and_get_successful_realizations()
        assert "Waiting for disk synchronization" in caplog.messages
        assert f"File {file_path.absolute()} checksum successful." in caplog.messages
        assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
