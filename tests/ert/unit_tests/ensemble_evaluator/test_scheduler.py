import asyncio
import json
import logging
from pathlib import Path

import pytest

from _ert.events import EESnapshot, EESnapshotUpdate, ForwardModelStepChecksum
from ert.ensemble_evaluator import EnsembleEvaluator, identifiers, state
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.scheduler.event import Event


@pytest.mark.integration_test
@pytest.mark.timeout(60)
async def test_scheduler_receives_checksum_and_waits_for_disk_sync(
    tmpdir, make_ensemble, monkeypatch, caplog
):
    async def rename_and_wait():
        Path("real_0/job_test_file").rename("real_0/test")
        for _ in iter(
            lambda: "Waiting for disk synchronization" not in caplog.messages, False
        ):
            await asyncio.sleep(0.1)
        Path("real_0/test").rename("real_0/job_test_file")

    async def _handle_events(events_to_brm: asyncio.Queue[Event]):
        while True:
            event = await events_to_brm.get()
            if type(event) is ForwardModelStepChecksum:
                # Monitor got the checksum message renaming the file
                # before the scheduler gets the same message
                await asyncio.wait_for(rename_and_wait(), timeout=5)
            if type(event) in {
                EESnapshot,
                EESnapshotUpdate,
            } and event.snapshot.get(identifiers.STATUS) in {
                state.ENSEMBLE_STATE_FAILED,
                state.ENSEMBLE_STATE_STOPPED,
            }:
                return

    def create_manifest_file():
        with open("real_0/manifest.json", mode="w", encoding="utf-8") as f:
            json.dump({"file": "job_test_file"}, f)

    with tmpdir.as_cwd():
        ensemble = make_ensemble(monkeypatch, tmpdir, 1, 2)

        # Creating testing manifest file
        create_manifest_file()
        file_path = Path("real_0/job_test_file")
        file_path.write_text("test")
        # actual_md5sum = hashlib.md5(file_path.read_bytes()).hexdigest()
        config = EvaluatorServerConfig(use_token=False)
        events_to_brm = asyncio.Queue()
        evaluator = EnsembleEvaluator(
            ensemble, config, send_to_brm=events_to_brm.put_nowait
        )
        with caplog.at_level(logging.DEBUG):
            run_task = asyncio.create_task(
                evaluator.run_and_get_successful_realizations()
            )
            await evaluator._server_started
            await asyncio.wait_for(_handle_events(events_to_brm), timeout=10)
            await run_task
        assert "Waiting for disk synchronization" in caplog.messages
        assert f"File {file_path.absolute()} checksum successful." in caplog.messages
        assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
