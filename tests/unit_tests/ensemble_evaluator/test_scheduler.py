import asyncio
import json
import logging
from pathlib import Path

import pytest

from ert.ensemble_evaluator import Monitor, identifiers, state
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.evaluator import EnsembleEvaluator


@pytest.mark.timeout(60)
def test_scheduler_receives_checksum_and_waits_for_disk_sync(
    tmpdir, make_ensemble_builder, monkeypatch, caplog, run_monitor_in_loop
):
    num_reals = 1
    custom_port_range = range(1024, 65535)

    async def rename_and_wait():
        Path("real_0/job_test_file").rename("real_0/test")
        while "Waiting for disk synchronization" not in caplog.messages:
            await asyncio.sleep(0.1)
        Path("real_0/test").rename("real_0/job_test_file")

    async def _run_monitor():
        async with Monitor(config) as monitor:
            async for e in monitor.track():
                if e["type"] == identifiers.EVTYPE_FORWARD_MODEL_CHECKSUM:
                    # Monitor got the checksum message renaming the file
                    # before the scheduler gets the same message
                    try:
                        await asyncio.wait_for(rename_and_wait(), timeout=5)
                    except TimeoutError:
                        await monitor.signal_done()
                if e["type"] in (
                    identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                    identifiers.EVTYPE_EE_SNAPSHOT,
                ) and e.data.get(identifiers.STATUS) in [
                    state.ENSEMBLE_STATE_FAILED,
                    state.ENSEMBLE_STATE_STOPPED,
                ]:
                    await monitor.signal_done()
        return True

    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(monkeypatch, tmpdir, num_reals, 2).build()

        # Creating testing manifest file
        with open("real_0/manifest.json", mode="w", encoding="utf-8") as f:
            json.dump({"file": "job_test_file"}, f)
        file_path = Path("real_0/job_test_file")
        file_path.write_text("test")
        # actual_md5sum = hashlib.md5(file_path.read_bytes()).hexdigest()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range,
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )
        evaluator = EnsembleEvaluator(ensemble, config, 0)
        with caplog.at_level(logging.DEBUG):
            evaluator.start_running()
            run_monitor_in_loop(_run_monitor)
        assert "Waiting for disk synchronization" in caplog.messages
        assert f"File {file_path.absolute()} checksum successful." in caplog.messages
        assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
