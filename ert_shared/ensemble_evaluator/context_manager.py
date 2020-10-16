import asyncio
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from cloudevents.http import event

from ert_shared.ensemble_evaluator.nfs_adaptor import nfs_adaptor
from ert_shared.ensemble_evaluator.queue_adaptor import JobQueueManagerAdaptor
from ert_shared.feature_toggling import FeatureToggling
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.ws_util import wait_for_ws


@contextmanager
def _attach(enkf_main):
    ws_url = "ws://localhost:8765"
    dispatch_url = f"{ws_url}/dispatch"
    # ee = EnsembleEvaluator()
    # ee.run()

    # print("waiting for ee ws")

    # wait_for_ws(ws_url)

    print("ee ws started")

    event_logs = [
        Path(path.runpath) / "event_log" for path in enkf_main.getRunpathList()
    ]
    dispatch_thread = Thread(target=_attach_to_dispatch, args=(dispatch_url, event_logs))
    dispatch_thread.start()

    # XXX: these magic strings will eventually come from EE itself
    JobQueueManagerAdaptor.ws_url = dispatch_url
    JobQueueManagerAdaptor.ee_id = str(uuid.uuid1()).split("-")[0]
    patcher = patch(
        "res.enkf.enkf_simulation_runner.JobQueueManager", new=JobQueueManagerAdaptor
    )
    patcher.start()
    yield
    dispatch_thread.join()
    patcher.stop()
    # ee.stop()


def attach_ensemble_evaluator(enkf_main):
    if FeatureToggling.is_enabled("ensemble-evaluator"):
        return _attach(enkf_main)
    return ExitStack()


def _attach_to_dispatch(ws_url, event_logs):
    asyncio.set_event_loop(asyncio.new_event_loop())
    futures = tuple(nfs_adaptor(event_log, ws_url) for event_log in event_logs)
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
