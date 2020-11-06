import asyncio
import logging
from ert_shared.ensemble_evaluator.entity.ensemble import (
    create_ensemble_builder_from_legacy,
)
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
from ert_shared.ensemble_evaluator.config import load_config

logger = logging.getLogger(__name__)


@contextmanager
def _attach(run_context, run_path_list, forward_model):
    asyncio.set_event_loop(asyncio.new_event_loop())
    ee_config = load_config()

    builder = create_ensemble_builder_from_legacy(run_context, forward_model)
    ensemble = builder.build()
    logger.debug(builder)

    ee = EnsembleEvaluator(ensemble, ee_config)
    logger.debug(ee)
    ee.run()

    logger.debug("waiting for ee ws")

    wait_for_ws(ee_config.get("url"))

    logger.debug("ee ws started")

    event_logs = [Path(path.runpath) / "event_log" for path in run_path_list]
    dispatch_thread = Thread(
        target=_attach_to_dispatch, args=(ee_config.get("dispatch_url"), event_logs)
    )
    dispatch_thread.start()

    JobQueueManagerAdaptor.ws_url = ee_config.get("dispatch_url")
    JobQueueManagerAdaptor.ee_id = str(uuid.uuid1()).split("-")[0]
    patcher = patch(
        "res.enkf.enkf_simulation_runner.JobQueueManager", new=JobQueueManagerAdaptor
    )
    patcher.start()
    yield
    dispatch_thread.join()
    patcher.stop()
    ee.stop()


def attach_ensemble_evaluator(run_context, run_path_list, forward_model):
    if FeatureToggling.is_enabled("ensemble-evaluator"):
        return _attach(run_context, run_path_list, forward_model)
    return ExitStack()


def _attach_to_dispatch(ws_url, event_logs):
    asyncio.set_event_loop(asyncio.new_event_loop())
    futures = tuple(nfs_adaptor(event_log, ws_url) for event_log in event_logs)
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
