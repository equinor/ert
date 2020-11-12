import asyncio
import os
import threading
from functools import partial
import logging
from pathlib import Path

from ert_shared.ensemble_evaluator.entity.ensemble_base import _Ensemble
from ert_shared.ensemble_evaluator.nfs_adaptor import nfs_adaptor
from ert_shared.ensemble_evaluator.queue_adaptor import QueueAdaptor
from res.enkf import EnKFState
from res.enkf.enums.realization_state_enum import RealizationStateEnum

CONCURRENT_INTERNALIZATION = 10

logger = logging.getLogger(__name__)


class _LegacyEnsemble(_Ensemble):
    def __init__(
        self,
        reals,
        metadata,
        *args,
    ):
        super().__init__(reals, metadata)
        (
            run_context,
            forward_model,
            queue_config,
            model_config,
            run_path_list,
            analysis_config,
            ecl_config,
            res_config,
        ) = args
        if not run_context:
            raise ValueError(f"{self} needs run_context")
        if not forward_model:
            raise ValueError(f"{self} needs forward_model")
        if not queue_config:
            raise ValueError(f"{self} needs queue_config")
        if not model_config:
            raise ValueError(f"{self} needs model_config")
        if not run_path_list:
            raise ValueError(f"{self} needs run_path_list")
        if not analysis_config:
            raise ValueError(f"{self} needs analysis_config")
        if not ecl_config:
            raise ValueError(f"{self} needs ecl_config")
        if not res_config:
            raise ValueError(f"{self} needs res_config")
        self._run_context = run_context
        self._forward_model = forward_model
        self._queue_config = queue_config
        self._model_config = model_config
        self._run_path_list = run_path_list
        self._analysis_config = analysis_config
        self._ecl_config = ecl_config
        self._res_config = res_config
        self._queue_thread = None
        self._dispatch_thread = None
        self._mon = None

    def evaluate(self, config, ee_id, mon):
        self._mon = mon
        self._queue_thread = threading.Thread(
            target=self._run_queue, args=(config, ee_id)
        )
        self._queue_thread.start()

        event_logs = [Path(path.runpath) / "event_log" for path in self._run_path_list]
        for event_log in event_logs:
            if os.path.isfile(event_log):
                os.remove(event_log)
        self._dispatch_thread = threading.Thread(
            target=_attach_to_dispatch, args=(config, event_logs)
        )
        self._dispatch_thread.start()

    def _run_queue(self, config, ee_id):
        asyncio.set_event_loop(asyncio.new_event_loop())
        ws_url = config.get("dispatch_url")
        job_queue = self._queue_config.create_job_queue()

        if self._run_context.get_step():
            self._ecl_config.assert_restart()

        iactive = self._run_context.get_mask()

        self._run_context.get_sim_fs().getStateMap().deselectMatching(
            iactive,
            RealizationStateEnum.STATE_LOAD_FAILURE
            | RealizationStateEnum.STATE_PARENT_FAILURE,
        )

        # TODO: update ensemble, deactivate newly deactivated reals

        max_runtime = self._analysis_config.get_max_runtime()
        if max_runtime == 0:
            max_runtime = None

        done_callback_function = EnKFState.forward_model_ok_callback
        exit_callback_function = EnKFState.forward_model_exit_callback

        # submit jobs
        for i in range(len(self._run_context)):
            if not self._run_context.is_active(i):
                continue
            run_arg = self._run_context[i]
            job_queue.add_job_from_run_arg(
                run_arg,
                self._res_config,
                max_runtime,
                done_callback_function,
                exit_callback_function,
            )

        job_queue.submit_complete()
        queue_evaluators = None
        if (
            self._analysis_config.get_stop_long_running()
            and self._analysis_config.minimum_required_realizations > 0
        ):
            queue_evaluators = [
                partial(
                    job_queue.stop_long_running_jobs,
                    self._analysis_config.minimum_required_realizations,
                )
            ]

        queue_adaptor = QueueAdaptor(job_queue, ws_url, ee_id)
        queue_adaptor.execute_queue(
            threading.BoundedSemaphore(value=CONCURRENT_INTERNALIZATION),
            queue_evaluators,
        )
        logger.debug("waiting for dispatchers to complete reporting...")
        self._dispatch_thread.join()
        logger.debug("request terminate")
        self._mon.exit_server()


def _attach_to_dispatch(config, event_logs):
    ws_url = config.get("dispatch_url")
    asyncio.set_event_loop(asyncio.new_event_loop())
    futures = tuple(nfs_adaptor(event_log, ws_url) for event_log in event_logs)
    for coro in asyncio.as_completed(futures):
        try:
            asyncio.get_event_loop().run_until_complete(coro)
        except Exception as e:
            import traceback

            logger.error(f"nfs adaptor {e}: {traceback.format_exc()}")
