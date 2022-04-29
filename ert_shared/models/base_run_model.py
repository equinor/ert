import logging
from contextlib import contextmanager

import time
import uuid
from typing import Optional, List, Union, Dict, Any, Iterator
import asyncio

from res.enkf import EnKFMain, QueueConfig
from res.enkf.ert_run_context import ErtRunContext
from res.job_queue import (
    RunStatusType,
    ForwardModel,
)
from ert_shared.libres_facade import LibresFacade
from ert_shared.feature_toggling import feature_enabled
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.storage.extraction import (
    post_ensemble_data,
    post_ensemble_results,
    post_update_data,
)
from ert.ensemble_evaluator import (
    EnsembleBuilder,
)


class ErtRunError(Exception):
    pass


class _LogAggregration(logging.Handler):
    """Logging handler which aggregates the log messages"""

    def __init__(self) -> None:
        self.messages: List[str] = []
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def captured_logs(level: int = logging.ERROR) -> Iterator[_LogAggregration]:
    handler = _LogAggregration()
    root_logger = logging.getLogger()
    handler.setLevel(level)
    root_logger.addHandler(handler)
    try:
        yield handler
    finally:
        root_logger.removeHandler(handler)


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class BaseRunModel:
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
        phase_count: int = 1,
    ):
        """

        Parameters
        ----------
        simulation_arguments : Parameters for running the simulation,
            eg. activate realizations, analysis module
        queue_config : QueueConfig
        phase_count : Optional[int], optional
            Number of data assimilation cycles / iterations an experiment will have,
            by default 1
        """
        self._phase: int = 0
        self._phase_count = phase_count
        self._phase_name: str = "Starting..."

        self._job_start_time: int = 0
        self._job_stop_time: int = 0
        self._indeterminate: bool = False
        self._fail_message: str = ""
        self._failed: bool = False
        self._queue_config: QueueConfig = queue_config
        self._initial_realizations_mask: List[bool] = []
        self._completed_realizations_mask: List[bool] = []
        self.support_restart: bool = True
        self._run_context: Optional[ErtRunContext] = None
        self._last_run_iteration: int = -1
        self._ert = ert
        self.facade = LibresFacade(ert)
        self._simulation_arguments = simulation_arguments
        self.reset()

    def ert(self) -> EnKFMain:
        return self._ert

    @property
    def _ensemble_size(self) -> int:
        return len(self._initial_realizations_mask)

    @property
    def _active_realizations(self) -> List[int]:
        return [
            idx
            for idx, mask_val in enumerate(self._initial_realizations_mask)
            if mask_val
        ]

    def reset(self) -> None:
        self._failed = False
        self._phase = 0

    def restart(self) -> None:
        active_realizations = self._create_mask_from_failed_realizations()
        self._simulation_arguments["active_realizations"] = active_realizations
        self._simulation_arguments[
            "prev_successful_realizations"
        ] = self._simulation_arguments.get("prev_successful_realizations", 0)
        self._simulation_arguments[
            "prev_successful_realizations"
        ] += self._count_successful_realizations()

    def has_failed_realizations(self) -> bool:
        return any(self._create_mask_from_failed_realizations())

    def _create_mask_from_failed_realizations(self) -> List[bool]:
        """
        Creates a list of bools representing the failed realizations,
        i.e., a realization that has failed is assigned a True value.
        """
        return [
            initial and not completed
            for initial, completed in zip(
                self._initial_realizations_mask, self._completed_realizations_mask
            )
        ]

    def _count_successful_realizations(self) -> int:
        """
        Counts the realizations completed in the prevoius ensemble run
        :return:
        """
        completed = self._completed_realizations_mask
        return completed.count(True)

    def start_simulations_thread(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.startSimulations(
            evaluator_server_config=evaluator_server_config,
        )

    def startSimulations(self, evaluator_server_config: EvaluatorServerConfig) -> None:
        try:
            with captured_logs() as logs:
                self._initial_realizations_mask = self._simulation_arguments[
                    "active_realizations"
                ]
                run_context = self.runSimulations(
                    evaluator_server_config=evaluator_server_config,
                )
                self._completed_realizations_mask = run_context.get_mask()
        except ErtRunError as e:
            self._completed_realizations_mask = []
            self._failed = True
            self._fail_message = str(e) + "\n" + "\n".join(logs.messages)
            self._simulationEnded()
        except UserWarning as e:
            self._fail_message = str(e) + "\n" + "\n".join(logs.messages)
            self._simulationEnded()
        except Exception as e:
            self._failed = True
            self._fail_message = str(e) + "\n" + "\n".join(logs.messages)
            self._simulationEnded()
            raise

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> ErtRunContext:
        raise NotImplementedError("Method must be implemented by inheritors!")

    def teardown_context(self) -> None:
        # Used particularly to delete last active run_context to notify
        # fs_manager that storage is not being written to.
        self._run_context = None

    def phaseCount(self) -> int:
        return self._phase_count

    def setPhaseCount(self, phase_count: int) -> None:
        self._phase_count = phase_count
        self.setPhase(0, "")

    def currentPhase(self) -> int:
        return self._phase

    def setPhaseName(
        self, phase_name: str, indeterminate: Optional[bool] = None
    ) -> None:
        self._phase_name = phase_name
        self.setIndeterminate(indeterminate)

    def getPhaseName(self) -> str:
        return self._phase_name

    def setIndeterminate(self, indeterminate: Union[bool, None]) -> None:
        if indeterminate is not None:
            self._indeterminate = indeterminate

    def isFinished(self) -> bool:
        return self._phase == self._phase_count or self.hasRunFailed()

    def hasRunFailed(self) -> bool:
        return self._failed

    def getFailMessage(self) -> str:
        return self._fail_message

    def _simulationEnded(self) -> None:
        self._job_stop_time = int(time.time())

    def setPhase(
        self, phase: int, phase_name: str, indeterminate: Optional[bool] = None
    ) -> None:
        self.setPhaseName(phase_name)
        if not 0 <= phase <= self._phase_count:
            raise ValueError(
                "Phase must be an integer between (inclusive) 0 and {self._phase_count}"
            )

        self.setIndeterminate(indeterminate)

        if phase == 0:
            self._job_start_time = int(time.time())

        if phase == self._phase_count:
            self._simulationEnded()

        self._phase = phase

    def stop_time(self) -> int:
        return self._job_stop_time

    def start_time(self) -> int:
        return self._job_start_time

    def get_runtime(self) -> Union[int, float]:
        if self.stop_time() < self.start_time():
            return time.time() - self.start_time()
        else:
            return self.stop_time() - self.start_time()

    @staticmethod
    def is_forward_model_finished(progress: ForwardModel) -> bool:
        return all(job.status == "Success" for job in progress)

    def isIndeterminate(self) -> bool:
        return not self.isFinished() and self._indeterminate

    def checkHaveSufficientRealizations(self, num_successful_realizations: int) -> None:
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed! All realizations failed!")
        if (
            not self.ert()
            .analysisConfig()
            .haveEnoughRealisations(
                num_successful_realizations, len(self._active_realizations)
            )
        ):
            raise ErtRunError(
                "Too many simulations have failed! You can add/adjust MIN_REALIZATIONS "
                + "to allow failures in your simulations."
            )

    def _checkMinimumActiveRealizations(self, run_context: ErtRunContext) -> None:
        active_realizations = self._count_active_realizations(run_context)
        if (
            not self.ert()
            .analysisConfig()
            .haveEnoughRealisations(active_realizations, len(self._active_realizations))
        ):
            raise ErtRunError(
                "Number of active realizations is less than the specified "
                + "MIN_REALIZATIONS in the config file"
            )

    def _count_active_realizations(self, run_context: ErtRunContext) -> int:
        return sum(run_context.get_mask())

    def run_ensemble_evaluator(
        self, run_context: ErtRunContext, ee_config: EvaluatorServerConfig
    ) -> int:
        if run_context.get_step():
            self.ert().eclConfig().assert_restart()

        ensemble = EnsembleBuilder.from_legacy(
            run_context,
            self.get_forward_model(),
            self._queue_config,
            self.ert().analysisConfig(),
            self.ert().resConfig(),
        ).build()

        self.ert().initRun(run_context)

        totalOk = EnsembleEvaluator(
            ensemble,
            ee_config,
            run_context.get_iter(),
            ee_id=str(uuid.uuid1()).split("-", maxsplit=1)[0],
        ).run_and_get_successful_realizations()

        for iens, run_arg in enumerate(run_context):
            if run_context.is_active(iens):
                if run_arg.run_status in (
                    RunStatusType.JOB_LOAD_FAILURE,
                    RunStatusType.JOB_RUN_FAILURE,
                ):
                    run_context.deactivate_realization(iens)

        run_context.get_sim_fs().fsync()
        return totalOk

    def get_forward_model(self) -> ForwardModel:
        return self.ert().resConfig().model_config.getForwardModel()

    def get_run_context(self) -> Optional[ErtRunContext]:
        return self._run_context

    @feature_enabled("new-storage")
    def _post_ensemble_data(self, update_id: Optional[str] = None) -> str:
        self.setPhaseName("Uploading data...")
        ensemble_id = post_ensemble_data(
            ert=self.facade,
            ensemble_size=self._ensemble_size,
            update_id=update_id,
            active_realizations=self._active_realizations,
        )
        self.setPhaseName("Uploading done")
        return ensemble_id

    @feature_enabled("new-storage")
    def _post_ensemble_results(self, ensemble_id: str) -> None:
        self.setPhaseName("Uploading results...")
        post_ensemble_results(ert=self.facade, ensemble_id=ensemble_id)
        self.setPhaseName("Uploading done")

    @feature_enabled("new-storage")
    def _post_update_data(self, parent_ensemble_id: str, algorithm: str) -> str:
        self.setPhaseName("Uploading update...")
        update_id = post_update_data(
            ert=self.facade,
            parent_ensemble_id=parent_ensemble_id,
            algorithm=algorithm,
        )
        self.setPhaseName("Uploading done")
        return update_id
