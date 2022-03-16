import time
import uuid
from typing import Optional, List, Union
import asyncio

from ecl.util.util import BoolVector
from res.enkf import EnKFMain, QueueConfig
from res.enkf.ert_run_context import ErtRunContext
from res.job_queue import (
    RunStatusType,
    ForwardModel,
)
from ert_shared.libres_facade import LibresFacade
from ert_shared.storage.extraction import (
    post_ensemble_data,
    post_ensemble_results,
    post_update_data,
)
from ert_shared.feature_toggling import feature_enabled
from ert_shared.ensemble_evaluator.ensemble.builder import (
    create_ensemble_builder_from_legacy,
)
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.models.types import Argument


class ErtRunError(Exception):
    pass


class BaseRunModel:
    def __init__(self, ert: EnKFMain, queue_config: QueueConfig, phase_count: int = 1):
        """

        Parameters
        ----------
        queue_config : QueueConfig
        phase_count : Optional[int], optional
            Number of data assimilation cycles / iterations an experiment will have, by default 1
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
        self.initial_realizations_mask: List[int] = []
        self.completed_realizations_mask: List[int] = []
        self.support_restart: bool = True
        self._run_context: Optional[ErtRunContext] = None
        self._last_run_iteration: int = -1
        self._ert = ert
        self.facade = LibresFacade(ert)
        self.reset()

    def ert(self) -> EnKFMain:
        return self._ert

    @property
    def _ensemble_size(self) -> int:
        return len(self.initial_realizations_mask)

    @property
    def _active_realizations(self) -> List[int]:
        realization_mask = self.initial_realizations_mask
        return [idx for idx, mask_val in enumerate(realization_mask) if mask_val]

    def reset(self) -> None:
        self._failed = False
        self._phase = 0

    def start_simulations_thread(
        self, arguments: Argument, evaluator_server_config: EvaluatorServerConfig
    ) -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.startSimulations(
            arguments=arguments, evaluator_server_config=evaluator_server_config
        )

    def startSimulations(
        self, arguments: Argument, evaluator_server_config: EvaluatorServerConfig
    ) -> None:
        try:
            self.initial_realizations_mask = arguments["active_realizations"]
            run_context = self.runSimulations(
                arguments, evaluator_server_config=evaluator_server_config
            )
            self.completed_realizations_mask = run_context.get_mask()
        except ErtRunError as e:
            self.completed_realizations_mask = BoolVector(default_value=False)
            self._failed = True
            self._fail_message = str(e)
            self._simulationEnded()
        except UserWarning as e:
            self._fail_message = str(e)
            self._simulationEnded()
        except Exception as e:
            self._failed = True
            self._fail_message = str(e)
            self._simulationEnded()
            raise

    def runSimulations(
        self, arguments: Argument, evaluator_server_config: EvaluatorServerConfig
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
        elif (
            not self.ert()
            .analysisConfig()
            .haveEnoughRealisations(
                num_successful_realizations, len(self._active_realizations)
            )
        ):
            raise ErtRunError(
                "Too many simulations have failed! You can add/adjust MIN_REALIZATIONS to allow failures in your simulations."
            )

    def checkMinimumActiveRealizations(self, run_context: ErtRunContext) -> None:
        active_realizations = self.count_active_realizations(run_context)
        if (
            not self.ert()
            .analysisConfig()
            .haveEnoughRealisations(active_realizations, len(self._active_realizations))
        ):
            raise ErtRunError(
                "Number of active realizations is less than the specified MIN_REALIZATIONS in the config file"
            )

    def count_active_realizations(self, run_context: ErtRunContext) -> int:
        return sum(run_context.get_mask())

    def run_ensemble_evaluator(
        self, run_context: ErtRunContext, ee_config: EvaluatorServerConfig
    ) -> int:
        if run_context.get_step():
            self.ert().eclConfig().assert_restart()

        ensemble = create_ensemble_builder_from_legacy(
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
            ee_id=str(uuid.uuid1()).split("-")[0],
        ).run_and_get_successful_realizations()

        for i in range(len(run_context)):
            if run_context.is_active(i):
                run_arg = run_context[i]
                if (
                    run_arg.run_status == RunStatusType.JOB_LOAD_FAILURE
                    or run_arg.run_status == RunStatusType.JOB_RUN_FAILURE
                ):
                    run_context.deactivate_realization(i)

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
