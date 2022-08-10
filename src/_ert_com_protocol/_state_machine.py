# pylint: disable=no-member # might be fixed when we migrate to proto3
import logging
from functools import singledispatchmethod
from typing import Optional, Union

import _ert_com_protocol._schema_pb2 as pb2

logger = logging.getLogger(__name__)


class _StateHandle:
    """The :class:`_StateHandle` implements a mechanism for handling the protobuf
    defined state machine.
    General note:
    We cannot directly assign values to a protobuf-map; eg:
    experiment.ensembles[ens_id] = Ensemble(id=ens_id)
    Useful links:
    https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/struct
    https://developers.google.com/protocol-buffers/docs/reference/python-generated#map-fields
    """

    def __init__(self) -> None:
        self._experiment: Optional[pb2.Experiment] = None

    def get_experiment(self, _id: pb2.ExperimentId) -> pb2.Experiment:
        logger.debug(f"experiment id: {_id}")
        if self._experiment:
            if self._experiment.id.id != _id.id:
                raise ExperimentStateMachine.IllegalStateUpdate(
                    f"Wrong experiment: expected {self._experiment.id.id} got {_id.id}"
                )
        else:
            self._experiment = pb2.Experiment(id=_id)
        return self._experiment

    def get_ensemble(self, _id: pb2.EnsembleId) -> pb2.Ensemble:
        logger.debug(f"ensemble id: {_id}")
        experiment: pb2.Experiment = self.get_experiment(_id.experiment)
        if _id.id not in experiment.ensembles:
            experiment.ensembles[_id.id].CopyFrom(pb2.Ensemble(id=_id))
        return experiment.ensembles[_id.id]

    def get_realization(self, _id: pb2.RealizationId) -> pb2.Realization:
        logger.debug(f"realization id: {_id}")
        ensemble: pb2.Ensemble = self.get_ensemble(_id.ensemble)
        if _id.realization not in ensemble.realizations:
            ensemble.realizations[_id.realization].CopyFrom(pb2.Realization(id=_id))
        return ensemble.realizations[_id.realization]

    def get_step(self, _id: pb2.StepId) -> pb2.Step:
        logger.debug(f"step id: {_id}")
        real: pb2.Realization = self.get_realization(_id.realization)
        if _id.step not in real.steps:
            real.steps[_id.step].CopyFrom(pb2.Step(id=_id))
        return real.steps[_id.step]

    def get_job(self, _id: pb2.JobId) -> pb2.Job:
        logger.debug(f"job id: {_id}")
        step: pb2.Step = self.get_step(_id.step)
        if _id.index not in step.jobs:
            step.jobs[_id.index].CopyFrom(pb2.Job(id=_id))
        return step.jobs[_id.index]


class ExperimentStateMachine:
    """The :class:`ExperimentStateMachine` implements a state machine for the
    entire experiment. It allows an experiment to track its own state and
    communicate it to others."""

    def __init__(self) -> None:
        self._state_handle: _StateHandle = _StateHandle()

    def successful_realizations(self, ensemble_id: str) -> int:
        """Return an integer indicating the number of successful realizations
        in the experiment given its ensemble ``id``.
        return 0 if the ensemble has no successful realizations or ensemble has
        no realizations registered."""

        if ensemble_id not in self.state.ensembles:
            return 0
        return sum(
            self.state.ensembles[ensemble_id].realizations[id_real].status
            == pb2.STEP_SUCCESS
            for id_real in self.state.ensembles[ensemble_id].realizations
        )

    class IllegalStateUpdate(Exception):
        def __init__(self, reason: str):
            super().__init__(reason)
            self.reason: str = reason

    class UninitializedState(IllegalStateUpdate):
        pass

    @singledispatchmethod
    def _update(
        self,
        _: Union[pb2.Job, pb2.Step, pb2.Realization, pb2.Ensemble, pb2.Experiment],
    ) -> None:
        pass

    @_update.register
    def _(self, job: pb2.Job) -> None:
        logger.debug(
            f"Updating job {job.id.index=} "
            f"status {pb2.JobStatus.Name(job.status)=} from pbuf!",
        )
        old_job = self._state_handle.get_job(job.id)
        old_job.MergeFrom(job)

    @_update.register
    def _(self, real: pb2.Realization) -> None:
        logger.debug(
            f"Updating realization id {real.id.realization} "
            f" statius {pb2.StepStatus.Name(real.status)} from pbuf!",
        )
        old_real = self._state_handle.get_realization(real.id)
        old_real.MergeFrom(real)

    @_update.register
    def _(self, step: pb2.Step) -> None:
        logger.debug(
            f"Updating step id {step.id.step} status "
            f"{pb2.StepStatus.Name(step.status)} from pbuf!",
        )
        old_step = self._state_handle.get_step(step.id)
        old_step.MergeFrom(step)
        # if step==success then set realization=success
        if step.status == pb2.STEP_SUCCESS:
            self._update(
                pb2.Realization(id=step.id.realization, status=pb2.STEP_SUCCESS),
            )

    @_update.register
    def _(self, ens: pb2.Ensemble) -> None:
        logger.debug(
            f"Updating step id {ens.id.id} status"
            f"{pb2.EnsembleStatus.Name(ens.status)} from pbuf!",
        )
        old_ens = self._state_handle.get_ensemble(ens.id)
        old_ens.MergeFrom(ens)

    @_update.register
    def _(self, exp: pb2.Experiment) -> None:
        logger.debug(f"Updating experiment id {exp.id.id} from pbuf!")
        old_exp = self._state_handle.get_experiment(exp.id)
        old_exp.MergeFrom(exp)

    async def update(
        self,
        msg: Union[
            pb2.Job,
            pb2.Step,
            pb2.Realization,
            pb2.Ensemble,
            pb2.Experiment,
            pb2.DispatcherMessage,
        ],
    ) -> None:
        try:
            if isinstance(msg, pb2.DispatcherMessage):
                self._update(getattr(msg, str(msg.WhichOneof("object"))))
            else:
                self._update(msg)
        except Exception:
            logger.error(f"Failed state machine update! Current state: {self.state}")
            raise

    @property
    def state(self) -> pb2.Experiment:
        if self._state_handle._experiment is None:
            raise ExperimentStateMachine.UninitializedState(
                "Experiment must be initialized first!"
            )
        return self._state_handle._experiment


def node_status_builder(  # pylint: disable=too-many-arguments
    status: str,
    experiment_id: str,
    ensemble_id: Optional[str] = None,
    realization_id: Optional[int] = None,
    step_id: Optional[int] = None,
    job_id: Optional[int] = None,
) -> pb2.DispatcherMessage:
    """Builds a DispatcherMessage based on the given argument list.
    It decides which Protobuf object to create based on the set of ids.
    It doesn't check whether the status is a valid one, it just raises ValueError
    in case the status was wrong.

    Args:
        status: get status and converts it protobuf status, validity not checked.
        experiment_id: mandatory id of experiment
        ensemble_id: id of ensemble.
        realization_id: realization index.
        step_id: step index.
        job_id: job index.

    Returns:
        Dispatcher message encapsulating the corresponding protobuf object.
    """
    experiment = pb2.ExperimentId(id=experiment_id)
    if ensemble_id is not None:
        ensemble = pb2.EnsembleId(
            id=ensemble_id,
            experiment=experiment,
        )
        if realization_id is not None:
            realization = pb2.RealizationId(
                realization=realization_id,
                ensemble=ensemble,
            )
            if step_id is not None:
                step = pb2.StepId(
                    step=step_id,
                    realization=realization,
                )
                if job_id is not None:
                    job = pb2.JobId(
                        index=job_id,
                        step=step,
                    )
                    return pb2.DispatcherMessage(
                        job=pb2.Job(id=job, status=pb2.JobStatus.Value(status))
                    )
                return pb2.DispatcherMessage(
                    step=pb2.Step(id=step, status=pb2.StepStatus.Value(status))
                )
            return pb2.DispatcherMessage(
                realization=pb2.Realization(
                    id=realization, status=pb2.StepStatus.Value(status)
                )
            )
        return pb2.DispatcherMessage(
            ensemble=pb2.Ensemble(id=ensemble, status=pb2.EnsembleStatus.Value(status))
        )
    return pb2.DispatcherMessage(
        experiment=pb2.Experiment(
            id=experiment, status=pb2.ExperimentStatus.Value(status)
        )
    )
