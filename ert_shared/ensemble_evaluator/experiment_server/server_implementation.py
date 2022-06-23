import asyncio
import queue
import grpc
from functools import singledispatchmethod
from google.protobuf.json_format import MessageToJson
import experimentserver_pb2_grpc
from experimentserver_pb2 import (Status,
                                  Job, JobId,
                                  Step, StepId,
                                  Realization, RealizationId,
                                  Ensemble, EnsembleId,
                                  Experiment, ExperimentId,
                                  ContinueOrAbortReply, DispatcherMessage
                                  )

class BaseModel:
    async def _raiseIllegalState(self, old_state, new_state):
            await self._context.abort(grpc.StatusCode.ABORTED,
                                      f"Illegal state-transition from {old_state} to {new_state}")
    # def reconnect(self, context):
    #     self._context = context

    @singledispatchmethod
    async def update(self, upd):
        pass

class JobModel(BaseModel):
    def __init__(self, id: JobId):
        self._job: Job = Job(id=id)

    def cancel(self):
        self.update(JobState(runstate=Status.CANCELLED))


    @BaseModel.update.register
    async def _(self, new_job: Job):
        print(f"Updating {self._job} with {new_job}")
        cur_status = self._job.status
        if cur_status in (
            Status.DONE, Status.CANCELLED, Status.FAILED
            ):
            await self._raiseIllegalState(cur_status, new_job.status)

        if new_job.status == Status.UNKNOWN:
            if cur_status not in (Status.UNKNOWN,):
                await self._raiseIllegalState(cur_status, new_job.status)

        elif new_job.status == Status.STARTED:
            if cur_status not in (Status.UNKNOWN, Status.STARTED,):
                await self._raiseIllegalState(cur_status, new_job.status)

        elif new_job.status == Status.RUNNING:
            if cur_status not in (Status.STARTED, Status.RUNNING,):
                await self._raiseIllegalState(cur_status, new_job.status)

        self._job.MergeFrom(new_job)
        print(f"Response: {self._job}")
        await self._context.write(self._job)

class StateMachine(BaseModel):
    def __init__(self, id: ExperimentId):
        self._experiment: Experiment = Experiment(id=id)

    @BaseModel.update.register
    async def _(self, new_experiment: Experiment):
        print(f"Updating {self} with EXP {new_experiment}")
        cur_status = self._experiment.status
        if cur_status in (
            Status.DONE, Status.CANCELLED, Status.FAILED
            ):
            await self._raiseIllegalState(new_experiment)

        if new_experiment.status == Status.UNKNOWN:
            if cur_state not in (Status.UNKNOWN,):
                await self._raiseIllegalState(new_experiment)

        elif new_experiment.status == Status.STARTED:
            if cur_state not in (Status.UNKNOWN, Status.STARTED,):
                await self._raiseIllegalState(new_experiment)

        elif new_experiment.status == Status.RUNNING:
            if cur_state not in (Status.STARTED, Status.RUNNING,):
                await self._raiseIllegalState(new_experiment)

        self._experiment.MergeFrom(new_experiment)
        return True

    @BaseModel.update.register
    async def _(self, new_job: Job):
        print(f"Updating {self} with JOB {new_job}")
        old_job = self.find_node_by_id(new_job.id)
        print(f"   updating {old_job} with {new_job}")


class ExperimentServer(experimentserver_pb2_grpc.ExperimentserverServicer):
#    continue_msg = ContinueOrAbortReply(reply=ContinueOrAbortReply.Reply.CONTINUE)
#    abort_msg    = ContinueOrAbortReply(reply=ContinueOrAbortReply.Reply.ABORT)
    def __init__(self):
        self._repo: ExperimentRepository = ExperimentRepository()

    async def dispatch(self, request_iter, context):
        async for msg in request_iter:
            print(f"Dispatch: {msg}")
            print(f"type={msg.WhichOneof('object')}")
            id = getattr(msg, msg.WhichOneof('object')).id
#            if id is None:
#                context.abort(f"Unknown object: {msg}")
            node = self._repo.find_node_by_id(id)
            print(f"Found {MessageToJson(node)}")

    async def client(self, msg, context):
        print(f"Client connect, delay is {msg.delay}s")
        id = getattr(msg, msg.WhichOneof('objectid'))
#        if id is None:
#            context.abort(f"Unknown object type: {msg}")
        node = self._repo.find_node_by_id(id)
        print(f"Found {MessageToJson(node)}")
        while True:
            await asyncio.sleep(msg.delay)
            await context.write(DispatcherMessage(experiment=node))

        print(f"Client disconnected") # why doesnt this print??

    # async def _handle_experiment_dispatcher(self, experiment: Experiment, context):
    #     exp_key = experiment.id.SerializeToString(deterministic=True)
    #     if exp_key not in self._experiments:
    #         print(f"New experiment: {experiment}")
    #         self._experiments[exp_key] = ExperimentModel(experiment.id)
    #
    #     print(f"Updating with {experiment}")
    #     model = self._experiments[exp_key]
    #     if await model.update(experiment):
    #         await context.write(self.continue_msg)
    #     else:
    #         await context.write(self.abort_msg)
    #
    # async def _handle_job_dispatcher(self, job: Job, context):
    #     exp_key = job.id.step.realization.ensemble.experiment.SerializeToString(deterministic=True)
    #     if exp_key not in self._experiments:
    #         print(f"New experiment: {experiment}")
    #         self._experiments[exp_key] = ExperimentModel(job.id.step.realization.ensemble.experiment)
    #
    #     print(f"Updating with {job}")
    #     model = self._experiments[exp_key]
    #     if await model.update(job):
    #         await context.write(self.continue_msg)
    #     else:
    #         await context.write(self.abort_msg)


class ExperimentRepository:
    """
    About maps and CopyFrom():

    We cannot directly assign values to a protobuf-map like this

        experiment.ensembles[ens_id] = Ensemble(id=ens_id)

    However, there is a workaround using a "well-known-type" called Struct

    https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/struct

    (Example of useage e.g. here https://stackoverflow.com/a/59724212)

    Some performance-testing should be done to figure out the best approach
    but I use the generic CopyFrom for now.
    """
    def __init__(self):
        self._experiments = dict()

    @singledispatchmethod
    def find_node_by_id(self, msg: DispatcherMessage):
        pass

    @find_node_by_id.register
    def _(self, id: ExperimentId) -> Experiment:
        print(f"experimentid: {id}")
        exp_key = id.SerializeToString(deterministic=True)
        if exp_key not in self._experiments:
            print(f"New experiment: {id}")
            self._experiments[exp_key] = Experiment(id=id)
        return self._experiments[exp_key]

    @find_node_by_id.register
    def _(self, id: EnsembleId) -> Ensemble:
        print(f"emsembleid: {id}")
        experiment: Experiment = self.find_node_by_id(id.experiment)
        if id.id not in experiment.ensembles:
            experiment.ensembles[id.id].CopyFrom(Ensemble(id=id))
        return experiment.ensembles[id.id]

    @find_node_by_id.register
    def _(self, id: RealizationId) -> Realization:
        print(f"realizationid: {id}")
        ensemble: Ensemble = self.find_node_by_id(id.ensemble)
        if id.realization not in ensemble.realizations:
            ensemble.realizations[id.realization].CopyFrom(Realization(id=id))
        return ensemble.realizations[id.realization]

    @find_node_by_id.register
    def _(self, id: StepId) -> Step:
        print(f"stepid: {id}")
        real: Realization = self.find_node_by_id(id.realization)
        if id.step not in real.steps:
            real.steps[id.step].CopyFrom(Step(id=id))
        return real.steps[id.step]

    @find_node_by_id.register
    def _(self, id: JobId) -> Job:
        print(f"jobid: {id}")
        step: Step = self.find_node_by_id(id.step)
        if id.index not in step.jobs:
            step.jobs[id.index].CopyFrom(Job(id=id))
        return step.jobs[id.index]


async def serve():
    server = grpc.aio.server()
    experimentserver_pb2_grpc.add_ExperimentserverServicer_to_server(ExperimentServer(),
                                                             server)
    server.add_insecure_port("localhost:50051")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
