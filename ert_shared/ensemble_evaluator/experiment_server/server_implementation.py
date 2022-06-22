import asyncio
import queue
import grpc
from functools import singledispatchmethod
import experimentserver_pb2_grpc
from experimentserver_pb2 import Status, Experiment, ExperimentId, Job, JobId

class BaseModel:
    async def _raiseIllegalState(self, old_state, new_state):
            await self._context.abort(grpc.StatusCode.ABORTED,
                                      f"Illegal state-transition from {old_state} to {new_state}")
    def reconnect(self, context):
        self._context = context

    @singledispatchmethod
    async def update(self, upd):
        pass

class JobModel(BaseModel):
    def __init__(self, id: JobId, context):
        self._context = context
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

class ExperimentModel(BaseModel):
    def __init__(self, message: Experiment, context):
        self._experiment: Experiment = message
        self._context = context

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
        print(f"Response: {self._experiment}")
        await self._context.write(self._experiment)

    @BaseModel.update.register
    async def _(self, new_job: Job):
        print(f"Updating {self} with JOB {new_job}")

class ExperimentServer(experimentserver_pb2_grpc.ExperimentserverServicer):
    def __init__(self):
        self._experiments = dict()

    async def connect_experiment(self, request_iter, context):
        print(f"Connect exp...")
        async for experiment in request_iter:
            break
        print(f"Event exp: {experiment}")
        key = experiment.id.SerializeToString(deterministic=True)
        if key in self._experiments:
            print(f"Reconnect exp {experiment}")
            self._experiments[key].reconnect(context)
        else:
            print(f"New experiment: {experiment}")
            self._experiments[key] = ExperimentModel(experiment, context)

        model = self._experiments[key]
        await context.write(model._experiment)

        async for experiment in request_iter:
            print(f"READ exp: {experiment }")
            await model.update(experiment)

        print(f"Disconnect exp {experiment}")



    async def connect_job(self, request_iter, context):
        async for job in request_iter:
            break
        print(f"Event job: {job}")

        exp_key = job.id.step.realization.ensemble.experiment.SerializeToString(deterministic=True)
        if exp_key not in self._experiments:
            raise RuntimeError(f"No corresponding Experiment")

        experiment = self._experiments[exp_key]
        await experiment.update(job)

        async for job in request_iter:
            print(f"READ job: {job}")
            await experiment.update(job)

        print(f"Disconnect job {job}")


async def serve():
    server = grpc.aio.server()
    experimentserver_pb2_grpc.add_ExperimentserverServicer_to_server(ExperimentServer(),
                                                             server)
    server.add_insecure_port("localhost:50051")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
