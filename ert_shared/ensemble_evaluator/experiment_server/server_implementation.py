import asyncio
import queue
import grpc
import experimentserver_pb2_grpc
from experimentserver_pb2 import (RunState,
                                  EnsembleId, EnsembleState, EnsembleMessage,
                                  JobId, JobState, JobMessage)

class JobModel:
    def __init__(self, id: JobId, context):
        self._jobid: JobId = id
        self._context = context

        self._state:JobState = JobState()

    def reconnect(self, context):
        self._context = context

    @property
    def state(self) -> JobState:
        return self._state

    @property
    def id(self)  -> JobId:
        return self._jobid

    def cancel(self):
        self.update(JobState(runstate=RunState.CANCELLED))

    async def _raiseIllegalState(self, new_state):
            await self._context.abort(grpc.StatusCode.ABORTED,
                                      f"Illegal state-transition from {self.state.runstate} to {new_state.runstate}")

    async def update(self, new_state: JobState):
        print(f"Updating {self.state} with {new_state}")
        cur_state = self.state.runstate
        if cur_state in (
            RunState.DONE, RunState.CANCELLED, RunState.FAILED
            ):
            await self._raiseIllegalState(new_state)

        if new_state.runstate == RunState.UNKNOWN:
            if cur_state not in (RunState.UNKNOWN,):
                await self._raiseIllegalState(new_state)

        elif new_state.runstate == RunState.STARTED:
            if cur_state not in (RunState.UNKNOWN, RunState.STARTED,):
                await self._raiseIllegalState(new_state)

        elif new_state.runstate == RunState.RUNNING:
            if cur_state not in (RunState.STARTED, RunState.RUNNING,):
                await self._raiseIllegalState(new_state)

        self.state.MergeFrom(new_state)
        print(f"Response: {self.state}")
        await self._context.write(self.state)


class ExperimentServer(experimentserver_pb2_grpc.ExperimentserverServicer):
    def __init__(self):
        self._jobs = dict()

    # TODO: merge into connect_job. No value in separate method
    async def _read(self, request_iter, job: JobModel):
        async for client_update in request_iter:
            print(f"READ: {client_update}")
            if "state" == client_update.WhichOneof("event"):
                print(f"Received state: {client_update.state}")
                await job.update(client_update.state)
            else:
                raise RuntimeError(f"Unexpected event {client_update}")

    async def connect_job(self, request_iter, context):
        async for client_update in request_iter:
            break
        print(f"Event: {client_update}")
        if "id" == client_update.WhichOneof("event"):
            key = client_update.SerializeToString(deterministic=True)
            if key in self._jobs:
                print(f"Reconnect {client_update}")
                self._jobs[key].reconnect(context)
            else:
                print(f"New Job: {client_update}")
                self._jobs[key] = JobModel(client_update.id, context)
                
            await context.write(self._jobs[key].state)
        else:
            raise RuntimeError(f"Must connect first with ID")

        # Pass iterator on to read-method
        # TODO: merge...
        await self._read(request_iter, self._jobs[key])
        print(f"Disconnect {client_update}")


async def serve():
    server = grpc.aio.server()
    experimentserver_pb2_grpc.add_ExperimentserverServicer_to_server(ExperimentServer(),
                                                             server)
    server.add_insecure_port("localhost:50051")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
