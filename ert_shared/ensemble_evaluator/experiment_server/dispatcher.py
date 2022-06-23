import time
import asyncio
import grpc
from queue import SimpleQueue
import experimentserver_pb2_grpc
from experimentserver_pb2 import (Status,
                                  Job, JobId,
                                  Step, StepId,
                                  Realization, RealizationId,
                                  Ensemble, EnsembleId,
                                  Experiment, ExperimentId,
                                  DispatcherMessage, ContinueOrAbortReply
                                )
                        
from google.protobuf.json_format import MessageToJson

class JobProxy(experimentserver_pb2_grpc.ExperimentserverStub):
    def __init__(self, channel, id: JobId):
        super().__init__(channel)
        self._job: Job = Job(id=id)
        
        self._start_event: asyncio.Event = asyncio.Event()

        self._current_state = self._job.SerializeToString(deterministic=True)
        self._response_stream = self.dispatch(self.updates())

    @property
    def done(self):
        return self._job.status in (
                Status.DONE, Status.FAILED, Status.CANCELLED
            )

    async def start(self):
        self._job.status = Status.STARTING
        self._job.start_time = time.time()
        
        await self._start_event.wait()

        time.sleep(0.5)  # TODO: Here the actual job should be started
        self._job.status = Status.RUNNING
    def cancel(self):
        self._job.status = Status.CANCELLED
        self._job.end_time = time.time()
    def finish(self):
        self._job.status = Status.DONE
        self._job.end_time = time.time()
    def memory(self, mem: int):
        self._job.current_memory = mem
        self._job.max_memory = max(mem, self._job.max_memory)

    async def responses(self):
        async for resp in self._response_stream:
            yield resp

    async def updates(self) -> DispatcherMessage:
        """
        This just implements a timer pushing full state for the job every second
        (if something has changed). It can easily be modified to push only delta,
        but I'm not sure it's worth the effort.

        The server uses MergeFrom() and will work unchanged if the dispatcher send
        deltas, unless it disconnects in the meantime. If it disconnects andt then
        re-connects, the server assumes whatever it receives is the complete state
        of the job, and deltas won't work
        """
        yield DispatcherMessage(job=self._job)
        while not self.done:
#            print(f"PRE : {MessageToJson(self._job)}")
            await asyncio.sleep(1)
            new_state = self._job.SerializeToString(deterministic=True)
            if new_state == self._current_state:
                print("JOB not changed")
            else:
                print(f"Dispatching: {MessageToJson(self._job)}")
                self._current_state = new_state
                yield DispatcherMessage(job=self._job)

            # TODO: remove! this is for poc/testing only!
            # This delays state-transition from STARTING to RUNNING on the
            # client-side since the server will yell otherwise. Relax that on
            # the server and remove this here!
            if self._job.status == Status.STARTING:
                self._start_event.set()


async def play(job: Job, delay):
    try:
        await asyncio.sleep(delay)
        await job.start()
        await asyncio.sleep(delay)
#        job.start() # this makes server respond with IllegalStateTransition
        job.cancel()
        currentMemory = 0
        for _ in range(5):
            currentMemory += 1000
            job.memory(currentMemory)
            await asyncio.sleep(delay)
#        job.cancel()
        await asyncio.sleep(delay)
        job.memory(int(currentMemory/3))
        await asyncio.sleep(delay)
        job.finish()
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        print("Exited prematurely")


async def main(job_idx):
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        try:
            await channel.channel_ready()
        except grpc.FutureTimeoutError:
            sys.exit('Error connecting to server')
        else:
            print("Connected to server...")
            job = JobProxy(channel, id=JobId(
                                      step=StepId(
                                        realization=RealizationId(
                                          ensemble=EnsembleId(
                                              experiment=ExperimentId(
                                                  id="test-experiment"),
                                               id="first ensemble"),
                                          realization=0),
                                        step=0),
                                      index=job_idx))

            play_task = asyncio.create_task(play(job, delay=1))
            try:
                async for response in job.responses():
                    print(f"Response: {response}")
                    
                    if response.reply == ContinueOrAbortReply.Reply.ABORT:
                        job.cancel()

                        # do whatever is needed to stop the dispatcher
                        play_task.cancel()
                        break

                results = await asyncio.gather(
                        play_task,
                    return_exceptions=False
                )
            except grpc.RpcError as ex:
                print(f"Server closed with '{ex.details()}'")
            except Exception as ex:
                print(f"Unexpected exception {ex}")

    print("Dispatcher finished")

if __name__=='__main__':
    import time
    import sys
    job_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(job_idx))
    except Exception as ex:
        print(ex)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

