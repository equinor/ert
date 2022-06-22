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
                                )
                        
from google.protobuf.json_format import MessageToJson

class ExperimentProxy(experimentserver_pb2_grpc.ExperimentserverStub):
    def __init__(self, channel, exp_id: str):
        super().__init__(channel)
        #self._channel: channel

        self._experiment = Experiment(id=ExperimentId(id=exp_id))
        self._current_state = self._experiment.SerializeToString(deterministic=True)
        self._response_stream = self.connect_experiment(self.updates())

    @property
    def done(self):
        return self._experiment.status in (
                Status.DONE, Status.FAILED, Status.CANCELLED
            )

#    def add_ensemble(self, ensemble):
    async def receiver(self):
        async for resp in self._response_stream:
            print(f"Received {resp}")
        print("Receiver done")

    async def updates(self) -> Experiment:
        yield self._experiment
        while not self.done:
            print(f"PRE  STATE: {MessageToJson(self._experiment)}")
            #time.sleep(5)
            await asyncio.sleep(5)
            new_state = self._experiment.SerializeToString(deterministic=True)
            if new_state == self._current_state:
                print("EXP State not changed")
            else:
                yield self._experiment
                #self._send_queue.put(JobMessage(state=new_state))
                print(f"POST STATE: {MessageToJson(self._experiment)}")
                self._current_state = new_state

class JobProxy(experimentserver_pb2_grpc.ExperimentserverStub):
    def __init__(self, channel, id: JobId):
        super().__init__(channel)
        #self._channel: channel

        self._job: Job = Job(id=id)
        self._current_state = self._job.SerializeToString(deterministic=True)
        self._response_stream = self.connect_job(self.updates())

    @property
    def done(self):
        return self._job.status in (
                Status.DONE, Status.FAILED, Status.CANCELLED
            )

    def start(self):
        self._job.status = Status.STARTED
        self._job.start_time = time.time()
    def cancel(self):
        self._job.status = Status.CANCELLED
        self._job.end_time = time.time()
    def finish(self):
        self._job.status = Status.DONE
        self._job.end_time = time.time()
    def memory(self, mem: int):
        self._job.current_memory = mem
        self._job.max_memory = max(mem, self._job.max_memory)

    
    async def receiver(self):
        async for resp in self._response_stream:
            print(f"Received {resp}")
        print("Receiver done")

    async def updates(self) -> Job:
        yield self._job
        while not self.done:
            print(f"PRE  STATE: {MessageToJson(self._job)}")
            #time.sleep(1)
            await asyncio.sleep(1)
            new_state = self._job.SerializeToString(deterministic=True)
            if new_state == self._current_state:
                print("JOB State not changed")
            else:
                print(f"POST STATE: {MessageToJson(self._job)}")
                self._current_state = new_state
                yield self._job


async def play(job: Job, delay):
    try:
        print("a")
        await asyncio.sleep(delay)
        print("b")
        job.start()
        await asyncio.sleep(delay)
        print("c")
        currentMemory = 0
        for _ in range(5):
            currentMemory += 1000
            job.memory(currentMemory)
            await asyncio.sleep(delay)
        job.memory(int(currentMemory/3))
        await asyncio.sleep(delay)
        job.finish()
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        print("Exited prematurely")


async def main():
    #with grpc.insecure_channel("localhost:50051") as channel:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        try:
            await channel.channel_ready()
            #grpc.channel_ready_future(channel).result(timeout=5)
        except grpc.FutureTimeoutError:
            sys.exit('Error connecting to server')
        else:
            print("Connected to server...")
            experiment = ExperimentProxy(channel, exp_id="test-experiment")
    
            job = JobProxy(channel, id=JobId(
                                    step=StepId(
                                        realization=RealizationId(
                                            ensemble=EnsembleId(experiment=experiment._experiment.id)))))
    
            exp_task = asyncio.create_task(experiment.receiver())
            job_task = asyncio.create_task(job.receiver())
            play_task = asyncio.create_task(play(job, delay=1))
    
            try:
                results = await asyncio.gather(
                        exp_task,
                        job_task,
                        play_task,
                    return_exceptions=False
                )
            except grpc.RpcError as ex:
                print(f"Server closed with '{ex.details()}'")
            except Exception as ex:
                print(f"Unexpected exception {ex}")

    print("FINITO")

if __name__=='__main__':
    import time
    import sys
    realization = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except Exception as ex:
        print(ex)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

