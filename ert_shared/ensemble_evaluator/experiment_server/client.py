import asyncio
import grpc
from queue import SimpleQueue
import experimentserver_pb2_grpc
from experimentserver_pb2 import JobId, JobState, JobMessage

from google.protobuf.json_format import MessageToJson


class Job(experimentserver_pb2_grpc.ExperimentserverStub):
    def __init__(self, channel, jobid):
        super().__init__(channel)
        self._jobid = jobid
        self._state = JobState()
        self._current_state = self.state.SerializeToString(deterministic=True)
        self._response_stream = self.connect_job(self.updates())

    @property
    def state(self) -> JobState:
        return self._state

    @property
    def id(self)  -> JobId:
        return self._jobid

    @property
    def done(self):
        return self.state.state not in (
                JobState.State.UNKNOWN, JobState.State.STARTED, JobState.State.RUNNING
            )

    async def receiver(self):
        await asyncio.sleep(0.1)
        try:
            for resp in self._response_stream:
                print(f"Received {resp}")
                await asyncio.sleep(0.1)
        except Exception as ex:
            print(f"Server says {ex}")
        print("Receiver done")

    def updates(self) -> JobMessage:
        yield JobMessage(id=self.id)
        while not self.done:
            print(f"PRE  STATE: {MessageToJson(self.state)}")
            time.sleep(1)
            #await asyncio.sleep(1)
            new_state = self.state.SerializeToString(deterministic=True)
            if new_state == self._current_state:
                print("State not changed")
            else:
                yield JobMessage(state=self.state)
                #self._send_queue.put(JobMessage(state=new_state))
                print(f"POST STATE: {MessageToJson(self.state)}")
                self._current_state = new_state

async def play(job: Job, delay):
    print("a")
    job.state.state = JobState.State.STARTED
    print("b")
    await asyncio.sleep(delay)
    print("c")
    job.state.state = JobState.State.RUNNING
    print("d")
    await asyncio.sleep(delay)
    print("e")
    for _ in range(5):
        job.state.currentMemory += 1000
        await asyncio.sleep(delay)
    job.state.state = JobState.State.DONE
    await asyncio.sleep(delay)


async def main(id="default-experiment"):
    with grpc.insecure_channel("localhost:50051") as channel:
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
        except grpc.FutureTimeoutError:
            sys.exit('Error connecting to server')
        else:
            job = Job(channel, JobId(experimentId=id))
            print("Connected to server...")
    
            recv_task = asyncio.create_task(job.receiver())
            play_task = asyncio.create_task(play(job, delay=1))

            await play_task
            await recv_task
            

if __name__=='__main__':
    import time
    import sys
    jobid = sys.argv[1] if len(sys.argv) > 1 else "default-id"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(id=jobid))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
