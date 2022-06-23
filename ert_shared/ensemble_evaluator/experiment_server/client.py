import time
import asyncio
import grpc
from functools import singledispatchmethod
from queue import SimpleQueue
import experimentserver_pb2_grpc
from experimentserver_pb2 import (Status,
                                  Job, JobId,
                                  Step, StepId,
                                  Realization, RealizationId,
                                  Ensemble, EnsembleId,
                                  Experiment, ExperimentId,
                                  ClientMessage
                                )
                        
from google.protobuf.json_format import MessageToJson

class Client(experimentserver_pb2_grpc.ExperimentserverStub):
    def __init__(self, channel):
        super().__init__(channel)

    @singledispatchmethod
    def _message(self, id) -> ClientMessage:
        pass

    @_message.register
    def _(self, id: ExperimentId) -> ClientMessage:
        return ClientMessage(experiment=id)

    @_message.register
    def _(self, id: JobId) -> ClientMessage:
        return ClientMessage(job=id)

    @singledispatchmethod
    async def monitor(self, id):
        async for resp in self.client((self._message(id))):
            yield resp

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
            client = Client(channel)
            try:
                async for resp in client.monitor(ExperimentId(id="test-experiment")):
                    print(f"{MessageToJson(resp)}")
            except grpc.RpcError as ex:
                print(f"Server closed with '{ex.details()}'")
            except Exception as ex:
                print(f"Unexpected exception {ex}")

    print("Client done")

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

