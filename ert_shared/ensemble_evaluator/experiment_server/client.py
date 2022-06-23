import time
import asyncio
import grpc
from functools import singledispatchmethod
from queue import SimpleQueue
import experimentserver_pb2_grpc
from experimentserver_pb2 import JobId, StepId, RealizationId, EnsembleId, ExperimentId, ClientMessage
                        
from google.protobuf.json_format import MessageToJson

class Client(experimentserver_pb2_grpc.ExperimentserverStub):
    def __init__(self, channel):
        super().__init__(channel)

    """
    This is just to re-use Client for all known objects-types.
    I.e. it resolves the issue of wrapping an object in a ClientMessage
    (using a oneof-group).

    It can alternatively be achieved by subclassing Client with e.g.
    ExperimentClient, JobClient etc, or by separate monitor-methods for each
    object-type like monitor_experiment(), monitor_job() etc (and probably
    there are other approaches).
    """
    @singledispatchmethod
    def _message(self, id) -> ClientMessage:
        pass
    @_message.register
    def _(self, id: ExperimentId) -> ClientMessage:
        return ClientMessage(experiment=id, delay=5) # Note: 5s delay (for demonstration)
    @_message.register
    def _(self, id: EnsembleId) -> ClientMessage:
        return ClientMessage(ensemble=id)
    @_message.register
    def _(self, id: RealizationId) -> ClientMessage:
        return ClientMessage(realization=id)
    @_message.register
    def _(self, id: StepId) -> ClientMessage:
        return ClientMessage(step=id)
    @_message.register
    def _(self, id: JobId) -> ClientMessage:
        return ClientMessage(job=id)


    async def monitor(self, id):
        # TODO: we might also want to transfer metadata to identify the client
        # for logging-purposes on the server by using context.invocation_metadata()
        # See https://github.com/grpc/grpc/blob/master/examples/python/metadata/metadata_client.py
        #
        async for resp in self.client(self._message(id)):
            yield resp


async def main():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
    # Alternatively, if you want a synchronous client
    #with grpc.insecure_channel("localhost:50051") as channel:
        try:
            await channel.channel_ready()
            # Alternatively, if you want a synchronous client
            #grpc.channel_ready_future(channel).result(timeout=5)
        except grpc.FutureTimeoutError:
            sys.exit('Error connecting to server')
        else:
            print("Connected to server...")
            client = Client(channel)
            try:
                # Drop "async" if using a synchronous channel
                async for resp in client.monitor(ExperimentId(id="test-experiment")):
                    print(f"{MessageToJson(resp)}")
            except grpc.RpcError as ex:
                print(f"Server closed with '{ex.details()}'")
            except Exception as ex:
                print(f"Unexpected exception {ex}")

    print("Client done")

if __name__=='__main__':
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

