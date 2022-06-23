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
        #print(f"experimentid: {id}")
        exp_key = id.SerializeToString(deterministic=True)
        if exp_key not in self._experiments:
            print(f"New experiment: {id}")
            self._experiments[exp_key] = Experiment(id=id)
        return self._experiments[exp_key]

    @find_node_by_id.register
    def _(self, id: EnsembleId) -> Ensemble:
        #print(f"emsembleid: {id}")
        experiment: Experiment = self.find_node_by_id(id.experiment)
        if id.id not in experiment.ensembles:
            experiment.ensembles[id.id].CopyFrom(Ensemble(id=id))
        return experiment.ensembles[id.id]

    @find_node_by_id.register
    def _(self, id: RealizationId) -> Realization:
        #print(f"realizationid: {id}")
        ensemble: Ensemble = self.find_node_by_id(id.ensemble)
        if id.realization not in ensemble.realizations:
            ensemble.realizations[id.realization].CopyFrom(Realization(id=id))
        return ensemble.realizations[id.realization]

    @find_node_by_id.register
    def _(self, id: StepId) -> Step:
        #print(f"stepid: {id}")
        real: Realization = self.find_node_by_id(id.realization)
        if id.step not in real.steps:
            real.steps[id.step].CopyFrom(Step(id=id))
        return real.steps[id.step]

    @find_node_by_id.register
    def _(self, id: JobId) -> Job:
        #print(f"jobid: {id}")
        step: Step = self.find_node_by_id(id.step)
        if id.index not in step.jobs:
            step.jobs[id.index].CopyFrom(Job(id=id))
        return step.jobs[id.index]


class StateMachine:
    def __init__(self, repo: ExperimentRepository):
        self._repo: ExperimentRepository = repo


    class IllegalStateTransition(Exception):
        """ Just a marker - replace with anything more suitable if desired """
        def __init__(self, reason: str):
            super().__init__(reason)
            self.reason:str = reason

    class AbortExecution(Exception):
        """ Just a marker - replace with anything more suitable if desired """
        def __init__(self, reason: str):
            super().__init__(reason)
            self.reason:str = reason

    def _ensure_sane_status_update(self, cur_status: Status, new_status: Status):
        if cur_status in (
            Status.DONE, Status.CANCELLED, Status.FAILED
            ):
            raise StateMachine.IllegalStateTransition(
                  f"Illegal state-transition from {cur_status} to {new_status}")

        if new_status == Status.UNKNOWN:
            if cur_status not in (Status.UNKNOWN,):
                raise StateMachine.IllegalStateTransition(
                      f"Illegal state-transition from {cur_status} to {new_status}")

        elif new_status == Status.STARTING:
            if cur_status not in (Status.UNKNOWN, Status.STARTING,):
                raise StateMachine.IllegalStateTransition(
                      f"Illegal state-transition from {cur_status} to {new_status}")

        elif new_status == Status.RUNNING:
            if cur_status not in (Status.STARTING, Status.RUNNING,):
                raise StateMachine.IllegalStateTransition(
                      f"Illegal state-transition from {cur_status} to {new_status}")
        
    @singledispatchmethod
    def update(self, upd, set_value=False):
        pass

    @update.register
    def _(self, new_job: Job, set_value=False):
        if set_value:
            # if a dispatcher re-connects or connects after running for some time
            # this flag will be set and we just stuff the job into the repo without
            # verifying state-transitions or such
            print(f"SET JOB {new_job}")
            step: Step = self._repo.find_node_by_id(new_job.id.step)
            step.jobs[new_job.id.index].CopyFrom(new_job)
        else:
            old_job = self._repo.find_node_by_id(new_job.id)
            print(f"Updating JOB {old_job} with {new_job}")
            self._ensure_sane_status_update(old_job.status, new_job.status)
            old_job.MergeFrom(new_job)

        # Proper signal back if status is set to cancel of fail
        if new_job.status in (Status.CANCELLED, Status.FAILED):
            raise StateMachine.AbortExecution(f"Status set to {new_job.status}")

        # TODO: propagate upwards


    @update.register
    def _(self, new_real: Realization, set_value=False):
        old_real = self._repo.find_node_by_id(new_real.id)
        print(f"Updating REAL {old_real} with {new_real}")
#        self._ensure_sane_status_update(old_real.status, new_real.status)
        old_real.MergeFrom(new_real)
        
        # TODO: propagate down to jobs if cancel

    @update.register
    def _(self, new_exp: Experiment, set_value=False):
        # the set_value flag is not useful here I believe...
        old_exp = self._repo.find_node_by_id(new_exp.id)
        print(f"Updating EXP {old_exp} with {new_exp}")
#        self._ensure_sane_status_update(old_exp.status, new_exp.status)
        old_exp.MergeFrom(new_exp)


class ExperimentServer(experimentserver_pb2_grpc.ExperimentserverServicer):
    continue_msg = ContinueOrAbortReply(reply=ContinueOrAbortReply.Reply.CONTINUE)
    abort_msg    = ContinueOrAbortReply(reply=ContinueOrAbortReply.Reply.ABORT)
    def __init__(self):
        self._repo: ExperimentRepository = ExperimentRepository()
        self._states: StateMachine = StateMachine(self._repo)

    async def dispatch(self, request_iter, context):
        context.add_done_callback(lambda _: print("Dispatcher disconnected"))
        initial_update = True
        async for msg in request_iter:
            #print(f"Dispatch: {msg}")
            #print(f"type={msg.WhichOneof('object')}")
            try:
                obj = getattr(msg, msg.WhichOneof('object'))
                self._states.update(obj, set_value=initial_update)
                await context.write(self.continue_msg)
                
                initial_update = False

            except StateMachine.AbortExecution as ex:
                await context.write(self.abort_msg)

            except StateMachine.IllegalStateTransition as ex:
                # TODO: Should we be more forgiving here and perhaps leave the
                # channel open (i.e. write an abort_msg like above) ?
                await context.abort(grpc.StatusCode.ABORTED, details=ex.reason)

    async def client(self, msg, context):
        print(f"Client connect, delay is {msg.delay}s")
        context.add_done_callback(lambda _: print("Client disconnected"))

        id = getattr(msg, msg.WhichOneof('objectid'))
        node = self._repo.find_node_by_id(id)
        print(f"Found {MessageToJson(node)}")
        while True:
            await asyncio.sleep(msg.delay)
            await context.write(DispatcherMessage(experiment=node))


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        server = grpc.aio.server()
        experimentserver_pb2_grpc.add_ExperimentserverServicer_to_server(
                    ExperimentServer(),
                    server)

        server.add_insecure_port("localhost:50051")
        loop.run_until_complete(server.start())
        loop.run_until_complete(server.wait_for_termination())
    except Exception as ex:
        print(ex)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
