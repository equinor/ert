import asyncio
import threading

import websockets
from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from _ert_job_runner.client import Client
from ert.ensemble_evaluator import Ensemble, identifiers
from ert.ensemble_evaluator._builder._realization import Realization
from ert.ensemble_evaluator._builder._step import LegacyJob, LegacyStep


def _mock_ws(host, port, messages, delay_startup=0):
    loop = asyncio.new_event_loop()
    done = loop.create_future()

    async def _handler(websocket, path):
        while True:
            msg = await websocket.recv()
            messages.append(msg)
            if msg == "stop":
                done.set_result(None)
                break

    async def _run_server():
        await asyncio.sleep(delay_startup)
        async with websockets.server.serve(
            _handler, host, port, ping_timeout=1, ping_interval=1
        ):
            await done

    loop.run_until_complete(_run_server())
    loop.close()


def send_dispatch_event(client, event_type, source, event_id, data, **extra_attrs):
    event1 = CloudEvent(
        {"type": event_type, "source": source, "id": event_id, **extra_attrs}, data
    )
    client.send(to_json(event1))


class TestEnsemble(Ensemble):
    __test__ = False

    def __init__(self, _iter, reals, steps, jobs, id_):
        self.iter = _iter
        self.test_reals = reals
        self.steps = steps
        self.jobs = jobs
        self.fail_jobs = []
        self.result = None
        self.result_datacontenttype = None
        self.fails = False

        the_reals = [
            Realization(
                real_no,
                steps=[
                    LegacyStep(
                        id_=step_no,
                        jobs=[
                            LegacyJob(
                                id_=job_no,
                                index=job_no,
                                name=f"job-{job_no}",
                                ext_job=None,
                            )
                            for job_no in range(0, jobs)
                        ],
                        name=f"step-{step_no}",
                        max_runtime=0,
                        ensemble_config=None,
                        num_cpu=0,
                        run_path=None,
                        run_arg=None,
                        job_name=None,
                        job_script=None,
                    )
                    for step_no in range(0, steps)
                ],
                active=True,
            )
            for real_no in range(0, reals)
        ]
        super().__init__(the_reals, {}, id_)

    def _evaluate(self, url):
        event_id = 0
        with Client(url + "/dispatch") as dispatch:
            send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_ENSEMBLE_STARTED,
                f"/ert/ensemble/{self.id_}",
                f"event-{event_id}",
                None,
            )
            if self.fails:
                event_id = event_id + 1
                send_dispatch_event(
                    dispatch,
                    identifiers.EVTYPE_ENSEMBLE_FAILED,
                    f"/ert/ensemble/{self.id_}",
                    f"event-{event_id}",
                    None,
                )
                return

            event_id = event_id + 1
            for real in range(0, self.test_reals):
                for step in range(0, self.steps):
                    job_failed = False
                    send_dispatch_event(
                        dispatch,
                        identifiers.EVTYPE_FM_STEP_UNKNOWN,
                        f"/ert/ensemble/{self.id_}/real/{real}/step/{step}",
                        f"event-{event_id}",
                        None,
                    )
                    event_id = event_id + 1
                    for job in range(0, self.jobs):
                        send_dispatch_event(
                            dispatch,
                            identifiers.EVTYPE_FM_JOB_RUNNING,
                            f"/ert/ensemble/{self.id_}/real/"
                            + f"{real}/step/{step}/job/{job}",
                            f"event-{event_id}",
                            {"current_memory_usage": 1000},
                        )
                        event_id = event_id + 1
                        if self._shouldFailJob(real, step, job):
                            send_dispatch_event(
                                dispatch,
                                identifiers.EVTYPE_FM_JOB_FAILURE,
                                f"/ert/ensemble/{self.id_}/real/"
                                + f"{real}/step/{step}/job/{job}",
                                f"event-{event_id}",
                                {},
                            )
                            event_id = event_id + 1
                            job_failed = True
                            break
                        send_dispatch_event(
                            dispatch,
                            identifiers.EVTYPE_FM_JOB_SUCCESS,
                            f"/ert/ensemble/{self.id_}/real/"
                            + f"{real}/step/{step}/job/{job}",
                            f"event-{event_id}",
                            {"current_memory_usage": 1000},
                        )
                        event_id = event_id + 1
                    if job_failed:
                        send_dispatch_event(
                            dispatch,
                            identifiers.EVTYPE_FM_STEP_FAILURE,
                            f"/ert/ensemble/{self.id_}/real/"
                            + f"{real}/step/{step}/job/{job}",
                            f"event-{event_id}",
                            {},
                        )
                        event_id = event_id + 1
                    else:
                        send_dispatch_event(
                            dispatch,
                            identifiers.EVTYPE_FM_STEP_SUCCESS,
                            f"/ert/ensemble/{self.id_}/real/"
                            + f"{real}/step/{step}/job/{job}",
                            f"event-{event_id}",
                            {},
                        )
                        event_id = event_id + 1

            data = self.result if self.result else None
            extra_attrs = {}
            if self.result_datacontenttype:
                extra_attrs["datacontenttype"] = self.result_datacontenttype
            send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_ENSEMBLE_STOPPED,
                f"/ert/ensemble/{self.id_}",
                f"event-{event_id}",
                data,
                **extra_attrs,
            )

    def join(self):
        self._eval_thread.join()

    def evaluate(self, config):
        self._eval_thread = threading.Thread(
            target=self._evaluate,
            args=(config.dispatch_uri,),
            name="TestEnsemble",
        )

    def start(self):
        self._eval_thread.start()

    def _shouldFailJob(self, real, step, job):
        return (real, 0, step, job) in self.fail_jobs

    def addFailJob(self, real, step, job):
        self.fail_jobs.append((real, 0, step, job))

    def with_result(self, result, datacontenttype):
        self.result = result
        self.result_datacontenttype = datacontenttype
        return self

    def with_failure(self):
        self.fails = True
        return self


class AutorunTestEnsemble(TestEnsemble):
    # pylint: disable=arguments-differ
    def _evaluate(self, client_url, dispatch_url):
        super()._evaluate(dispatch_url)
        with Client(client_url) as client:
            client.send(
                to_json(
                    CloudEvent(
                        {
                            "type": identifiers.EVTYPE_EE_USER_DONE,
                            "source": f"/ert/ensemble/{self.id_}",
                            "id": "event-user-done",
                        }
                    )
                )
            )

    def evaluate(self, config):
        self._eval_thread = threading.Thread(
            target=self._evaluate,
            args=(config.client_uri, config.dispatch_uri),
            name="AutorunTestEnsemble",
        )

        self._eval_thread.start()

    def cancel(self):
        pass

    @property
    def cancellable(self) -> bool:
        return True
