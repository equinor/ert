import time
import queue
import flask
import threading
import requests
from ert_shared.ensemble_evaluator.monitor import (
    create_monitor,
    create_event,
    create_event_from_dict,
)

import asyncio
import websockets

_app = flask.Flask(__name__)


class EnsembleEvaluator:
    def __init__(self):
        # super().__init__(test)

        self._url = "http://localhost:5000"
        self._kill_switch = queue.Queue()
        self._queues = {}
        self._events = []

        self._app = flask.Flask("ert ee")
        self._app.add_url_rule("/ping", "ping", self._ping)
        self._app.add_url_rule("/await_event/<ident>", "await_event", self._await_event)
        self._app.add_url_rule("/stop", "stop", self._stop, methods=["POST"])

        self._api_thread = threading.Thread(
            name="ert_ee_api",
            target=self._app.run,
        )
        self._ee_thread = threading.Thread(
            name="ert_ee",
            target=self._evaluate,
        )
        self._ws_thread = threading.Thread(
            name="ert_ee_wsocket",
            target=self._wsocket,
        )

    def _wsocket(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def hello(websocket, path):
            name = await websocket.recv()
            print(f"< {name}")

            greeting = f"Hello {name}!"

            await websocket.send(greeting)
            print(f"> {greeting}")

        start_server = websockets.serve(hello, "localhost", 8765)

        loop.run_until_complete(start_server)
        loop.run_forever()

    def _evaluate(self):
        i = 0
        while self._kill_switch.empty():
            self._put_event(create_event(i, 0, 0, "eclipse"))
            time.sleep(1)
            i += 1
            if i == 15:
                self._put_event(create_event(i, 0, 0, None, "done"))
                break
        self._put_event(create_event(i, 0, 0, None, "terminated"))

    def run(self):
        self._api_thread.start()
        self._ee_thread.start()
        self._ws_thread.start()
        return create_monitor(self._url)

    def _ping(self):
        return "pong"

    def stop(self):
        requests.post("{}/stop".format(self._url))

    def _stop(self):
        if not self._kill_switch.empty():
            return "Shutdown already in progress"

        self._kill_switch.put(None)
        flask.request.environ.get("werkzeug.server.shutdown")()
        self._ee_thread.join()
        return "Server shutting down."

    # def get_snapshot(self):
    #     return squash(self._events)

    def _await_event(self, ident):
        response = {}
        if ident not in self._queues:
            self._queues[ident] = queue.Queue()
            # response["state"] =

        while True:
            try:
                event = self._queues[ident].get()
                return flask.jsonify(event.to_dict())
            except queue.Empty:
                print("queue was empty after some time, sleeping...")
                time.sleep(2)

    def _put_event(self, event):
        self._events.append(event)
        for _, q in self._queues.items():
            q.put(event)
