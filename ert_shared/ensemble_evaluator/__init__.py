import time
import queue
import flask
import threading
import requests

_app = flask.Flask(__name__)


class _Monitor:
    def __init__(self, url):
        self._url = url

    def track(self):
        time.sleep(1)
        while True:
            # we add owl to fake a unique session
            resp = requests.get("{}/{}/owl".format(self._url, "await_event")).json()
            if resp.get("status", "not_done") == "done":
                return
            yield resp


class EnsembleEvaluator:
    def __init__(self):
        self._url = "http://localhost:5000"
        self._kill_switch = queue.Queue()
        self._queues = {}
        self._events = []

        self._app = flask.Flask("ert ee")
        self._app.add_url_rule("/ping", "ping", self._ping)
        self._app.add_url_rule("/await_event/<ident>", "await_event", self._await_event)
        self._app.add_url_rule("/stop", "stop", self._stop, methods=['POST'])
        self._api_thread = threading.Thread(
            name="ert_ee_api",
            target=self._app.run,
        )
        self._ee_thread = threading.Thread(
            name="ert_ee",
            target=self._evaluate,
        )

    def _evaluate(self):
        i = 0
        while self._kill_switch.empty():
            self._put_event({"iter": 0, "index": 0, "job": "eclipse", "event_index": i})
            time.sleep(1)
            i += 1
            if i == 30:
                break
        self._put_event({"iter": 0, "index": 0, "status": "done", "event_index": i})

    def run(self):
        self._api_thread.start()
        self._ee_thread.start()
        return _Monitor(self._url)

    def _ping(self):
        return "pong"

    def stop(self):
        if self._kill_switch.empty():
            self._kill_switch.put(None)
            requests.post("{}/stop".format(self._url))

    def _stop(self):
        self._kill_switch.put(None)
        flask.request.environ.get("werkzeug.server.shutdown")()
        # self._api_thread.join()
        self._ee_thread.join()
        return "Server shutting down."

    # def get_snapshot(self):
    #     return squash(self._events)

    def _await_event(self, ident):
        if ident not in self._queues:
            self._queues[ident] = queue.Queue()

        while True:
            try:
                return flask.jsonify(self._queues[ident].get())
            except queue.Empty:
                print("queue was empty after some time, sleeping...")
                time.sleep(2)

    def _put_event(self, event):
        self._events.append(event)
        for _, q in self._queues.items():
            q.put(event)
