import time
import requests


class _Event:
    def __init__(self, event_index, iteration, index, job=None, status=None):
        self._iter = iteration
        self._index = index
        self._job = job
        self._status = status
        self._event_index = event_index

    def is_terminated(self):
        return self._status == "terminated"

    def is_done(self):
        return self._status == "done"

    def is_running(self):
        return not (self.is_terminated() or self.is_done())

    def to_dict(self):
        return {
            "iter": self._iter,
            "index": self._index,
            "status": self._status,
            "event_index": self._event_index,
        }

    @classmethod
    def from_dict(cls, data):
        event_index = data["event_index"]
        iteration = data["iter"]
        index = data["index"]
        job = data.get("job")
        status = data.get("status")
        return cls(event_index, iteration, index, job, status)


def create_event(event_index, iteration, index, job=None, status=None):
    return _Event(event_index, iteration, index, job, status)


def create_event_from_dict(data):
    return _Event.from_dict(data)


class _Monitor:
    def __init__(self, url):
        self._url = url

    def track(self):
        time.sleep(1)  # XXX: fix
        while True:
            # we add owl to fake a unique session
            event_dict = requests.get("{}/{}/owl".format(self._url, "await_event")).json()
            event = _Event.from_dict(event_dict)

            yield event
            if not event.is_running():
                print("monitor ending tracking")
                return


def create_monitor(url):
    return _Monitor(url)
