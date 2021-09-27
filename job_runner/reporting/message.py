from datetime import datetime as dt

_JOB_STATUS_SUCCESS = "Success"
_JOB_STATUS_RUNNING = "Running"
_JOB_STATUS_FAILURE = "Failure"
_JOB_STATUS_WAITING = "Waiting"

_RUNNER_STATUS_INITIALIZED = "Initialized"
_RUNNER_STATUS_SUCCESS = "Success"
_RUNNER_STATUS_FAILURE = "Failure"


_JOB_EXIT_FAILED_STRING = """Job {job_name} FAILED with code {exit_code}
----------------------------------------------------------
Error message: {error_message}
----------------------------------------------------------
"""


class _MetaMessage(type):
    def __repr__(cls):
        return f"MessageType<{cls.__name__}>"


class Message(metaclass=_MetaMessage):
    def __init__(self, job=None):
        self.timestamp = dt.now()
        self.job = job
        self.error_message = None

    def __repr__(self):
        return type(self).__name__

    def with_error(self, message):
        self.error_message = message
        return self

    def success(self):
        return self.error_message is None


# manager level messages


class Init(Message):
    def __init__(self, jobs, run_id, ert_pid, ee_id=None, real_id=None, step_id=None):
        super(Init, self).__init__()
        self.jobs = jobs
        self.run_id = run_id
        self.ert_pid = ert_pid
        self.ee_id = ee_id
        self.real_id = real_id
        self.step_id = step_id


class Finish(Message):
    def __init__(self):
        super(Finish, self).__init__()


# job level messages


class Start(Message):
    def __init__(self, job):
        super(Start, self).__init__(job)


class Running(Message):
    def __init__(self, job, max_memory_usage, current_memory_usage):
        super(Running, self).__init__(job)
        self.max_memory_usage = max_memory_usage
        self.current_memory_usage = current_memory_usage


class Exited(Message):
    def __init__(self, job, exit_code):
        super(Exited, self).__init__(job)
        self.exit_code = exit_code
