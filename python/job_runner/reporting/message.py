from datetime import datetime as dt


class Message(object):
    def __init__(self, job=None):
        self.timestamp = dt.now()
        self.job = job
        self.error_message = None

    def with_error(self, message):
        self.error_message = message
        return self

    def success(self):
        return self.error_message is None


# manager level messages


class Init(Message):
    def __init__(self, jobs, run_id, ert_pid):
        super(Init, self).__init__()
        self.jobs = jobs
        self.run_id = run_id
        self.ert_pid = ert_pid


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
