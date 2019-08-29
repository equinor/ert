from job_runner.reporting.message import Finish, Start


class Interactive():

    def report(self, status):
        if isinstance(status, Start):
            print("Running job: {} ... ".format(status.job.name()))
        elif not status.success() and not isinstance(status, Finish):
            print("""failed ....
            -----------------------------------------------------------------
            Error: {}
            -----------------------------------------------------------------
            """.format(status.error_message))
        elif isinstance(status, Finish) and status.success():
            print("OK")
