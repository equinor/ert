from job_runner.reporting.message import Finish, Start
from job_runner.reporting.misc import Report


class Interactive(Report):
    def report(self, msg):
        if isinstance(msg, Start):
            print("Running job: {} ... ".format(msg.job.name()))
        elif not msg.success() and not isinstance(msg, Finish):
            print(
                """failed ....
            -----------------------------------------------------------------
            Error: {}
            -----------------------------------------------------------------
            """.format(
                    msg.error_message
                )
            )
        elif isinstance(msg, Finish) and msg.success():
            print("OK")
