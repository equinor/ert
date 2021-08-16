import os, pwd
from typing import Any

LOG_URL = "http://devnull.statoil.no:4444"
USER = pwd.getpwuid(os.getuid()).pw_name
APPLICATION_NAME = "ERT"
BASE_MESSAGE = {
    "user": USER,
    "application": APPLICATION_NAME,
    "komodo_release": os.getenv("KOMODO_RELEASE", "--------"),
}


# Disabled temporarily, according to issue #1095.
#
# The tests in tests/job_runner/test_network_reporter.py are disabled
# accordingly.
def log_message(input_payload: Any) -> None:
    pass
