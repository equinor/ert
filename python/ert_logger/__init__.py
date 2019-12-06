import os, pwd, json, requests
from copy import deepcopy

LOG_URL = "http://devnull.statoil.no:4444"
USER = pwd.getpwuid(os.getuid()).pw_name
APPLICATION_NAME = "ERT"
BASE_MESSAGE = {
    "user": USER,
    "application": APPLICATION_NAME,
    "komodo_release": os.getenv("KOMODO_RELEASE", "--------"),
}


def log_message(input_payload):
    payload = deepcopy(BASE_MESSAGE)
    payload.update(input_payload)
    try:
        data = json.dumps(payload)
        # Disabling proxies
        proxies = {"http": None, "https": None}
        requests.post(
            LOG_URL,
            timeout=3,
            headers={"Content-Type": "application/json"},
            data=data,
            proxies=proxies,
        )
    except:  # noqa
        pass
