import ert


@ert.plugin(name="ert")  # type: ignore
def site_config_lines():
    return [
        "JOB_SCRIPT fm_dispatch.py",
        "QUEUE_SYSTEM LOCAL",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
    ]
