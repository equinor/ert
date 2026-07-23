from ert.config.forward_model_step import ForwardModelStepJSON


def create_stub_forward_model_step_json(
    name: str = "fmstep1",
    executable: str = "echo",
    argList: list[str] | None = None,
    stderr: str | None = "stderr",
    stdout: str | None = "stdout",
    max_running_minutes: int | None = None,
    target_file: str | None = None,
) -> ForwardModelStepJSON:
    if argList is None:
        argList = []
    return {
        "executable": executable,
        "name": name,
        "stdout": stdout,
        "stderr": stderr,
        "argList": argList,
        "environment": {},
        "max_running_minutes": max_running_minutes,
        "stdin": None,
        "target_file": target_file,
        "error_file": None,
        "start_file": None,
    }
