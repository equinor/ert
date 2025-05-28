import sys
from typing import Any

from ert.services._base_service import BaseService


class WebvizErt(BaseService):
    service_name = "webviz-ert"

    def __init__(self, **kwargs: Any) -> None:
        exec_args = [sys.executable, "-m", "webviz_ert"]
        if kwargs.get("experimental_mode"):
            exec_args.append("--experimental-mode")
        if kwargs.get("verbose"):
            exec_args.append("--verbose")
        exec_args.extend(["--title", str(kwargs.get("title"))])
        project = kwargs.get("project")
        exec_args.extend(["--project_identifier", str(project)])

        super().__init__(exec_args, project=project)
