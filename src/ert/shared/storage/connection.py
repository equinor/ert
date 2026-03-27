from __future__ import annotations

import os
from typing import Any

from ert.services.ert_server import create_ert_server_controller


def get_info(
    project_id: os.PathLike[str],
) -> dict[str, str | tuple[str, Any]]:
    controller = create_ert_server_controller(project=project_id)
    return {
        "baseurl": controller.fetch_url(),
        "auth": controller.fetch_auth(),
        "cert": controller.fetch_connection_info()["cert"],
    }
