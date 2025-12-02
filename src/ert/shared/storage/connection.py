from __future__ import annotations

import os
from typing import Any

from ert.services import ErtServer


def get_info(
    project_id: os.PathLike[str],
) -> dict[str, str | tuple[str, Any]]:
    client = ErtServer.connect(project=project_id)
    return {
        "baseurl": client.fetch_url(),
        "auth": client.fetch_auth(),
        "cert": client.fetch_connection_info()["cert"],
    }
