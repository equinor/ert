from __future__ import annotations

import os
from typing import Any

from ert.services import StorageService


def get_info(
    project_id: os.PathLike[str],
) -> dict[str, str | tuple[str, Any]]:
    client = StorageService.connect(project=project_id)
    return {
        "baseurl": client.fetch_url(),
        "auth": client.fetch_auth(),
        "cert": client.fetch_conn_info()["cert"],
    }
