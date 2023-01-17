import os
from typing import Optional

from ert.services import StorageService


def get_info(project_id: Optional[os.PathLike] = None):
    client = StorageService.connect(project=project_id)
    return {
        "baseurl": client.fetch_url(),
        "auth": client.fetch_auth(),
    }
