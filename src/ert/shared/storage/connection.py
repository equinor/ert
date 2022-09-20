import os
from typing import Optional

from ert.services import Storage


def get_info(project_id: Optional[os.PathLike] = None):
    client = Storage.connect(project=project_id)
    return {
        "baseurl": client.fetch_url(),
        "auth": client.fetch_auth(),
    }
