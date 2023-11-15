from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

from ert.services import StorageService


def get_info(
    project_id: Optional[os.PathLike[str]] = None,
) -> Dict[str, Union[str, Tuple[str, Any]]]:
    client = StorageService.connect(project=project_id)
    return {
        "baseurl": client.fetch_url(),
        "auth": client.fetch_auth(),
    }
