import json
import logging
from uuid import UUID

from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import get_storage, reraise_as_http_errors
from ert.storage import Storage

router = APIRouter(tags=["update"])
logger = logging.getLogger(__name__)


DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get("/ensembles/{ensemble_id}/update/artifacts")
async def get_update_artifacts_for_ensemble(
    *, storage: Storage = DEFAULT_STORAGE, ensemble_id: UUID
) -> list[js.ArtifactOut]:
    with reraise_as_http_errors(logger, {404: "Ensemble not found"}):
        ensemble = storage.get_ensemble(ensemble_id)

    artifacts: list[js.ArtifactOut] = []
    for path in ensemble.list_blob_data():
        if not path.name.endswith(".json"):
            continue
        with path.open() as f:
            data = json.load(f)
        if blob_type := data.get("blob_type", ""):
            artifacts.append(
                js.ArtifactOut(
                    blob_type=blob_type,
                    name=path.stem,
                    uri=data["uri"],
                    file_size=data.get("file_size", 0),
                    sparse=data.get("sparse", False),
                    shape=tuple(data.get("shape", (0, 0))),
                )
            )
    return artifacts
