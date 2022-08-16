from typing import Mapping, Any
from fastapi import APIRouter, Depends
from ert_storage.database import Session, get_db

router = APIRouter(tags=["info"])


@router.get("/server/info", response_model=Mapping[str, Any])
def info(
    *,
    db: Session = Depends(get_db),
) -> Mapping[str, Any]:
    return {"name": "Ert Storage Server"}
