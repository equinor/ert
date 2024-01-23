from fastapi import APIRouter

router = APIRouter(tags=["ensemble"])


@router.post("/updates/facade")
def refresh_facade() -> None:
    pass
