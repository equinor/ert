import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

DEFAULT_TOKEN = "hunter2"
_security_header = APIKeyHeader(name="Token", auto_error=False)


async def security(*, token: str | None = Security(_security_header)) -> None:
    if os.getenv("ERT_STORAGE_NO_TOKEN"):
        return
    if not token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authenticated"
        )
    real_token = os.getenv("ERT_STORAGE_TOKEN", DEFAULT_TOKEN)
    if token == real_token:
        # Success
        return

    # HTTP 403 is when the user has authorized themselves, but aren't allowed to
    # access this resource
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")
