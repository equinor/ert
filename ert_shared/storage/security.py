import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyCookie, APIKeyHeader, APIKeyQuery


_security_cookie = APIKeyCookie(name="ERT-Storage-Token", auto_error=False)
_security_header = APIKeyHeader(name="Token", auto_error=False)
_security_query = APIKeyQuery(name="token", auto_error=False)


async def security_token(
    *,
    cookie_token: str = Security(_security_cookie),
    header_token: str = Security(_security_header),
    query_token: str = Security(_security_query),
) -> None:
    if not any((cookie_token, header_token, query_token)):
        # HTTP 401 is when the user didn't attempt to authorize themselves
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )

    real_token = os.getenv("ERT_STORAGE_TOKEN")
    for token in cookie_token, header_token, query_token:
        if token == real_token:
            return
    # HTTP 403 is when the user has authorized themselves, but aren't allowed to
    # access this resource
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")


if "ERT_STORAGE_TOKEN" not in os.environ:

    async def security_token():
        """
        This is a dummy function that doesn't depend on the APIKey*. This is done so
        that FastAPI doesn't produce an OpenAPI document that incorrectly states
        that there is security enabled.
        """
