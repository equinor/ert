import httpx

from ._session import ConnInfo, find_conn_info


class AsyncClient(httpx.AsyncClient):
    """
    Wrapper class for httpx.AsyncClient that provides a user-friendly way to
    interact with ERT Storage's API
    """

    def __init__(self, conn_info: ConnInfo | None = None) -> None:
        if conn_info is None:
            conn_info = find_conn_info()

        headers = {}
        if conn_info.auth_token is not None:
            headers = {"Token": conn_info.auth_token}

        super().__init__(base_url=conn_info.base_url, headers=headers)
