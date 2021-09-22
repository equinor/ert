import os
import requests
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Mapping,
    MutableMapping,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Union,
)
from pprint import pformat
from contextlib import contextmanager
from fastapi import Depends
from sqlalchemy.orm.session import Session
from sqlalchemy.engine.base import Transaction
from starlette.testclient import (
    TestClient as StarletteTestClient,
    ASGI2App,
    ASGI3App,
    Cookies,
    Params,
    DataType,
    TimeOut,
    FileType,
)
from sqlalchemy.orm import sessionmaker
from graphene import Schema as GrapheneSchema
from graphene.test import Client as GrapheneClient

from ert_storage.security import security


if TYPE_CHECKING:
    from promise import Promise
    from rx import Observable
    from graphql.execution import ExecutionResult

    GraphQLResult = Union[ExecutionResult, Observable, Promise[ExecutionResult]]


class ClientError(RuntimeError):
    pass


class _TestClient:
    __test__ = False  # Pytest should ignore this class

    def __init__(
        self,
        app: Union[ASGI2App, ASGI3App],
        session: sessionmaker,
        gql_schema: GrapheneSchema,
        base_url: str = "http://testserver",
        raise_server_exceptions: bool = True,
        root_path: str = "",
    ) -> None:
        self.raise_on_client_error = True
        self.http_client = StarletteTestClient(
            app, base_url, raise_server_exceptions, root_path
        )
        self.gql_client = GrapheneClient(gql_schema)
        self.session = session

    def get(
        self,
        url: str,
        check_status_code: Optional[int] = 200,
        **kwargs: Any,
    ) -> requests.Response:
        resp = self.http_client.get(
            url,
            **kwargs,
        )
        self._check(check_status_code, resp)
        return resp

    def post(
        self,
        url: str,
        check_status_code: Optional[int] = 200,
        **kwargs: Any,
    ) -> requests.Response:
        resp = self.http_client.post(url, **kwargs)
        self._check(check_status_code, resp)
        return resp

    def put(
        self,
        url: str,
        check_status_code: Optional[int] = 200,
        **kwargs: Any,
    ) -> requests.Response:
        resp = self.http_client.put(
            url,
            **kwargs,
        )
        self._check(check_status_code, resp)
        return resp

    def patch(
        self,
        url: str,
        check_status_code: Optional[int] = 200,
        **kwargs: Any,
    ) -> requests.Response:
        resp = self.http_client.patch(
            url,
            **kwargs,
        )
        self._check(check_status_code, resp)
        return resp

    def delete(
        self,
        url: str,
        check_status_code: Optional[int] = 200,
        **kwargs: Any,
    ) -> requests.Response:
        resp = self.http_client.delete(
            url,
            **kwargs,
        )
        self._check(check_status_code, resp)
        return resp

    def gql_execute(
        self,
        request_string: str,
        variable_values: Optional[Mapping[str, Any]] = None,
        check: bool = True,
    ) -> dict:
        doc = self.gql_client.execute(request_string, variable_values=variable_values)
        if self.raise_on_client_error and check and "errors" in doc:
            raise ClientError(f"GraphQL query returned an error:\n{pformat(doc)}")

        return doc

    def _check(
        self, check_status_code: Optional[int], response: requests.Response
    ) -> None:
        if (
            not self.raise_on_client_error
            or check_status_code is None
            or response.status_code == check_status_code
        ):
            return

        try:
            doc = response.json()
        except:
            doc = response.content
        raise ClientError(
            f"Status code was {response.status_code}, expected {check_status_code}:\n{doc}"
        )


@contextmanager
def testclient_factory() -> Generator[_TestClient, None, None]:
    env_key = "ERT_STORAGE_DATABASE_URL"
    env_unset = False
    if env_key not in os.environ:
        os.environ[env_key] = "sqlite:///:memory:"
        env_unset = True
        print("Using in-memory SQLite database for tests")

    if os.getenv("ERT_STORAGE_NO_ROLLBACK", ""):
        print(
            "Environment variable 'ERT_STORAGE_NO_ROLLBACK' is set.\n"
            "Will keep data in database."
        )
        rollback = False
    else:
        rollback = True

    from ert_storage.app import app
    from ert_storage.graphql import schema

    session, transaction, connection = _begin_transaction()
    schema.override_session = session

    yield _TestClient(app, session=session, gql_schema=schema)

    schema.override_session = None
    _end_transaction(transaction, connection, rollback)

    if env_unset:
        del os.environ[env_key]


_TransactionInfo = Tuple[sessionmaker, Transaction, Any]


def _override_get_db(session: sessionmaker) -> None:
    from ert_storage.app import app
    from ert_storage.database import (
        get_db,
        IS_POSTGRES,
    )

    async def override_get_db(
        *, _: None = Depends(security)
    ) -> AsyncGenerator[Session, None]:
        db = session()

        # Make PostgreSQL return float8 columns with highest precision. If we don't
        # do this, we may lose up to 3 of the least significant digits.
        if IS_POSTGRES:
            db.execute("SET extra_float_digits=3")
        try:
            yield db
            db.commit()
            db.close()
        except:
            db.rollback()
            db.close()
            raise

    app.dependency_overrides[get_db] = override_get_db


def _begin_transaction() -> _TransactionInfo:
    from ert_storage.database import (
        engine,
        IS_SQLITE,
        HAS_AZURE_BLOB_STORAGE,
    )
    from ert_storage.database_schema import Base

    if IS_SQLITE:
        Base.metadata.create_all(bind=engine)
    if HAS_AZURE_BLOB_STORAGE:
        import asyncio
        from ert_storage.database import create_container_if_not_exist

        loop = asyncio.get_event_loop()
        loop.run_until_complete(create_container_if_not_exist())

    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(autocommit=False, autoflush=False, bind=connection)

    _override_get_db(session)

    return session, transaction, connection


def _end_transaction(transaction: Transaction, connection: Any, rollback: bool) -> None:
    if rollback:
        transaction.rollback()
    else:
        transaction.commit()
    connection.close()
