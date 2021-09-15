import os
from typing import Any
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from ert_storage.security import security


class DatabaseConfig:
    def __init__(self) -> None:
        self.ENV_RDBMS = "ERT_STORAGE_DATABASE_URL"
        self.ENV_BLOB = "ERT_STORAGE_AZURE_CONNECTION_STRING"
        self.ENV_BLOB_CONTAINER = "ERT_STORAGE_AZURE_BLOB_CONTAINER"
        self._session = None
        self._engine = None

    def get_env_rdbms(self) -> str:
        if not self.ENV_RDBMS_AVAILABLE:
            raise EnvironmentError(f"Environment variable '{self.ENV_RDBMS}' not set")
        return os.environ[self.ENV_RDBMS]

    @property
    def ENV_RDBMS_AVAILABLE(self) -> bool:
        return self.ENV_RDBMS in os.environ

    @property
    def URI_RDBMS(self) -> str:
        return self.get_env_rdbms()

    @property
    def IS_SQLITE(self) -> bool:
        return self.URI_RDBMS.startswith("sqlite")

    @property
    def IS_POSTGRES(self) -> bool:
        return self.URI_RDBMS.startswith("postgres")

    @property
    def HAS_AZURE_BLOB_STORAGE(self) -> bool:
        return self.ENV_BLOB in os.environ

    @property
    def BLOB_CONTAINER(self) -> str:
        return os.getenv(self.ENV_BLOB_CONTAINER, "ert")

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            if self.IS_SQLITE:
                self._engine = create_engine(
                    self.URI_RDBMS, connect_args={"check_same_thread": False}
                )
            else:
                self._engine = create_engine(
                    self.URI_RDBMS, pool_size=50, max_overflow=100
                )
        return self._engine

    @property
    def Session(self) -> sessionmaker:
        if self._session is None:
            self._session = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
        return self._session


database_config = DatabaseConfig()
Base = declarative_base()


async def get_db(*, _: None = Depends(security)) -> Any:
    db = database_config.Session()

    # Make PostgreSQL return float8 columns with highest precision. If we don't
    # do this, we may lose up to 3 of the least significant digits.
    if database_config.IS_POSTGRES:
        db.execute("SET extra_float_digits=3")
    try:
        yield db
        db.commit()
        db.close()
    except:
        db.rollback()
        db.close()
        raise


if database_config.HAS_AZURE_BLOB_STORAGE:
    import asyncio
    from azure.core.exceptions import ResourceNotFoundError
    from azure.storage.blob.aio import ContainerClient

    azure_blob_container = ContainerClient.from_connection_string(
        os.environ[database_config.ENV_BLOB], database_config.BLOB_CONTAINER
    )

    async def create_container_if_not_exist() -> None:
        try:
            await azure_blob_container.get_container_properties()
        except ResourceNotFoundError:
            await azure_blob_container.create_container()
