import os
from typing import Any
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ert_storage.security import security


ENV_RDBMS = "ERT_STORAGE_DATABASE_URL"
ENV_BLOB = "ERT_STORAGE_AZURE_CONNECTION_STRING"
ENV_BLOB_CONTAINER = "ERT_STORAGE_AZURE_BLOB_CONTAINER"


def get_env_rdbms() -> str:
    if ENV_RDBMS not in os.environ:
        raise EnvironmentError(f"Environment variable '{ENV_RDBMS}' not set")
    return os.environ[ENV_RDBMS]


URI_RDBMS = get_env_rdbms()
IS_SQLITE = URI_RDBMS.startswith("sqlite")
IS_POSTGRES = URI_RDBMS.startswith("postgres")
HAS_AZURE_BLOB_STORAGE = ENV_BLOB in os.environ
BLOB_CONTAINER = os.getenv(ENV_BLOB_CONTAINER, "ert")


if IS_SQLITE:
    engine = create_engine(URI_RDBMS, connect_args={"check_same_thread": False})
else:
    engine = create_engine(URI_RDBMS, pool_size=50, max_overflow=100)
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


async def get_db(*, _: None = Depends(security)) -> Any:
    db = Session()

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


if HAS_AZURE_BLOB_STORAGE:
    import asyncio
    from azure.core.exceptions import ResourceNotFoundError
    from azure.storage.blob.aio import ContainerClient

    azure_blob_container = ContainerClient.from_connection_string(
        os.environ[ENV_BLOB], BLOB_CONTAINER
    )

    async def create_container_if_not_exist() -> None:
        try:
            await azure_blob_container.get_container_properties()
        except ResourceNotFoundError:
            await azure_blob_container.create_container()
