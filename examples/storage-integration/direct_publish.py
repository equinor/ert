#!/usr/bin/env python

import uuid
import os
from azure.storage.blob import ContainerClient
from azure.core.exceptions import ResourceNotFoundError
import asyncio
import time
import json

ENV_BLOB = "ERT_STORAGE_AZURE_CONNECTION_STRING"
BLOB_CONTAINER = "ert"

azure_blob_container = ContainerClient.from_connection_string(
    os.environ[ENV_BLOB], BLOB_CONTAINER
)
key = f"{uuid.uuid4()}"
blob = azure_blob_container.get_blob_client(key)
block_size = 32 * (1024 ** 2)
size = 1024 ** 3


def generate_blob():
    with open("/dev/urandom", "rb") as file_handle:
        data = file_handle.read(size)
        with open("file.blob", "wb") as out:
            out.write(data)


def create_container_if_not_exist() -> None:
    try:
        azure_blob_container.get_container_properties()
    except ResourceNotFoundError:
        azure_blob_container.create_container()


def read_in_chunks(file_handle, chunk_size):
    while True:
        data = file_handle.read(chunk_size)
        if not data:
            break
        yield data


def upload_block(index, chunk):
    blob.stage_block(uuid.uuid4(), chunk)


def main():
    create_container_if_not_exist()
    with open("file.blob", "rb") as file_handle:
        blob.upload_blob(file_handle, max_concurrency=16)


if __name__ == "__main__":

    generate_blob()
    start = time.perf_counter()
    main()
    finish = time.perf_counter() - start

    with open("output.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "sum": 1,
                    "upload_time": finish,
                },
            )
        )
    print(f"finished in {finish} seconds")
