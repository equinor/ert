#!/usr/bin/env python
import os
import io
import sys
import json
import requests
import numpy
import pandas as pd
import hashlib
import time
import asyncio
import aiohttp
import concurrent.futures

# "s034-0113.s034.oc.equinor.com:8000"
try:
    ERT_STORAGE_URL = os.environ["ERT_STORAGE_URL"]
except KeyError:
    ERT_STORAGE_URL = "http://127.0.0.1:8000"

FILE_RECORD_NAMES = {"published_file"}
MATRIX_RECORD_NAMES = {"published_matrix"}


def generate_matrix():
    return numpy.random.rand(1000, 1).tolist()


size = 1024 ** 3
block_size = 4 * 1024 ** 2


def generate_blob_chunks(chunk_size):
    data = []
    with open("/dev/urandom", "rb") as file_handle:
        for i in range(size // block_size):
            data.append(file_handle.read(chunk_size))
    return data


def upload_block(index, chunk, ensemble_id, realization_index, record):
    requests.put(
        f"{ERT_STORAGE_URL}/ensembles/{ensemble_id}/records/{record}/blob",
        params={"realization_index": realization_index, "block_index": index},
        files={"file": ("somefile", io.BytesIO(chunk), "some/type")},
    )


async def post_blocks(executor, ensemble_id, realization_index, record, chunks):
    loop = asyncio.get_event_loop()
    blocking_tasks = [
        loop.run_in_executor(
            executor,
            upload_block,
            index,
            chunk,
            ensemble_id,
            realization_index,
            record,
        )
        for index, chunk in enumerate(chunks)
    ]

    completed, pending = await asyncio.wait(blocking_tasks)


def post_large_blob(ensemble_id, realization_index, record, chunks):
    requests.post(
        f"{ERT_STORAGE_URL}/ensembles/{ensemble_id}/records/{record}/blob",
        params={"realization_index": realization_index},
    )
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=8,
    )
    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(
            post_blocks(executor, ensemble_id, realization_index, record, chunks)
        )
    finally:
        event_loop.close()
    requests.patch(
        f"{ERT_STORAGE_URL}/ensembles/{ensemble_id}/records/{record}/blob",
        params={"realization_index": realization_index},
    )


def main():
    start = time.time()
    with open("realization.json") as f:
        realization = json.loads(f.read())
    realization_index = int(realization["iens"])
    ensemble_id = sys.argv[1]

    for record in MATRIX_RECORD_NAMES:
        start_matrix = time.time()
        data = generate_matrix()
        resp = requests.post(
            f"{ERT_STORAGE_URL}/ensembles/{ensemble_id}/records/{record}/matrix?realization_index={realization_index}",
            data=json.dumps(data),
        )
        duration_matrix = time.time() - start_matrix

        assert resp.status_code == 200

        md5 = hashlib.md5()
        md5.update(bytes(str(data).replace(" ", ""), "utf-8"))
        matrix_hash = md5.hexdigest()  # str
        resp = requests.post(
            f"{ERT_STORAGE_URL}/ensembles/{ensemble_id}/records/{record}_checksum/file?realization_index={realization_index}",
            files={
                "file": (
                    "somefile",
                    io.StringIO(matrix_hash),
                    "application/octat-stream",
                )
            },
        )
        assert resp.status_code == 200

    chunks = generate_blob_chunks(block_size)
    for record in FILE_RECORD_NAMES:
        start_upload = time.time()
        post_large_blob(ensemble_id, realization_index, record, chunks)
        upload_duration = time.time() - start_upload
        assert resp.status_code == 200

        md5 = hashlib.md5()
        data = b"".join(chunks)
        md5.update(data)
        blob_hash = md5.hexdigest()  # str
        resp = requests.post(
            f"{ERT_STORAGE_URL}/ensembles/{ensemble_id}/records/{record}_checksum/file?realization_index={realization_index}",
            files={
                "file": ("somefile", io.StringIO(blob_hash), "application/octat-stream")
            },
        )
        assert resp.status_code == 200

    duration = time.time() - start
    with open("output.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "sum": 1,
                    "execution_time": duration,
                    "upload_time": upload_duration,
                    "matrix_upload_time": duration_matrix,
                },
            )
        )


if __name__ == "__main__":
    main()
