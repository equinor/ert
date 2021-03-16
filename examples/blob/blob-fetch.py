#!/usr/bin/env python3
import json
import os
import sys
import time
import uuid

try:
    from azure.core.exceptions import ResourceExistsError
    from azure.storage.blob import BlobServiceClient
except ImportError:
    sys.exit("Could not import Azure Python SDK.\npip install azure-storage-blob")


start = time.time()

try:
    # The connection string to Azure Storage.
    # For the local developer engine, use:
    # DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;
    CONNECTION_STRING = os.environ["ERT_CONNECTION_STRING"]

    # Container in which to store the blobs. Isn't important, as it seems that
    # there is no limit to how many blobs can exist in a container.
    CONTAINER = os.getenv("ERT_CONTAINER", "fmutest1")

    # A unique token for the ensemble evaluation
    # Can be generated with:
    # import uuid; os.environ["ERT_ENSEMBLE_TOKEN"] = uuid.uuid4()
    ENSEMBLE_TOKEN = sys.argv[2]  # os.environ["ERT_ENSEMBLE_TOKEN"]

    # The realization index for the current forward model.
    # Can be unset.
    REALIZATION_INDEX = os.getenv("ERT_REALIZATION_INDEX")
except KeyError as exc:
    sys.exit(f"Environment variable {exc.args[0]} not set")


if len(sys.argv) != 3:
    sys.exit(f"Usage: {sys.argv[0]} [file to download to] [ENSEMBLE_TOKEN]")
if os.path.exists(sys.argv[1]):
    sys.exit(f"'{sys.argv[1]}' already exists, please remove it manually")

fetch_path = (
    "/mnt/resource/ert-compute/ert-azure-blob-test/"
    + str(uuid.uuid4())
    + "/"
    + sys.argv[1]
)
fetch_name = os.path.basename(fetch_path)

os.makedirs(os.path.dirname(fetch_path))

# Connect to the server and make the container if it doesn't already exist
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
try:
    container_client = blob_service_client.create_container(CONTAINER)
except ResourceExistsError:
    container_client = blob_service_client.get_container_client(CONTAINER)

# Create a client. This blob doesn't exist yet in the server.
blob_client = blob_service_client.get_blob_client(
    container=CONTAINER,
    blob=f"{fetch_name}@{REALIZATION_INDEX}@{ENSEMBLE_TOKEN}",
)

start_download = time.time()

# Download the blob by streaming
print(f"Fetching '{fetch_name}'")
with open(fetch_path, "wb") as f:
    # readinto streams the blob into the Python stream f
    blob_client.download_blob().readinto(f)

download_duration = time.time() - start_download
print("Download successful")


with open(fetch_path) as f:
    s = len(f.read())  # sum(int(d) for d in f.readline().strip())

duration = time.time() - start

with open("output.json", "w") as f:
    json.dump(
        {"sum": s, "execution_time": duration, "download_time": download_duration}, f
    )
