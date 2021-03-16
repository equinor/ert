#!/usr/bin/env python3
import os
import sys
import uuid
import json
import time

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError
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
    REALIZATION_INDEX = str(uuid.uuid4())
except KeyError as exc:
    sys.exit(f"Environment variable {exc.args[0]} not set")


if len(sys.argv) != 3:
    sys.exit(f"Usage: {sys.argv[0]} [file to upload]")

publish_path = sys.argv[1]
publish_name = os.path.basename(publish_path)

# Connect to the server and make the container if it doesn't already exist
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
try:
    container_client = blob_service_client.create_container(CONTAINER)
except ResourceExistsError:
    container_client = blob_service_client.get_container_client(CONTAINER)

# Create a client. This blob doesn't exist yet in the server.
blob_client = blob_service_client.get_blob_client(
    container=CONTAINER,
    blob=f"{publish_name}@{REALIZATION_INDEX}@{ENSEMBLE_TOKEN}",
)


with open(publish_path, "w") as f:
    txt = int(10 ** 9 / len(str(REALIZATION_INDEX))) * str(REALIZATION_INDEX)
    f.write(txt)
    s = len(txt)

start_upload = time.time()

# Upload the blob by streaming
print(f"Uploading '{publish_name}'")
with open(publish_path, "rb") as f:
    blob_client.upload_blob(f, overwrite=False)
print("Upload successful")
upload_duration = time.time() - start_upload


duration = time.time() - start

with open("output.json", "w") as f:
    json.dump({"sum": s, "execution_time": duration, "upload_time": upload_duration}, f)
