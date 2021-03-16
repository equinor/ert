#!/usr/bin/env python3
import os
import sys

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError
except ImportError:
    sys.exit("Could not import Azure Python SDK.\npip install azure-storage-blob")

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
    ENSEMBLE_TOKEN = os.environ["ERT_ENSEMBLE_TOKEN"]

    # The realization index for the current forward model.
    # Can be unset.
    REALIZATION_INDEX = os.getenv("ERT_REALIZATION_INDEX")
except KeyError as exc:
    sys.exit(f"Environment variable {exc.args[0]} not set")


if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} [file to upload]")
if not os.path.isfile(sys.argv[1]):
    sys.exit(f"'{sys.argv[1]}' must be a regular file")
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

# Upload the blob by streaming
print(f"Uploading '{publish_name}'")
with open(publish_path, "rb") as f:
    blob_client.upload_blob(f, overwrite=False)
print("Upload successful")

# List all current blobs for good measure
print("Listing all blobs in container:")
for blob in container_client.list_blobs():
    print(f"- {blob.name} ({blob.size} bytes)")
