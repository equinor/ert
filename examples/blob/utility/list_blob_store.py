#!/usr/bin/env python3
import os
import sys

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError
except ImportError:
    sys.exit("Could not import Azure Python SDK.\npip install azure-storage-blob")

try:
    CONNECTION_STRING = os.environ["ERT_CONNECTION_STRING"]
    CONTAINER = os.getenv("ERT_CONTAINER", "fmutest1")
except KeyError as exc:
    sys.exit(f"Environment variable {exc.args[0]} not set")


blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
try:
    container_client = blob_service_client.create_container(CONTAINER)
except ResourceExistsError:
    container_client = blob_service_client.get_container_client(CONTAINER)

print("Listing all blobs in container:")
for blob in container_client.list_blobs():
    print(f"- {blob.name} ({blob.size} bytes)")
