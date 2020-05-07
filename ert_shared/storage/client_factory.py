from ert_shared.feature_toggling import feature_enabled
from ert_shared.storage.autoclient import AutoClient
from ert_shared.storage.client import StorageClient

@feature_enabled("new-storage")
def create_client(args):

    if not args.storage_api_url:
        return AutoClient(args.storage_api_bind)
    else:
        return StorageClient(args.storage_api_url)
