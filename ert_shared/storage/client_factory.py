from ert_shared.feature_toggling import FeatureToggling
from ert_shared.storage.server_client import ServerClient
from ert_shared.storage.client import StorageClient


def create_client(args):
    if not FeatureToggling.is_enabled("new-storage"):
        return None

    if not args.storage_api_url:
        return ServerClient(args.storage_api_bind)
    else:
        return StorageClient(args.storage_api_url)
