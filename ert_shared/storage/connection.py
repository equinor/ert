from ert_shared.services import Storage


def get_info(project_id):
    client = Storage.connect(project=project_id)
    return {
        "baseurl": client.fetch_url(),
        "auth": client.fetch_auth(),
    }
