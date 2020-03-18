from ert_shared.storage.repository import ErtRepository


def get_response_data(name, ensemble_name, repository=None):
    if repository is None:
        repository = ErtRepository()

    with repository:
        for response in repository.get_response_data(name, ensemble_name):
            yield response
