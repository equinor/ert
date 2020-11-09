from typing import Union


def _h(id_or_name: Union[int, str]) -> str:
    if isinstance(id_or_name, int):
        return str(id_or_name)
    else:
        return f"name/{id_or_name}"


def observations() -> str:
    return "/observations"


def observation(id_or_name: Union[int, str]) -> str:
    return f"/observations/{_h(id_or_name)}"


def observation_attributes(id_or_name: Union[int, str]) -> str:
    return f"/observations/{_h(id_or_name)}/attributes"


def observation_data(id: int) -> str:
    return f"/observation/{id}/data"


def ensembles() -> str:
    return "/ensembles"


def ensemble(id_or_name: Union[int, str]) -> str:
    return f"/ensembles/{_h(id_or_name)}"


def response(ensemble_id, id) -> str:
    return f"/ensembles/{ensemble_id}/responses/{id}"


def response_data(ensemble_id: int, id_or_name: Union[int, str]) -> str:
    return f"/ensembles/{ensemble_id}/responses/{_h(id_or_name)}/data"


def realization(ensemble_id: int, index: int) -> str:
    return f"/ensembles/{ensemble_id}/realizations/{index}"


def parameter(ensemble_id: int, id: Union[int, str]) -> str:
    return f"/ensembles/{ensemble_id}/parameters/{_h(id)}"


def parameter_data(ensemble_id: int, id: int) -> str:
    return f"/ensembles/{ensemble_id}/parameters/{id}/data"
