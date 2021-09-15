from ert_shared.dark_storage.enkf import get_res
from typing import List
import pandas as pd


def ensemble_parameters(ensemble_name: str) -> List[dict]:
    res = get_res()
    return [
        dict(
            name=key,
            values=parameter.values.tolist(),
        )
        for key, parameter in (
            (key, res.gather_gen_kw_data(ensemble_name, key))
            for key in res.all_data_type_keys()
            if res.is_gen_kw_key(key)
        )
    ]


def ensemble_parameter_names(ensemble_name: str) -> List[str]:
    res = get_res()
    return res.gen_kw_keys()


def get_response_names():
    res = get_res()
    response_names = [
        key
        for key in res.all_data_type_keys()
        if res.is_gen_data_key(key) or res.is_summary_key(key)
    ]
    return response_names


def data_for_key(case, key):
    """Returns a pandas DataFrame with the datapoints for a given key for a given case. The row index is
    the realization number, and the columns are an index over the indexes/dates"""

    res = get_res()
    if key.startswith("LOG10_"):
        key = key[6:]

    if res.is_summary_key(key):
        data = res.gather_summary_data(case, key).T
    elif res.is_gen_kw_key(key):
        data = res.gather_gen_kw_data(case, key)
        data.columns = pd.Index([0])
    elif res.is_gen_data_key(key):
        data = res.gather_gen_data_data(case, key).T
    else:
        raise ValueError("no such key {}".format(key))

    try:
        return data.astype(float)
    except ValueError:
        return data
