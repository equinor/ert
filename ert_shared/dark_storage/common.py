from ert_shared.dark_storage.enkf import get_res
from typing import List


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
