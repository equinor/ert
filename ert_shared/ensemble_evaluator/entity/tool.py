import collections.abc
import copy


def recursive_update(d, u):
    for k, v in u.items():
        if k not in d:
            raise ValueError(f"Illegal field {k}")
        if isinstance(v, collections.abc.Mapping):
            d_val = d.get(k, {})
            if d_val is None:
                d[k] = copy.deepcopy(v)
            else:
                d[k] = recursive_update(d_val, v)
        else:
            d[k] = v
    return d
