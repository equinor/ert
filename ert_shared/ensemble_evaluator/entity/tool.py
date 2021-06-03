import re
from pyrsistent import freeze
import collections


def recursive_update(left, right, check_key=True):
    for k, v in right.items():
        if check_key and k not in left:
            raise ValueError(f"Illegal field {k}")
        if isinstance(v, collections.abc.Mapping):
            d_val = left.get(k)
            if not d_val:
                left = left.set(k, freeze(v))
            else:
                left = left.set(k, recursive_update(d_val, v, check_key))
        else:
            left = left.set(k, v)
    return left


_regexp_pattern = r"(?<=/{token}/)[^/]+"


def _match_token(token, source):
    f_pattern = _regexp_pattern.format(token=token)
    match = re.search(f_pattern, source)
    return match if match is None else match.group()


def get_real_id(source):
    return _match_token("real", source)


def get_step_id(source):
    return _match_token("step", source)


def get_job_id(source):
    return _match_token("job", source)
