import re
from pyrsistent import PMap


def recursive_update(left, right):
    for k, v in right.items():
        if k not in left:
            raise ValueError(f"Illegal field {k}")
        if isinstance(v, PMap):
            d_val = left.get(k)
            if not d_val:
                left = left.set(k, v)
            else:
                left = left.set(k, recursive_update(d_val, v))
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


def get_stage_id(source):
    return _match_token("stage", source)


def get_step_id(source):
    return _match_token("step", source)


def get_job_id(source):
    return _match_token("job", source)
