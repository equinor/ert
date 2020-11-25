import collections.abc
import copy
import re


def recursive_update(d, u):
    for k, v in u.items():
        if k not in d:
            raise ValueError(f"Illegal field {k}")
        if isinstance(v, collections.abc.Mapping):
            d_val = d.get(k, {})
            if not d_val:
                d[k] = copy.deepcopy(v)
            else:
                d[k] = recursive_update(d_val, v)
        else:
            d[k] = v
    return d


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
