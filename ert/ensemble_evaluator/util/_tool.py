import collections
import re
from typing import Any, Union, Mapping
from pyrsistent import freeze
from pyrsistent.typing import PMap as TPMap


def recursive_update(
    left: TPMap[str, Any],
    right: Union[Mapping[str, Any], TPMap[str, Any]],
    check_key: bool = True,
) -> TPMap[str, Any]:
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


def _match_token(token: str, source: str) -> str:
    f_pattern = _regexp_pattern.format(token=token)
    match = re.search(f_pattern, source)
    return match if match is None else match.group()  # type: ignore


def get_real_id(source: str) -> str:
    return _match_token("real", source)


def get_step_id(source: str) -> str:
    return _match_token("step", source)


def get_job_id(source: str) -> str:
    return _match_token("job", source)
