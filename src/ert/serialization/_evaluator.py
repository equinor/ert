from typing import Any

import orjson


def evaluator_marshaller(content: Any) -> Any:
    if content is None:
        return None
    try:
        return orjson.dumps(content)
    except TypeError:
        return content


def evaluator_unmarshaller(content: Any) -> Any:
    """
    Due to internals of CloudEvent content is double-encoded, therefore double-decoding
    """
    if content is None:
        return None
    try:
        return orjson.loads(content)
    except TypeError:
        return content
