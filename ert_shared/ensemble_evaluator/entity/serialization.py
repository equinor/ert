import datetime
import json
from typing import Any

from dateutil import parser


class EvaluatorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return {"__type__": "isoformat8601", "value": obj.isoformat()}
        return json.JSONEncoder.default(self, obj)


def evaluator_marshaller(content: Any):
    if content is None:
        return None
    try:
        return json.dumps(content, cls=EvaluatorEncoder)
    except TypeError:
        return content


def object_hook(obj):
    if "__type__" not in obj:
        return obj

    if obj["__type__"] == "isoformat8601":
        return parser.parse(obj["value"])

    return obj


def evaluator_unmarshaller(content: Any):
    """
    Due to internals of CloudEvent content is double-encoded, therefore double-decoding
    """
    if content is None:
        return None
    try:
        content = json.loads(content)
        return json.loads(content, object_hook=object_hook)
    except TypeError:
        return content
