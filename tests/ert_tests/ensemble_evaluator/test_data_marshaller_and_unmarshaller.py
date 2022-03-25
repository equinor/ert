import datetime

from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent

from ert import serialization


def test_data_marshaller_and_unmarshaller():
    data = {"start_time": datetime.datetime.now()}
    out_cloudevent = CloudEvent(
        {
            "type": "com.equinor.ert.ee.snapshot",
            "source": f"/ert/ee/{0}",
            "id": 0,
        },
        data,
    )

    ce_to_json = to_json(
        out_cloudevent, data_marshaller=serialization.evaluator_marshaller
    )
    ce_from_json = from_json(
        ce_to_json, data_unmarshaller=serialization.evaluator_unmarshaller
    )

    assert isinstance(ce_from_json.data["start_time"], datetime.datetime)
    assert out_cloudevent == ce_from_json
