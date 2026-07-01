from unittest.mock import MagicMock

from ert.logging import ObjectAttributeFilter


def test_that_sensitive_attribute_is_only_redacted_from_sensitive_objects():
    record = MagicMock(msg="Obj1(val=20) Obj2(val=30)")
    ObjectAttributeFilter(["Obj1"], ["val"]).filter(record)
    assert record.msg == "Obj1(val=REDACTED) Obj2(val=30)"


def test_that_object_sharing_suffix_with_sensitive_object_is_not_matched():
    record = MagicMock(msg="Obj(val=20) NotAObj(val=30)")
    ObjectAttributeFilter(["Obj"], ["val"]).filter(record)
    assert record.msg == "Obj(val=REDACTED) NotAObj(val=30)"


def test_that_inner_objects_of_sensitive_objects_also_gets_attributes_redacted():
    record = MagicMock(msg="Obj(val=20, inner=InnerObj(val=30))")
    ObjectAttributeFilter(["Obj"], ["val"]).filter(record)
    assert record.msg == "Obj(val=REDACTED, inner=InnerObj(val=REDACTED))"


def test_that_redaction_from_no_matches_leaves_string_unchanged():
    record = MagicMock(msg="Obj(val=20, inner=InnerObj(val=30))")
    ObjectAttributeFilter(["Foo"], ["val"]).filter(record)
    assert record.msg == "Obj(val=20, inner=InnerObj(val=30))"


def test_that_noise_between_object_and_attributes_results_in_unfiltered_record():
    original_record = "Obj noise (val=20, inner=InnerObj(val=30))"
    record = MagicMock(msg=original_record)
    ObjectAttributeFilter(["Obj"], ["val"]).filter(record)
    assert record.msg == original_record


def test_that_space_between_object_and_attributes_results_in_unfiltered_record():
    original_record = "Obj (val=20, inner=InnerObj(val=30))"
    record = MagicMock(msg=original_record)
    ObjectAttributeFilter(["Obj"], ["val"]).filter(record)
    assert record.msg == original_record


def test_that_sensitive_object_inside_not_sensitive_object_gets_values_redacted():
    original_record = "NotSensitive(val=20, inner=Sensitive(val=30))"
    record = MagicMock(msg=original_record)
    ObjectAttributeFilter(["Sensitive"], ["val"]).filter(record)
    assert record.msg == "NotSensitive(val=20, inner=Sensitive(val=REDACTED))"


def test_that_sensitive_observation_info_is_redacted_from_a_large_log_example(caplog):
    sensitive_objects = ["SensitiveObservation", "ShapeRegistry"]
    sensitive_attributes = ["value", "date", "north", "east"]

    sensitive_attributes_ = {
        "value": "42.42",
        "date": "'2012-10-10T00:00:00'",
        "east": "99",
        "north": "71",
    }
    insensitive_attributes = {
        "value": "24.24",
        "date": "'2013-01-10T00:00:00'",
        "radius": "100",
    }

    sensitive_obs = (
        "SensitiveObservation(type='summary_observation', name='WOPR_OP1_102', "
        f"value={sensitive_attributes_['value']}, error=4.0, key='WOPR:OP1', "
        f"date={sensitive_attributes_['date']}, shape_id=0, error_mode=None, "
        f"error_min=None)"
    )
    insensitive_obs = (
        "CasualObservation(type='summary_observation', name='WOPR_OP2_217', "
        f"value={insensitive_attributes['value']}, error=4.0, key='WOPR:OP2', "
        f"date={insensitive_attributes['date']}, shape_id=None, error_mode=None, "
        f"error_min=None)"
    )
    shape_registry = (
        "ShapeRegistry(shapes={0: CircleShapeConfig(shape_id=0, "
        f"type='circle', "
        f"east={sensitive_attributes_['east']}, "
        f"north={sensitive_attributes_['north']}, "
        f"radius={insensitive_attributes['radius']}}})"
    )

    noise = " Hello, I am a log. I have nothing to do with sensitive information. "
    record = MagicMock(
        msg=(
            noise
            + sensitive_obs
            + noise
            + insensitive_obs
            + noise
            + shape_registry
            + noise
        )
    )

    ObjectAttributeFilter(sensitive_objects, sensitive_attributes).filter(record)
    for sensitive_value in sensitive_attributes_.values():
        assert sensitive_value not in record.msg

    for insensitive_value in insensitive_attributes.values():
        assert insensitive_value in record.msg

    # Assert that noise is unchanged
    assert record.msg.count(noise) == 4
