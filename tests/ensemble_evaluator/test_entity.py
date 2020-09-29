import ert_shared.ensemble_evaluator.entity as ee_entity


def _dict_equal(d1, d2):
    if set(d1.keys()) != set(d2.keys()):
        return False

    for k in d1:
        if type(d1[k]) is dict:
            if not _dict_equal(d1[k], d2[k]):
                return False
        else:
            if d1[k] != d2[k]:
                return False
    return True


_REALIZATION_INDEXES = [0, 1, 3, 4, 5, 9]


def _create_snapshot():
    return ee_entity.create_evaluator_snapshot(
        [
            ee_entity.create_forward_model_job(1, "test1"),
            ee_entity.create_forward_model_job(2, "test2", (1,)),
            ee_entity.create_forward_model_job(3, "test3", (1,)),
            ee_entity.create_forward_model_job(4, "test4", (2, 3)),
        ],
        _REALIZATION_INDEXES,
    )


def test_snapshot_merge():
    snapshot = _create_snapshot()

    snapshot.merge_event(
        ee_entity.create_evaluator_event(
            event_index=0, realizations={1: {"status": "running"}}, status="running"
        )
    )

    assert snapshot._event_index == 0
    assert snapshot._status == "running"

    assert snapshot._realizations[1]["status"] == "running"
    for index in set(_REALIZATION_INDEXES) - set((1,)):
        assert snapshot._realizations[index]["status"] == "unknown"

    snapshot.merge_event(
        ee_entity.create_evaluator_event(
            event_index=1,
            realizations={
                1: {
                    "forward_models": {
                        1: {"status": "done", "data": {"memory": 1000}},
                        2: {"status": "running"},
                    },
                },
                9: {
                    "status": "running",
                    "forward_models": {
                        1: {"status": "running"},
                    },
                },
            },
            status="running",
        )
    )

    assert snapshot._event_index == 1
    assert snapshot._status == "running"

    assert snapshot._realizations[1]["status"] == "running"
    assert _dict_equal(
        snapshot._realizations[1]["forward_models"][1],
        {"status": "done", "data": {"memory": 1000}},
    )
    assert snapshot._realizations[1]["forward_models"][2] == {
        "status": "running",
        "data": None,
    }

    assert snapshot._realizations[9]["status"] == "running"
    assert snapshot._realizations[9]["forward_models"][1] == {
        "status": "running",
        "data": None,
    }

    for index in set(_REALIZATION_INDEXES) - set((1, 9)):
        assert snapshot._realizations[index]["status"] == "unknown"


def test_event_to_dict():
    snapshot = _create_snapshot()
    assert type(snapshot.to_dict()) == dict
    assert _dict_equal(
        snapshot.to_dict(),
        ee_entity.create_evaluator_event_from_dict(snapshot.to_dict()).to_dict(),
    )
