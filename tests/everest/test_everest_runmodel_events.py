import orjson
import pytest

from ert.ensemble_evaluator import FullSnapshotEvent


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "config_file",
    [
        "config_advanced.yml",
        "config_minimal.yml",
        "config_multiobj.yml",
    ],
)
def test_everest_events(config_file, snapshot, cached_example):
    _, config_file, _, events_list = cached_example(f"math_func/{config_file}")

    full_snapshots = [e for e in events_list if isinstance(e, FullSnapshotEvent)]

    event_info_json = {
        "num_full_snapshots": len(full_snapshots),
    }

    snapshot_str = (
        orjson.dumps(event_info_json, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        .decode("utf-8")
        .strip()
        + "\n"
    )
    snapshot.assert_match(snapshot_str, "snapshot.json")
