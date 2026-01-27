from pathlib import Path

import orjson
import pytest
import yaml

from ert.ensemble_evaluator import FullSnapshotEvent
from ert.run_models.event import EverestBatchResultEvent, EverestStatusEvent


def round_floats(obj, decimals=6):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    elif isinstance(obj, dict):
        return {key: round_floats(value, decimals) for key, value in obj.items()}
    return obj


@pytest.mark.slow
@pytest.mark.parametrize(
    "config_file",
    [
        pytest.param(
            "config_advanced.yml",
            marks=pytest.mark.xdist_group("math_func/config_advanced.yml"),
        ),
        pytest.param(
            "config_minimal.yml",
            marks=pytest.mark.xdist_group("math_func/config_minimal.yml"),
        ),
        pytest.param(
            "config_multiobj.yml",
            marks=pytest.mark.xdist_group("math_func/config_multiobj.yml"),
        ),
    ],
)
def test_everest_events(config_file, snapshot, cached_example):
    _, config_file, _, events_list = cached_example(f"math_func/{config_file}")

    config_content = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    config_content["simulator"] = {"queue_system": {"name": "local"}}
    Path(config_file).write_text(
        yaml.dump(config_content, default_flow_style=False), encoding="utf-8"
    )

    full_snapshots = [e for e in events_list if isinstance(e, FullSnapshotEvent)]
    everest_events = [
        e.model_dump()
        for e in events_list
        if isinstance(e, EverestStatusEvent | EverestBatchResultEvent)
    ]
    event_info_json = {
        "num_full_snapshots": len(full_snapshots),
        "everest_events": everest_events,
    }

    snapshot_str = (
        orjson.dumps(
            round_floats(event_info_json),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        .decode("utf-8")
        .strip()
        + "\n"
    )
    snapshot.assert_match(snapshot_str, "snapshot.json")
