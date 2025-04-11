from queue import SimpleQueue

import numpy as np
import polars as pl
import pytest
from orjson import orjson

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models import StatusEvents
from ert.run_models.event import EverestCacheHitEvent
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


@pytest.mark.integration_test
def test_simulator_cache(copy_math_func_test_data_to_tmp, snapshot):
    n_invocations = 0

    def new_call(*args):
        nonlocal n_invocations
        result = original_call(*args)
        n_invocations += 1
        return result

    config = EverestConfig.load_file("config_minimal.yml")
    config_dict = config.model_dump(exclude_none=True)
    config_dict["optimization"]["max_batch_num"] = 4
    config = EverestConfig.model_validate(config_dict)

    status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
    run_model = EverestRunModel.create(config, status_queue=status_queue)
    evaluator_server_config = EvaluatorServerConfig()

    # Modify the forward model function to track number of calls:
    original_call = run_model._evaluate_and_postprocess
    run_model._evaluate_and_postprocess = new_call

    # First run populates the cache:
    run_model.run_experiment(evaluator_server_config)
    assert n_invocations > 0
    variables1 = list(run_model.result.controls.values())
    assert np.allclose(variables1, [0.5, 0.5, 0.5], atol=0.02)

    # Now do another run, where the functions should come from the cache:
    n_invocations = 0

    # The batch_id was used as a stopping criterion, so it must be reset:
    # Note: We expect batch N to take cached results from batch N-1
    # as should be reflected by the snapshot
    run_model._batch_id = 1

    run_model.run_experiment(evaluator_server_config)
    assert n_invocations == 0
    variables2 = list(run_model.result.controls.values())
    assert np.array_equal(variables1, variables2)

    events_list = []
    while not status_queue.empty():
        event = status_queue.get()
        events_list.append(event)

    cache_hit_events = [e for e in events_list if isinstance(e, EverestCacheHitEvent)]
    cache_hit_event_dicts = [e.model_dump() for e in cache_hit_events]

    snapshot.assert_match(
        orjson.dumps(
            cache_hit_event_dicts,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2,
        )
        .decode("utf-8")
        .strip()
        + "\n",
        "cache_hit_events.json",
    )


def test_cached_result_lookup():
    control_values_to_evaluate = np.array(
        [
            [0.5, 0.5, 0.5],  # 0
            [0.4, 0.4, 0.4],  # 2
            [0.3, 0.3, 0.3],  # 3
            [0.2, 0.2, 0.2],  # 4
            [0.1, 0.1, 0.1],  # 5
        ]
    )

    model_realizations_to_evaluate = [0, 2, 3, 4, 5]

    # Shuffling some to make some hits be function evaluations, and some perturbations
    all_results = pl.DataFrame(
        {
            "batch": [0, 0, 0, 0, 0],
            "model_realization": [5, 0, 2, 4, 3],
            "perturbation": [-1, 0, 1, 2, 3],
            "realization": [0, 1, 2, 3, 4],
            "distance": np.random.default_rng(666).normal(loc=0.5, size=5, scale=0.1),
            "point.x": ([0.1, 0.5, 0.4, 0.3, 0.2]),
            "point.y": ([0.1, 0.5, 0.4, 0.3, 0.2]),
            "point.z": ([0.1, 0.5, 0.4, 0.3, 0.2]),
        },
        schema={
            "batch": pl.Int32,
            "model_realization": pl.UInt16,
            "perturbation": pl.Int32,
            "realization": pl.UInt16,
            "distance": pl.Float32,
            "point.x": pl.Float64,
            "point.y": pl.Float64,
            "point.z": pl.Float64,
        },
    )

    cache_hits = EverestRunModel.find_cached_results(
        control_values_to_evaluate,
        model_realizations_to_evaluate,
        all_results,
        ["point.x", "point.y", "point.z"],
    )

    # The cache hit is basically a mapping from
    # flat_index, which is the index of the control to be evaluated
    # (flat_index and simulation_id are both implicitly associated with
    # a model realization and perturbation)
    # to (batch, simulation_id)
    interesting_columns = [
        "flat_index",
        "batch",
        "simulation_id",
    ]
    assert cache_hits.select(interesting_columns).sort(
        interesting_columns
    ).to_dicts() == [
        {
            "batch": 0,
            "simulation_id": 1,
            "flat_index": 0,
        },
        {
            "batch": 0,
            "simulation_id": 2,
            "flat_index": 1,
        },
        {
            "batch": 0,
            "simulation_id": 0,
            "flat_index": 4,
        },
    ]
