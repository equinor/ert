import networkx as nx
import numpy as np
import polars as pl

from ert.analysis._enif_update import _load_numeric_parameters, prune_nan_nodes
from ert.config import SurfaceConfig
from ert.config.gen_data_config import GenDataConfig
from ert.config.gen_kw_config import DataSource, GenKwConfig
from ert.storage import open_storage


def test_that_prune_nan_nodes_preserves_adjacency_and_relabels():
    """
    Remove NaN boundary nodes from a 4x3 surface graph and verify:
    - Surviving nodes get contiguous 0..n-1 labels
    - Spatial adjacency is preserved
    - No false edges appear across the NaN gap
    """

    # 4x3 grid → 12 nodes
    config = SurfaceConfig(
        name="s",
        forward_init=False,
        update=True,
        ncol=4,
        nrow=3,
        xori=0,
        yori=0,
        xinc=1,
        yinc=1,
        rotation=0,
        yflip=1,
        forward_init_file="f",
        output_file="o",
        base_surface_path="b",
    )
    graph = config.load_parameter_graph()
    assert graph.number_of_nodes() == 12

    # NaN out the last row (nodes 2, 5, 8, 11 all have y=2)
    nan_mask = np.array(
        [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
        ]
    )
    pruned = prune_nan_nodes(graph, nan_mask)

    assert pruned.number_of_nodes() == 8
    assert set(pruned.nodes()) == set(range(8))

    # Relabeled: 0→0, 1→1, 3→2, 4→3, 6→4, 7→5, 9→6, 10→7
    expected_edges = {
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({4, 5}),
        frozenset({6, 7}),
        frozenset({0, 2}),
        frozenset({1, 3}),
        frozenset({2, 4}),
        frozenset({3, 5}),
        frozenset({4, 6}),
        frozenset({5, 7}),
    }
    assert set(map(frozenset, pruned.edges())) == expected_edges


def test_that_prune_nan_wall_creates_disconnected_components():
    """
    A NaN wall splitting a 3x3 surface creates two disconnected components
    with no false edges across the wall.
    """

    config = SurfaceConfig(
        name="s",
        forward_init=False,
        update=True,
        ncol=3,
        nrow=3,
        xori=0,
        yori=0,
        xinc=1,
        yinc=1,
        rotation=0,
        yflip=1,
        forward_init_file="f",
        output_file="o",
        base_surface_path="b",
    )
    graph = config.load_parameter_graph()

    # NaN out the middle row (nodes 1, 4, 7 all have y=1)
    nan_mask = np.array(
        [
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
        ]
    )
    pruned = prune_nan_nodes(graph, nan_mask)

    assert pruned.number_of_nodes() == 6

    components = list(nx.connected_components(pruned))
    assert len(components) == 2
    # After relabeling, original nodes 0,3,6 (first row) → 0,2,4
    # and original nodes 2,5,8 (last row) → 1,3,5
    assert {frozenset(c) for c in components} == {
        frozenset({0, 2, 4}),
        frozenset({1, 3, 5}),
    }

    for left in (0, 2, 4):
        for right in (1, 3, 5):
            assert not pruned.has_edge(left, right)


ENSEMBLE_SIZE = 10


def _dm_cfg(name):
    return GenKwConfig(
        name=name,
        group="DESIGN_MATRIX",
        update=False,
        input_source=DataSource.DESIGN_MATRIX,
        distribution={"name": "raw"},
    ).model_dump(mode="json")


def _sampled_genkw(name):
    return GenKwConfig(
        name=name,
        group=name,
        distribution={"name": "uniform", "min": 0.0, "max": 1.0},
    ).model_dump(mode="json")


def test_that_load_numeric_parameters_returns_float_and_int_but_not_string(tmp_path):
    """_load_numeric_parameters should return float and integer parameter
    groups and skip string/categorical ones."""
    rng = np.random.default_rng(42)

    params = [
        _sampled_genkw("MULT"),
        _dm_cfg("DM_FLOAT"),
        _dm_cfg("DM_INT"),
        _dm_cfg("DM_STRING"),
    ]

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="test",
            experiment_config={
                "parameter_configuration": params,
                "response_configuration": [
                    GenDataConfig(keys=["R"]).model_dump(mode="json")
                ],
                "observations": {},
            },
        )
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=ENSEMBLE_SIZE, iteration=0, name="e"
        )

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "MULT": rng.normal(size=ENSEMBLE_SIZE).tolist(),
                    "realization": range(ENSEMBLE_SIZE),
                }
            )
        )
        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "DM_FLOAT": rng.uniform(size=ENSEMBLE_SIZE).tolist(),
                    "DM_INT": rng.integers(1, 10, ENSEMBLE_SIZE).tolist(),
                    "realization": range(ENSEMBLE_SIZE),
                }
            )
        )
        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "DM_STRING": ["a", "b"] * (ENSEMBLE_SIZE // 2),
                    "realization": range(ENSEMBLE_SIZE),
                }
            )
        )

        iens = np.arange(ENSEMBLE_SIZE)
        names, arrays = _load_numeric_parameters(ensemble, iens)

        assert names == ["MULT", "DM_FLOAT", "DM_INT"]
        assert "DM_STRING" not in names
        for arr in arrays.values():
            assert np.issubdtype(arr.dtype, np.floating)


def test_that_load_numeric_parameters_casts_integers_to_float(tmp_path):
    """Integer parameter arrays should be cast to float64 so downstream
    linear algebra works without type errors."""
    params = [_dm_cfg("DM_INT")]

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            name="test",
            experiment_config={
                "parameter_configuration": params,
                "response_configuration": [
                    GenDataConfig(keys=["R"]).model_dump(mode="json")
                ],
                "observations": {},
            },
        )
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=ENSEMBLE_SIZE, iteration=0, name="e"
        )
        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "DM_INT": list(range(ENSEMBLE_SIZE)),
                    "realization": range(ENSEMBLE_SIZE),
                }
            )
        )

        iens = np.arange(ENSEMBLE_SIZE)
        names, arrays = _load_numeric_parameters(ensemble, iens)

        assert names == ["DM_INT"]
        assert arrays["DM_INT"].dtype == np.float64
