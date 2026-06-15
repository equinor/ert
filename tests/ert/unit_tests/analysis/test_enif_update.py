import networkx as nx
import numpy as np
import pytest

from ert.analysis._enif_update import compute_nan_masks, prune_nan_nodes
from ert.config import SurfaceConfig


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
        update_strategy="global",
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
        update_strategy="global",
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


NAN = np.nan


@pytest.mark.parametrize(
    ("param_arrays", "expected_sizes", "expected_nan_count", "expected_clean_rows"),
    [
        pytest.param(
            {
                "A": np.array(
                    [[1, 2, 3], [NAN, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float
                ),
                "B": np.array([[10, 11, 12], [13, 14, 15]], dtype=float),
            },
            {"A": 3, "B": 2},
            1,
            5,
            id="one_nan_row_in_first_group",
        ),
        pytest.param(
            {"P": np.array([[1, 2, 3], [1, NAN, 3], [1, 2, 3]], dtype=float)},
            {"P": 2},
            1,
            2,
            id="partial_nan_in_single_realization",
        ),
        pytest.param(
            {
                "A": np.array([[1, 2], [3, 4]], dtype=float),
                "B": np.array([[5, 6]], dtype=float),
            },
            {"A": 2, "B": 1},
            0,
            3,
            id="no_nans",
        ),
        pytest.param(
            {"X": np.array([[NAN, NAN], [NAN, NAN]], dtype=float)},
            {"X": 0},
            2,
            0,
            id="all_nan",
        ),
        pytest.param(
            {
                "A": np.array([[1, 2], [NAN, NAN]], dtype=float),
                "B": np.array([[3, 4]], dtype=float),
                "C": np.array([[NAN, 5], [6, 7], [8, 9]], dtype=float),
            },
            {"A": 1, "B": 1, "C": 2},
            2,
            4,
            id="multiple_groups_mixed_nans",
        ),
    ],
)
def test_that_compute_nan_masks_filters_nan_rows(
    param_arrays, expected_sizes, expected_nan_count, expected_clean_rows
):
    _masks, sizes, X_full, nan_row_mask = compute_nan_masks(param_arrays)

    assert sizes == expected_sizes
    assert int(nan_row_mask.sum()) == expected_nan_count
    assert X_full[~nan_row_mask].shape[0] == expected_clean_rows

    total_rows = sum(a.shape[0] for a in param_arrays.values())
    assert X_full.shape[0] == total_rows
    assert len(nan_row_mask) == total_rows
