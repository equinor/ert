import numpy as np
import pytest

from ert.config.graph_utils import create_flattened_cube_graph


@pytest.mark.parametrize(
    ("px", "py", "pz"),
    [
        (1, 1, 1),
        (1, 1, 8),
        (1, 8, 1),
        (8, 1, 1),
        (2, 2, 2),
        (2, 3, 4),
        (3, 5, 7),
        (8, 8, 8),
    ],
)
def test_that_flattened_graph_edges_connect_neighboring_grid_cells(px, py, pz):
    flattened_graph = create_flattened_cube_graph(px, py, pz)
    grid_shape = (px, py, pz)

    for flat_node_a, flat_node_b in flattened_graph.edges():
        cell_coord_a = np.unravel_index(flat_node_a, grid_shape)
        cell_coord_b = np.unravel_index(flat_node_b, grid_shape)

        manhattan_distance = sum(
            abs(coord_a - coord_b)
            for coord_a, coord_b in zip(cell_coord_a, cell_coord_b, strict=True)
        )

        assert manhattan_distance == 1, (
            f"Nodes {flat_node_a} and {flat_node_b} are connected, but "
            f"their coordinates {cell_coord_a} and {cell_coord_b} are not neighbors."
        )
