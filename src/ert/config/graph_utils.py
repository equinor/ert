from __future__ import annotations

import networkx as nx


def create_flattened_cube_graph(px: int, py: int, pz: int) -> nx.Graph[int]:
    """Creates a grid graph for sparse precision estimation in EnIF updates.

    The graph encodes spatial adjacency between parameter cells. Integer node
    labels (0 to px*py*pz-1) match C-order flattening so precision matrix
    indices align with parameter vector positions.
    """
    grid_graph = nx.grid_graph(dim=[px, py, pz])
    return nx.convert_node_labels_to_integers(grid_graph, ordering="sorted")
