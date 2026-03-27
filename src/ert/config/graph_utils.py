from __future__ import annotations

import itertools

import networkx as nx


def create_flattened_cube_graph(px: int, py: int, pz: int) -> nx.Graph[int]:
    """Graph created with nodes numbered from 0 to px*py*pz
    corresponds to the "vectorization" or flattening of
    a 3D cube with shape (px,py,pz) in the same way as
    reshaping such a cube into a one-dimensional array.
    The indexing scheme used to create the graph reflects
    this flattening process"""

    graph: nx.Graph[int] = nx.Graph()
    for x, y, z in itertools.product(range(px), range(py), range(pz)):
        index = x * py * pz + y * pz + z

        if y < py - 1:
            graph.add_edge(index, index + pz)

        if x < px - 1:
            graph.add_edge(index, index + py * pz)

        if z < pz - 1:
            graph.add_edge(index, index + 1)

    return graph
