import hypothesis.strategies as st
import numpy as np
import xtgeo
from resfo_utilities.testing import EGrid, egrids

indices = st.integers(min_value=1, max_value=4)
finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)


def _has_contiguous_active_columns(egrid: EGrid) -> bool:
    grid = egrid.global_grid
    if grid.actnum is None:
        return True
    head = grid.grid_head
    is_active_column = grid.actnum.reshape(
        (head.num_x, head.num_y, head.num_z), order="F"
    ).any(axis=(1, 2))
    return bool(np.all(np.diff(np.flatnonzero(is_active_column)) == 1))


# xtgeo cannot give the handedness of nested hybrid grids, which it
# recognises by fully inactive I-columns separating active ones.
field_egrids = egrids.filter(_has_contiguous_active_columns)


@st.composite
def xtgeo_box_grids(draw, dimensions=None):
    """Generate XTGeo Grid instances using create_box_grid."""
    # Draw grid dimensions
    if dimensions:
        nx, ny, nz = dimensions
    else:
        nx = draw(indices)
        ny = draw(indices)
        nz = draw(indices)

    # Draw origin coordinates
    origin = (
        draw(finites),  # x origin
        draw(finites),  # y origin
        draw(
            st.floats(min_value=0, max_value=100, width=32)
        ),  # z origin (positive depth)
    )

    # Draw increments (cell sizes) - keep positive
    increment = (
        draw(st.floats(min_value=10, max_value=1000, width=32)),  # x increment
        draw(st.floats(min_value=10, max_value=1000, width=32)),  # y increment
        draw(st.floats(min_value=1, max_value=100, width=32)),  # z increment
    )

    # Draw rotation and flip
    rotation = draw(st.floats(min_value=0, max_value=360, width=32))
    flip = draw(st.sampled_from([1, -1]))

    # Create the grid using XTGeo
    return xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=origin,
        increment=increment,
        rotation=rotation,
        flip=flip,
    )
