import hypothesis.strategies as st
import xtgeo

indices = st.integers(min_value=1, max_value=4)
finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)


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
    grid = xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=origin,
        increment=increment,
        rotation=rotation,
        flip=flip,
    )

    return grid
