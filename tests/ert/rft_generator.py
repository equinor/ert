from functools import partial

import numpy as np


def well_etc(
    time_units=b"HOURS",
    lgr_name=b"",
    data_category=b"R",
    well_name=b"WELL1",
):
    return np.array(
        [
            time_units,
            well_name,
            lgr_name,
            b"METRES",
            b"BARSA",
            data_category,
            b"STANDARD",
            b"SM3/DAY",
            b"SM3/DAY",
            b"RM3/DAY",
            b"",
            b"M/SEC",
            b"CP",
            b"KG/SM3",
            b"KG/DAY",
            b"KG/KG",
        ]
    )


float_arr = partial(np.array, dtype=np.float32)
int_arr = partial(np.array, dtype=np.int32)


def cell_start(date=(1, 1, 2000), ijks=((1, 1, 1), (2, 1, 2)), *args, **kwargs):
    return [
        ("TIME    ", float_arr([24.0])),
        ("DATE    ", int_arr(date)),
        ("WELLETC ", well_etc(*args, **kwargs)),
        ("CONIPOS ", int_arr([i for i, _, _ in ijks])),
        ("CONJPOS ", int_arr([j for _, j, _ in ijks])),
        ("CONKPOS ", int_arr([k for _, _, k in ijks])),
    ]


def pad_to(lst: list[int], target_len: int):
    return np.pad(
        np.array(lst, dtype=np.int32), (0, target_len - len(lst)), mode="constant"
    )


def create_egrid(
    nx, ny, nz, x_width, y_width, layer_height, x_offset=0.0, y_offset=0.0, z_offset=0.0
):
    """EGrid file contents with nz layers, nx cells in the i direction and ny cells in
    the j direction.

    Each cell has width x_width in the i direction and y_width in the j direction and
    height layer_height in the z direction.

    The first cell starts at (x_offset, y_offset, z_offset) which defaults to (0,0,0)
    """

    height = nz * layer_height
    cells_per_layer = nx * ny

    coord = np.array(
        [
            [
                i * x_width + x_offset,
                j * y_width + y_offset,
                z_offset,
                i * x_width + x_offset,
                j * y_width + y_offset,
                height + z_offset,
            ]
            for j in range(ny + 1)
            for i in range(nx + 1)
        ],
        dtype=">f4",
    )

    zcoord = np.array(
        [
            [z * layer_height + z_offset] * (cells_per_layer * 4)
            + [(z + 1) * layer_height + z_offset] * (cells_per_layer * 4)
            for z in range(nz)
        ],
        dtype=">f4",
    )
    return [
        ("FILEHEAD", pad_to([3, 2007, 0, 0, 0, 0, 1], 100)),
        ("MAPAXES ", np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=">f4")),
        ("GRIDUNIT", np.array([b"METRES  ", b"        "], dtype="|S8")),
        ("GRIDHEAD", pad_to([1, nx, ny, nz], 100)),
        ("COORD   ", coord.ravel()),
        ("ZCORN   ", zcoord.ravel()),
        ("ACTNUM  ", np.ones(nx * ny * nz, dtype=">i4")),
        ("ENDGRID ", np.array([], dtype=">i4")),
    ]
