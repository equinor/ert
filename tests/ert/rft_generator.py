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
