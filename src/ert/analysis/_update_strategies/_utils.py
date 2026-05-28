import io

import numpy as np
import numpy.typing as npt
import scipy as sp


def matrix_to_bytes(
    matrix: npt.NDArray[np.float64],
) -> tuple[bytes, str, tuple[int, int], bool]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    data_type = str(matrix.dtype)

    sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)
    sparse = bool(sparsity > 0.5)

    buf = io.BytesIO()
    if sparse:
        sp.sparse.save_npz(buf, sp.sparse.csc_array(matrix))
    else:
        np.save(buf, matrix)

    return buf.getvalue(), data_type, shape, sparse
