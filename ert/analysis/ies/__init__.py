from typing import Optional, Union, Any
import numpy as np
import numpy.typing as npt
import res

# pylint: disable=import-error
from res._lib.ies import (  # type: ignore
    make_E,
    make_D,
    init_update,
    ModuleData,
    Config,
    inversion_type,
)


def make_X(  # pylint: disable=too-many-arguments
    S: npt.NDArray[np.double],
    R: npt.NDArray[np.double],
    E: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    A: npt.NDArray[np.double] = np.empty(shape=(0, 0)),
    ies_inversion: inversion_type = inversion_type.EXACT,
    truncation: Union[float, int] = 0.98,
    W0: Optional[npt.NDArray[np.double]] = None,
    step_length: float = 1.0,
    iteration: int = 1,
) -> Any:
    if W0 is None:
        W0 = np.zeros((S.shape[1], S.shape[1]))
    return res._lib.ies.make_X(  # pylint: disable=no-member, c-extension-no-member
        A,
        S,
        R,
        E,
        D,
        ies_inversion,
        truncation,
        W0,
        step_length,
        iteration,
    )


def update_A(  # pylint: disable=too-many-arguments
    data: ModuleData,
    A: npt.NDArray[np.double],
    Y: npt.NDArray[np.double],
    R: npt.NDArray[np.double],
    E: npt.NDArray[np.double],
    D: npt.NDArray[np.double],
    ies_inversion: inversion_type = inversion_type.EXACT,
    truncation: Union[float, int] = 0.98,
    step_length: float = 1.0,
) -> None:

    if not np.isfortran(A):
        raise TypeError("A matrix must be F_contiguous")
    res._lib.ies.update_A(  # pylint: disable=no-member, c-extension-no-member
        data, A, Y, R, E, D, ies_inversion, truncation, step_length
    )
