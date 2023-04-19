from dataclasses import astuple, dataclass
from enum import Enum, auto, unique
from typing import Any, List, Optional, Tuple

import ecl_data_io as eclio
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import arrays


@unique
class Units(Enum):
    METRES = auto()
    CM = auto()
    FEET = auto()

    def to_ecl(self):
        return self.name.ljust(8)


@unique
class GridRelative(Enum):
    """GridRelative is the second value given GRIDUNIT keyword.

    MAP means map relative units, while
    leaving it blank means relative to the origin given by the
    MAPAXES keyword.
    """

    MAP = auto()
    ORIGIN = auto()

    def to_ecl(self) -> str:
        if self == GridRelative.MAP:
            return "MAP".ljust(8)
        else:
            return "".ljust(8)


@dataclass
class GrdeclKeyword:
    """An abstract grdecl keyword.

    Gives a general implementation of to/from grdecl which recurses on
    fields. Ie. a dataclass such as
    >>> class A(GrdeclKeyword):
    ...     ...
    >>> class B(GrdeclKeyword):
    ...     ...

    >>> @dataclass
    ... class MyKeyword(GrdeclKeyword):
    ...     field1: A
    ...     field2: B

    will have a to_ecl method that will be similar to

    >>> def to_ecl(self):
    ...     return [self.field1.to_ecl(), self.field2.to_ecl]
    """

    def to_ecl(self) -> List[Any]:
        return [value.to_ecl() for value in astuple(self)]


@dataclass
class GridUnit(GrdeclKeyword):
    """Defines the units used for grid dimensions.

    The first value is a string describing the units used, defaults to METRES,
    known accepted other units are FIELD and LAB. The last value describes
    whether the measurements are relative to the map or to the origin of
    MAPAXES.
    """

    unit: Units = Units.METRES
    grid_relative: GridRelative = GridRelative.ORIGIN


@unique
class CoordinateType(Enum):
    """The coordinate system type given in the SPECGRID keyword.

    This is given by either T or F in the last value of SPECGRID, meaning
    either cylindrical or cartesian coordinates respectively.
    """

    CARTESIAN = auto()
    CYLINDRICAL = auto()

    def to_ecl(self) -> int:
        if self == CoordinateType.CARTESIAN:
            return 0
        else:
            return 1


@unique
class TypeOfGrid(Enum):
    """
    A Grid has three possible data layout formats, UNSTRUCTURED, CORNER_POINT,
    BLOCK_CENTER and COMPOSITE (meaning combination of the two former). Only
    CORNER_POINT layout is supported by XTGeo.
    """

    COMPOSITE = 0
    CORNER_POINT = 1
    UNSTRUCTURED = 2
    BLOCK_CENTER = 3

    @property
    def alternate_value(self):
        """Inverse of alternate_code."""
        alternate_value = 0
        if self == TypeOfGrid.CORNER_POINT:
            alternate_value = 0
        elif self == TypeOfGrid.UNSTRUCTURED:
            alternate_value = 1
        elif self == TypeOfGrid.COMPOSITE:
            alternate_value = 2
        elif self == TypeOfGrid.BLOCK_CENTER:
            alternate_value = 3
        else:
            raise ValueError(f"Unknown grid type {self}")
        return alternate_value


@unique
class RockModel(Enum):
    """
    Type of rock model.
    """

    SINGLE_PERMEABILITY_POROSITY = 0
    DUAL_POROSITY = 1
    DUAL_PERMEABILITY = 2


@unique
class GridFormat(Enum):
    """
    The format of the "original grid", ie., what
    method was used to construct the values in the file.
    """

    UNKNOWN = 0
    IRREGULAR_CORNER_POINT = 1
    REGULAR_CARTESIAN = 2


@dataclass
class Filehead:
    """
    The first keyword in an egrid file is the FILEHEAD
    keyword, containing metadata about the file and its
    content.
    """

    version_number: int
    year: int
    version_bound: int
    type_of_grid: TypeOfGrid
    rock_model: RockModel
    grid_format: GridFormat

    def to_ecl(self) -> np.ndarray:
        """
        Returns:
            List of values, as layed out after the FILEHEAD keyword for
            the given filehead.
        """
        # The data is expected to consist of
        # 100 integers, but only a subset is used.
        result = np.zeros((100,), dtype=np.int32)
        result[0] = self.version_number
        result[1] = self.year
        result[3] = self.version_bound
        result[4] = self.type_of_grid.alternate_value
        result[5] = self.rock_model.value
        result[6] = self.grid_format.value
        return result


@dataclass
class GridHead:
    """
    Both for lgr (see LGRSection) and the global grid (see GlobalGrid)
    the GRIDHEAD keyword indicates the start of the grid layout for that
    section.
    """

    type_of_grid: TypeOfGrid
    num_x: int
    num_y: int
    num_z: int
    grid_reference_number: int
    numres: int
    nseg: int
    coordinate_type: CoordinateType
    lgr_start: Tuple[int, int, int]
    lgr_end: Tuple[int, int, int]

    def to_ecl(self) -> np.ndarray:
        # The data is expected to consist of
        # 100 integers, but only a subset is used.
        result = np.zeros((100,), dtype=np.int32)
        result[0] = self.type_of_grid.value
        result[1] = self.num_x
        result[2] = self.num_y
        result[3] = self.num_z
        result[4] = self.grid_reference_number
        result[24] = self.numres
        result[25] = self.nseg
        result[26] = self.coordinate_type.to_ecl()
        result[[27, 28, 29]] = np.array(self.lgr_start)
        result[[30, 31, 32]] = np.array(self.lgr_end)
        return result


@dataclass
class GlobalGrid:
    """
    The global grid contains the layout of the grid before
    refinements, and the sectioning into grid coarsening
    through the optional corsnum keyword.
    """

    grid_head: GridHead
    coord: np.ndarray
    zcorn: np.ndarray
    actnum: Optional[np.ndarray] = None

    def __eq__(self, other):
        if not isinstance(other, GlobalGrid):
            return False
        return (
            self.grid_head == other.grid_head
            and np.array_equal(self.actnum, other.actnum)
            and np.array_equal(self.coord, other.coord)
            and np.array_equal(self.zcorn, other.zcorn)
        )

    def to_ecl(self) -> List[Tuple[str, Any]]:
        result = [
            ("GRIDHEAD", self.grid_head.to_ecl()),
            ("COORD   ", self.coord.astype(np.float32)),
            ("ZCORN   ", self.zcorn.astype(np.float32)),
        ]
        if self.actnum is not None:
            result.append(("ACTNUM  ", self.actnum.astype(np.int32)))
        result.append(("ENDGRID ", np.array([], dtype=np.int32)))
        return result


@dataclass
class EGrid:
    """Contains the data of an EGRID file.

    Args:
        file_head: The file header starting with the FILEHEAD keyword
        global_grid: The global grid
        lgr_sections: List of local grid refinements.
        nnc_sections: Describe non-neighboring sections as a list of either
            NNCSections or AmalgamationSection's.
    """

    file_head: Filehead
    global_grid: GlobalGrid

    def to_file(
        self,
        filelike,
    ):
        """
        write the EGrid to file.
        Args:
            filelike (str,Path,stream): The egrid file to write to.
            file_format (ecl_data_io.Format): The format of the file.
        """
        contents = []
        contents.append(("FILEHEAD", self.file_head.to_ecl()))
        contents += self.global_grid.to_ecl()
        eclio.write(filelike, contents)


finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)

indices = st.integers(min_value=4, max_value=6)
units = st.sampled_from(Units)
grid_relatives = st.sampled_from(GridRelative)
coordinate_types = st.sampled_from(CoordinateType)
grid_units = st.builds(GridUnit, units, grid_relatives)


@st.composite
def zcorns(draw, dims):
    return draw(
        arrays(
            shape=8 * dims[0] * dims[1] * dims[2],
            dtype=np.float32,
            elements=finites,
        )
    )


types_of_grid = st.just(TypeOfGrid.CORNER_POINT)
rock_models = st.sampled_from(RockModel)
grid_formats = st.just(GridFormat.IRREGULAR_CORNER_POINT)
file_heads = st.builds(
    Filehead,
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=2000, max_value=2022),
    st.integers(min_value=0, max_value=5),
    types_of_grid,
    rock_models,
    grid_formats,
)

grid_heads = st.builds(
    GridHead,
    types_of_grid,
    indices,
    indices,
    indices,
    indices,
    st.just(1),
    st.just(1),
    coordinate_types,
    st.tuples(indices, indices, indices),
    st.tuples(indices, indices, indices),
)


@st.composite
def global_grids(draw):
    grid_head = draw(grid_heads)
    dims = (grid_head.num_x, grid_head.num_y, grid_head.num_z)
    corner_size = (dims[0] + 1) * (dims[1] + 1) * 6
    coord = arrays(
        shape=corner_size,
        dtype=np.float32,
        elements=finites,
    )
    actnum = st.one_of(
        st.just(None),
        arrays(
            shape=dims[0] * dims[1] * dims[2],
            dtype=np.int32,
            elements=st.integers(min_value=0, max_value=3),
        ),
    )
    return GlobalGrid(
        coord=draw(coord),
        zcorn=draw(zcorns(dims)),
        actnum=draw(actnum),
        grid_head=grid_head,
    )


egrids = st.builds(EGrid, file_heads, global_grids())
