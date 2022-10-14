import logging
from typing import TYPE_CHECKING, Dict, Optional, Type, cast

from ert.data import TransformationDirection

if TYPE_CHECKING:
    import ert


logger = logging.getLogger(__name__)


class IO:
    def __init__(
        self,
        name: str,
        transformation: Optional["ert.data.RecordTransformation"] = None,
    ):
        self.transformation = transformation
        self.name = name


class DummyIO(IO):
    pass


class _Input(IO):
    pass


class _Output(IO):
    pass


class IOBuilder:
    _concrete_cls: Optional[Type[IO]] = None

    _TRANSMITTER_FACTORY_ALL = -1

    def __init__(self) -> None:
        self._name: Optional[str] = None
        self._transformation: Optional["ert.data.RecordTransformation"] = None
        self._transmitter_factories: Dict[int, "ert.data.transmitter_factory"] = {}

    def set_name(self: "IOBuilder", name: str) -> "IOBuilder":
        self._name = name
        return self

    def set_transformation(
        self, transformation: "ert.data.RecordTransformation"
    ) -> "IOBuilder":
        # Validate that a transformation can transform in the required direction.
        # The constraints are in tuple form: (IO type, required direction).
        constraints = (
            (_Input, TransformationDirection.FROM_RECORD),
            (_Output, TransformationDirection.TO_RECORD),
        )
        for constraint in constraints:
            cls, direction = constraint
            if self._concrete_cls == cls and direction not in transformation.direction:
                raise ValueError(
                    f"cannot set transformation to '{type(transformation).__name__}' "
                    + f"since it does not allow '{direction}', only "
                    + f"'{transformation.direction}'. A {cls.__name__} "
                    + "transformation must allow this."
                )
        self._transformation = transformation
        return self

    def set_transmitter_factory(
        self, factory: "ert.data.transmitter_factory", index: Optional[int] = None
    ) -> "IOBuilder":
        """Fix the transmitter factory for this IO to index if index is >= 0. If the
        index is omitted or is < 0, it the factory will be called for all indices not
        fixed to an index. So either one transmitter factory is used for all indices
        (i.e. ensemble members), or all ensemble members have a unique factory.
        """
        if index is None or index < 0:
            index = self._TRANSMITTER_FACTORY_ALL
        self._transmitter_factories[index] = factory
        return self

    def transmitter_factory(
        self, index: Optional[int] = None
    ) -> Optional["ert.data.transmitter_factory"]:
        """Return a fixed transmitter factory for index, a ensemble-wide transmitter
        if index is -1, or None.
        """
        global_factory = self._TRANSMITTER_FACTORY_ALL in self._transmitter_factories
        if index is None or index == self._TRANSMITTER_FACTORY_ALL:
            if not global_factory:
                return None
        if index not in self._transmitter_factories:
            if not global_factory:
                return None
            return self._transmitter_factories[self._TRANSMITTER_FACTORY_ALL]
        return self._transmitter_factories[index]

    def build(self) -> IO:
        if self._concrete_cls is None:
            raise TypeError("cannot build IO")
        if self._name is None:
            raise ValueError("missing name for IO")
        return self._concrete_cls(  # pylint: disable=not-callable
            self._name, transformation=self._transformation
        )


class DummyIOBuilder(IOBuilder):
    _concrete_cls = DummyIO

    def build(self) -> DummyIO:
        super().set_name("dummy i/o")
        return cast(DummyIO, super().build())


class InputBuilder(IOBuilder):
    _concrete_cls = _Input


class OutputBuilder(IOBuilder):
    _concrete_cls = _Output
