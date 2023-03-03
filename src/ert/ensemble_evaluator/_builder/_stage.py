import logging
import uuid
from typing import MutableSequence, Sequence

from typing_extensions import Self

from ._io_ import IO, DummyIOBuilder, IOBuilder

logger = logging.getLogger(__name__)


class Stage:
    def __init__(
        self, id_: str, name: str, inputs: Sequence[IO], outputs: Sequence[IO]
    ) -> None:
        self.id_ = id_
        self.inputs = inputs
        self.outputs = outputs
        self.name = name


class StageBuilder:
    def __init__(self) -> None:
        self._id: str = ""
        self._name: str = ""
        self._inputs: MutableSequence[IOBuilder] = []
        self._outputs: MutableSequence[IOBuilder] = []

    def set_id(self, id_: str) -> Self:
        self._id = id_
        return self

    def set_name(self, name: str) -> Self:
        self._name = name
        return self

    def add_output(self, output: IOBuilder) -> Self:
        self._outputs.append(output)
        return self

    def add_input(self, input_: IOBuilder) -> Self:
        self._inputs.append(input_)
        return self

    def build(self) -> Stage:
        if not self._id:
            self._id = str(uuid.uuid4())
        if not self._name:
            raise ValueError(f"invalid name for stage {self._name}")
        inputs = [builder.build() for builder in self._inputs]
        outputs = [builder.build() for builder in self._outputs]
        return Stage(self._id, self._name, inputs, outputs)

    def set_dummy_io(self) -> Self:
        self.add_input(DummyIOBuilder())
        self.add_output(DummyIOBuilder())
        return self
