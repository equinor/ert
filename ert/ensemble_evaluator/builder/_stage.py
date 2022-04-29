import logging
import uuid
from typing import (
    MutableSequence,
    Sequence,
)

from ._io_ import _IO, _IOBuilder, _DummyIOBuilder


logger = logging.getLogger(__name__)


class _Stage:
    def __init__(
        self, id_: str, name: str, inputs: Sequence[_IO], outputs: Sequence[_IO]
    ) -> None:
        self.id_ = id_
        self.inputs = inputs
        self.outputs = outputs
        self.name = name


class _StageBuilder:
    def __init__(self) -> None:
        self._id: str = ""
        self._name: str = ""
        self._inputs: MutableSequence[_IOBuilder] = []
        self._outputs: MutableSequence[_IOBuilder] = []

    def set_id(self, id_: str) -> "_StageBuilder":
        self._id = id_
        return self

    def set_name(self, name: str) -> "_StageBuilder":
        self._name = name
        return self

    def add_output(self, output: _IOBuilder) -> "_StageBuilder":
        self._outputs.append(output)
        return self

    def add_input(self, input_: _IOBuilder) -> "_StageBuilder":
        self._inputs.append(input_)
        return self

    def build(self) -> _Stage:
        if not self._id:
            self._id = str(uuid.uuid4())
        if not self._name:
            raise ValueError(f"invalid name for stage {self._name}")
        inputs = [builder.build() for builder in self._inputs]
        outputs = [builder.build() for builder in self._outputs]
        return _Stage(self._id, self._name, inputs, outputs)

    def set_dummy_io(self) -> "_StageBuilder":
        self.add_input(_DummyIOBuilder())
        self.add_output(_DummyIOBuilder())
        return self
