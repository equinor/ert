import logging
from collections import defaultdict
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Tuple

from typing_extensions import Self

from ._step import Step, StepBuilder

SOURCE_TEMPLATE_REAL = "/real/{iens}"
if TYPE_CHECKING:
    from ._stage import StageBuilder


logger = logging.getLogger(__name__)


def _sort_steps(steps: Sequence["Step"]) -> Tuple[str, ...]:
    """Return a tuple comprised by step names in the order they should be
    executed."""
    graph = defaultdict(set)
    if len(steps) == 1:
        return (steps[0].name,)
    edged_nodes = set()
    for step in steps:
        for other in steps:
            if step == other:
                continue
            step_outputs = set(io.name for io in step.outputs)
            other_inputs = set(io.name for io in other.inputs)
            if len(step_outputs) > 0 and not step_outputs.isdisjoint(other_inputs):
                graph[other.name].add(step.name)
                edged_nodes.add(step.name)
                edged_nodes.add(other.name)

    isolated_nodes = set(step.name for step in steps) - edged_nodes
    for node in isolated_nodes:
        graph[node] = set()

    ts = TopologicalSorter(graph)
    return tuple(ts.static_order())


class Realization:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        iens: int,
        steps: Sequence[Step],
        active: bool,
        source: str,
        ts_sorted_steps: Optional[Tuple[str, ...]] = None,
    ):
        if iens is None:
            raise ValueError(f"{self} needs iens")
        if steps is None:
            raise ValueError(f"{self} needs steps")
        if active is None:
            raise ValueError(f"{self} needs to be set either active or not")

        self.iens = iens
        self.steps = steps
        self.active = active

        self._source = source

        self._ts_sorted_indices = None
        if ts_sorted_steps is not None:
            self._ts_sorted_indices = list(range(0, len(ts_sorted_steps)))
            for idx, name in enumerate(ts_sorted_steps):
                for step_idx, step in enumerate(steps):
                    if step.name == name:
                        self._ts_sorted_indices[idx] = step_idx
            if len(self._ts_sorted_indices) != len(steps):
                raise ValueError(
                    "disparity between amount of sorted items "
                    f"({self._ts_sorted_indices}) and steps, possibly duplicate "
                    + "step name?"
                )

    def set_active(self, active: bool) -> None:
        self.active = active

    def source(self) -> str:
        return self._source

    def get_steps_sorted_topologically(self) -> Iterator[Step]:
        steps = self.steps
        if not self._ts_sorted_indices:
            raise NotImplementedError("steps were not sorted")
        for idx in self._ts_sorted_indices:
            yield steps[idx]


class RealizationBuilder:
    def __init__(self) -> None:
        self._steps: List[StepBuilder] = []
        self._stages: List["StageBuilder"] = []
        self._active: Optional[bool] = None
        self._iens: Optional[int] = None
        self._parent_source: Optional[str] = None

    def active(self, active: bool) -> Self:
        self._active = active
        return self

    def add_step(self, step: StepBuilder) -> Self:
        self._steps.append(step)
        return self

    def add_stage(self, stage: "StageBuilder") -> Self:
        self._stages.append(stage)
        return self

    def set_iens(self, iens: int) -> Self:
        self._iens = iens
        return self

    def set_parent_source(self, source: str) -> Self:
        self._parent_source = source
        return self

    def build(self) -> Realization:
        if not self._iens:
            # assume this is being used as a forward model, thus should be 0
            self._iens = 0
        if not self._parent_source:
            raise ValueError(f"need parent_source for realization: {self._iens}")

        source = self._parent_source + SOURCE_TEMPLATE_REAL.format(iens=self._iens)

        if self._active is None:
            raise ValueError(f"realization {self._iens}: active should be set")

        steps = [builder.set_parent_source(source).build() for builder in self._steps]

        ts_sorted_steps = _sort_steps(steps)

        return Realization(
            self._iens,
            steps,
            self._active,
            source=source,
            ts_sorted_steps=ts_sorted_steps,
        )
