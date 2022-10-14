import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ert._c_wrappers.enkf import QueueConfig
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig

from ._ensemble import Ensemble
from ._io_ import DummyIOBuilder
from ._io_map import InputMap, OutputMap
from ._legacy import LegacyEnsemble
from ._prefect import PrefectEnsemble
from ._realization import RealizationBuilder

if TYPE_CHECKING:
    import ert

SOURCE_TEMPLATE_ENS = "/ert/ensemble/{ens_id}"
logger = logging.getLogger(__name__)


class EnsembleBuilder:  # pylint: disable=too-many-instance-attributes
    def __init__(self) -> None:
        self._reals: List[RealizationBuilder] = []
        self._forward_model: Optional[RealizationBuilder] = None
        self._size: int = 0
        self._metadata: Dict[str, Any] = {}
        self._legacy_dependencies: Optional[Tuple[QueueConfig, AnalysisConfig]] = None
        self._inputs: Dict[int, Dict[str, "ert.data.RecordTransmitter"]] = {}
        self._outputs: Dict[int, Dict[str, "ert.data.RecordTransmitter"]] = {}

        self._custom_port_range: Optional[range] = None
        self._max_running = 10000
        self._max_retries = 0
        self._retry_delay = 5
        self._executor: str = "local"
        self._id: Optional[str] = None

    def set_forward_model(self, forward_model: RealizationBuilder) -> "EnsembleBuilder":
        if self._reals:
            raise ValueError(
                "Cannot set forward model when realizations are already specified"
            )
        self._forward_model = forward_model
        return self

    def add_realization(self, real: RealizationBuilder) -> "EnsembleBuilder":
        if self._forward_model:
            raise ValueError("Cannot add realization when forward model is specified")

        self._reals.append(real)
        return self

    def set_metadata(self, key: str, value: Any) -> "EnsembleBuilder":
        self._metadata[key] = value
        return self

    def set_ensemble_size(self, size: int) -> "EnsembleBuilder":
        """Duplicate the ensemble members that existed at build time so as to
        get the desired state."""
        self._size = size
        return self

    def set_legacy_dependencies(
        self, queue_config: QueueConfig, analysis_config: AnalysisConfig
    ) -> "EnsembleBuilder":
        self._legacy_dependencies = (queue_config, analysis_config)
        return self

    def set_inputs(self, inputs: InputMap) -> "EnsembleBuilder":
        self._inputs = inputs.to_dict()
        return self

    def set_outputs(self, outputs: OutputMap) -> "EnsembleBuilder":
        self._outputs = outputs.to_dict()
        return self

    def set_custom_port_range(self, custom_port_range: range) -> "EnsembleBuilder":
        self._custom_port_range = custom_port_range
        return self

    def set_max_running(self, max_running: int) -> "EnsembleBuilder":
        self._max_running = max_running
        return self

    def set_max_retries(self, max_retries: int) -> "EnsembleBuilder":
        self._max_retries = max_retries
        return self

    def set_retry_delay(self, retry_delay: int) -> "EnsembleBuilder":
        self._retry_delay = retry_delay
        return self

    def set_executor(self, executor: str) -> "EnsembleBuilder":
        self._executor = executor
        return self

    def set_id(self, id_: str) -> "EnsembleBuilder":
        self._id = id_
        return self

    def _build_io_maps(self, reals: List[RealizationBuilder]) -> None:
        i_matrix: Dict[int, Dict[str, Optional["ert.data.RecordTransmitter"]]] = {
            i: {} for i in range(len(reals))
        }
        o_matrix: Dict[int, Dict[str, Optional["ert.data.RecordTransmitter"]]] = {
            i: {} for i in range(len(reals))
        }
        o_names = set()
        for iens, real in enumerate(reals):
            for step in real._steps:
                for output in step._outputs:
                    if isinstance(output, DummyIOBuilder):
                        continue
                    factory = output.transmitter_factory(iens)
                    assert output._name  # mypy
                    o_matrix[iens][output._name] = factory() if factory else None
                    o_names.add(output._name)
        for iens, real in enumerate(reals):
            for step in real._steps:
                for input_ in step._inputs:
                    if isinstance(input_, DummyIOBuilder):
                        continue
                    if input_._name not in o_names:
                        factory = input_.transmitter_factory(iens)
                        assert input_._name  # mypy
                        i_matrix[iens][input_._name] = factory() if factory else None
        self._inputs = InputMap.from_dict(i_matrix).validate().to_dict()
        self._outputs = OutputMap.from_dict(o_matrix).validate().to_dict()

    def build(self) -> Ensemble:
        if not (self._reals or self._forward_model):
            raise ValueError("Either forward model or realizations needs to be set")

        if self._id is None:
            raise ValueError("ID must be set prior to building")

        real_builders: List[RealizationBuilder] = []
        if self._forward_model:
            # duplicate the original forward model into realizations
            for i in range(self._size):
                logger.debug(f"made deep-copied real {i}")
                real = copy.deepcopy(self._forward_model)
                real.set_iens(i)
                real_builders.append(real)
        else:
            real_builders = self._reals

        # legacy has dummy IO, so no need to build an IO map
        if not self._legacy_dependencies:
            self._build_io_maps(real_builders)

        source = SOURCE_TEMPLATE_ENS.format(ens_id=self._id)

        reals = [builder.set_parent_source(source).build() for builder in real_builders]

        if self._legacy_dependencies:
            return LegacyEnsemble(
                reals, self._metadata, *self._legacy_dependencies, id_=self._id
            )
        else:
            return PrefectEnsemble(
                reals=reals,
                inputs=self._inputs,
                outputs=self._outputs,
                max_running=self._max_running,
                max_retries=self._max_retries,
                executor=self._executor,
                retry_delay=self._retry_delay,
                custom_port_range=self._custom_port_range,
                id_=self._id,
            )
