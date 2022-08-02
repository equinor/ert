import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from res import _lib
from res.enkf import QueueConfig
from res.enkf.analysis_config import AnalysisConfig
from res.enkf.ert_run_context import RunContext
from res.enkf.res_config import ResConfig
from res.job_queue.forward_model import ForwardModel

from ._ensemble import _Ensemble
from ._io_ import _DummyIOBuilder
from ._io_map import InputMap, OutputMap
from ._job import _LegacyJobBuilder
from ._legacy import _LegacyEnsemble
from ._prefect import PrefectEnsemble
from ._realization import _RealizationBuilder
from ._step import _StepBuilder

if TYPE_CHECKING:
    import ert

logger = logging.getLogger(__name__)


class _EnsembleBuilder:  # pylint: disable=too-many-instance-attributes
    def __init__(self) -> None:
        self._reals: List[_RealizationBuilder] = []
        self._forward_model: Optional[_RealizationBuilder] = None
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

    def set_forward_model(
        self, forward_model: _RealizationBuilder
    ) -> "_EnsembleBuilder":
        if self._reals:
            raise ValueError(
                "Cannot set forward model when realizations are already specified"
            )
        self._forward_model = forward_model
        return self

    def add_realization(self, real: _RealizationBuilder) -> "_EnsembleBuilder":
        if self._forward_model:
            raise ValueError("Cannot add realization when forward model is specified")

        self._reals.append(real)
        return self

    def set_metadata(self, key: str, value: Any) -> "_EnsembleBuilder":
        self._metadata[key] = value
        return self

    def set_ensemble_size(self, size: int) -> "_EnsembleBuilder":
        """Duplicate the ensemble members that existed at build time so as to
        get the desired state."""
        self._size = size
        return self

    def set_legacy_dependencies(
        self, queue_config: QueueConfig, analysis_config: AnalysisConfig
    ) -> "_EnsembleBuilder":
        self._legacy_dependencies = (queue_config, analysis_config)
        return self

    def set_inputs(self, inputs: InputMap) -> "_EnsembleBuilder":
        self._inputs = inputs.to_dict()
        return self

    def set_outputs(self, outputs: OutputMap) -> "_EnsembleBuilder":
        self._outputs = outputs.to_dict()
        return self

    def set_custom_port_range(self, custom_port_range: range) -> "_EnsembleBuilder":
        self._custom_port_range = custom_port_range
        return self

    def set_max_running(self, max_running: int) -> "_EnsembleBuilder":
        self._max_running = max_running
        return self

    def set_max_retries(self, max_retries: int) -> "_EnsembleBuilder":
        self._max_retries = max_retries
        return self

    def set_retry_delay(self, retry_delay: int) -> "_EnsembleBuilder":
        self._retry_delay = retry_delay
        return self

    def set_executor(self, executor: str) -> "_EnsembleBuilder":
        self._executor = executor
        return self

    @staticmethod
    def from_legacy(
        run_context: RunContext,
        forward_model: ForwardModel,
        queue_config: QueueConfig,
        analysis_config: AnalysisConfig,
        res_config: ResConfig,
    ) -> "_EnsembleBuilder":
        builder = _EnsembleBuilder().set_legacy_dependencies(
            queue_config,
            analysis_config,
        )

        num_cpu = res_config.queue_config.num_cpu
        if num_cpu == 0:
            num_cpu = res_config.ecl_config.num_cpu

        for iens, run_arg in enumerate(run_context):
            active = run_context.is_active(iens)
            real = _RealizationBuilder().set_iens(iens).active(active)
            step = _StepBuilder().set_id("0").set_dummy_io().set_name("legacy step")
            if active:
                real.active(True).add_step(step)
                for index in range(0, len(forward_model)):
                    ext_job = forward_model.iget_job(index)
                    step.add_job(
                        _LegacyJobBuilder()
                        .set_id(str(index))
                        .set_index(str(index))
                        .set_name(ext_job.name())
                        .set_ext_job(ext_job)
                    )
                step.set_max_runtime(
                    analysis_config.get_max_runtime()
                ).set_callback_arguments((run_arg, res_config)).set_done_callback(
                    _lib.model_callbacks.forward_model_ok
                ).set_exit_callback(
                    _lib.model_callbacks.forward_model_exit
                ).set_num_cpu(
                    num_cpu
                ).set_run_path(
                    run_arg.runpath
                ).set_job_script(
                    res_config.queue_config.job_script
                ).set_job_name(
                    run_arg.job_name
                ).set_run_arg(
                    run_arg
                )
            builder.add_realization(real)
        return builder

    def _build_io_maps(self, reals: List[_RealizationBuilder]) -> None:
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
                    if isinstance(output, _DummyIOBuilder):
                        continue
                    factory = output.transmitter_factory(iens)
                    assert output._name  # mypy
                    o_matrix[iens][output._name] = factory() if factory else None
                    o_names.add(output._name)
        for iens, real in enumerate(reals):
            for step in real._steps:
                for input_ in step._inputs:
                    if isinstance(input_, _DummyIOBuilder):
                        continue
                    if input_._name not in o_names:
                        factory = input_.transmitter_factory(iens)
                        assert input_._name  # mypy
                        i_matrix[iens][input_._name] = factory() if factory else None
        self._inputs = InputMap.from_dict(i_matrix).validate().to_dict()
        self._outputs = OutputMap.from_dict(o_matrix).validate().to_dict()

    def build(self) -> _Ensemble:
        if not (self._reals or self._forward_model):
            raise ValueError("Either forward model or realizations needs to be set")

        real_builders: List[_RealizationBuilder] = []
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

        reals = [builder.build() for builder in real_builders]

        if self._legacy_dependencies:
            return _LegacyEnsemble(reals, self._metadata, *self._legacy_dependencies)
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
            )
