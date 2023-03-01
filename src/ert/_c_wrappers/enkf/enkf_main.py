from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import xtgeo
from jinja2 import Template

from ert._c_wrappers.analysis.configuration import UpdateConfiguration
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
from ert._c_wrappers.enkf.enkf_obs import EnkfObs
from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.enkf.enums.enkf_var_type_enum import EnkfVarType
from ert._c_wrappers.enkf.enums.ert_impl_type_enum import ErtImplType
from ert._c_wrappers.enkf.ert_run_context import RunContext
from ert._c_wrappers.enkf.model_config import ModelConfig
from ert._c_wrappers.enkf.queue_config import QueueConfig
from ert._c_wrappers.enkf.runpaths import Runpaths
from ert._c_wrappers.util.substitution_list import SubstitutionList
from ert._clib import trans_func  # noqa: no_type_check

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert._c_wrappers.enkf import ErtConfig
    from ert._c_wrappers.enkf.config import GenKwConfig
    from ert.storage import EnsembleAccessor, EnsembleReader, StorageAccessor

logger = logging.getLogger(__name__)


def _backup_if_existing(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%SZ")
    new_path = path.parent / f"{path.name}_backup_{timestamp}"
    path.rename(new_path)


def _value_export_txt(
    run_path: str, export_base_name: str, values: Mapping[str, Mapping[str, float]]
) -> None:
    path = Path(run_path) / f"{export_base_name}.txt"
    _backup_if_existing(path)

    if len(values) == 0:
        return

    with path.open("w") as f:
        for key, param_map in values.items():
            for param, value in param_map.items():
                print(f"{key}:{param} {value:g}", file=f)


def _value_export_json(
    run_path: str, export_base_name: str, values: Mapping[str, Mapping[str, float]]
) -> None:
    path = Path(run_path) / f"{export_base_name}.json"
    _backup_if_existing(path)

    if len(values) == 0:
        return

    # Hierarchical
    json_out: Dict[str, Union[float, Dict[str, float]]] = {
        key: dict(param_map.items()) for key, param_map in values.items()
    }

    # Composite
    json_out.update(
        {
            f"{key}:{param}": value
            for key, param_map in values.items()
            for param, value in param_map.items()
        }
    )

    # Disallow NaN from being written: ERT produces the parameters and the only
    # way for the output to be NaN is if the input is invalid or if the sampling
    # function is buggy. Either way, that would be a bug and we can report it by
    # having json throw an error.
    json.dump(
        json_out, path.open("w"), allow_nan=False, indent=0, separators=(", ", " : ")
    )


def _generate_gen_kw_parameter_file(
    fs: EnsembleReader,
    realization: int,
    config: "GenKwConfig",
    target_file: str,
    run_path: Path,
    exports: Dict[str, Dict[str, float]],
) -> None:
    key = config.getKey()
    gen_kw_dict = fs.load_gen_kw_as_dict(key, realization)
    transformed = gen_kw_dict[key]
    if not len(transformed) == len(config):
        raise ValueError(
            f"The configuration of GEN_KW parameter {key}"
            f" is of size {len(config)}, expected {len(transformed)}"
        )

    with open(config.getTemplateFile(), "r", encoding="utf-8") as f:
        template = Template(
            f.read(), variable_start_string="<", variable_end_string=">"
        )

    if target_file.startswith("/"):
        target_file = target_file[1:]

    Path.mkdir(Path(run_path / target_file).parent, exist_ok=True, parents=True)
    with open(run_path / target_file, "w", encoding="utf-8") as f:
        f.write(
            template.render({key: f"{value:.6g}" for key, value in transformed.items()})
        )

    exports.update(gen_kw_dict)


def _generate_ext_parameter_file(
    fs: EnsembleReader,
    realization: int,
    key: str,
    target_file: str,
    run_path: Path,
) -> None:
    file_path = run_path / target_file
    Path.mkdir(file_path.parent, exist_ok=True, parents=True)
    data = fs.load_ext_param(key, realization)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _generate_surface_file(
    fs: EnsembleReader,
    realization: int,
    key: str,
    target_file: str,
    run_path: Path,
) -> None:
    file_path = run_path / target_file
    Path.mkdir(file_path.parent, exist_ok=True, parents=True)
    surf = fs.load_surface_file(key, realization)
    surf.to_file(run_path / target_file, fformat="irap_ascii")


def _generate_field_parameter_file(
    fs: EnsembleReader,
    realization: int,
    key: str,
    target_file: Path,
    run_path: Path,
) -> None:
    field_info = fs.load_field_info(key)
    file_format = field_info["file_format"]
    file_out = run_path.joinpath(target_file)
    if os.path.islink(file_out):
        os.unlink(file_out)
    fs.export_field(key, realization, file_out, file_format)


def _generate_parameter_files(
    ens_config: "EnsembleConfig",
    export_base_name: str,
    run_path: str,
    iens: int,
    fs: EnsembleReader,
) -> None:
    """
    Generate parameter files that are placed in each runtime directory for
    forward-model jobs to consume.

    Args:
        ens_config: Configuration which contains the parameter nodes for this
            ensemble run.
        export_base_name: Base name for the GEN_KW parameters file. Ie. the
            `parameters` in `parameters.json`.
        run_path: Path to the runtime directory
        iens: Realisation index
        fs: EnsembleReader from which to load parameter data
    """
    exports: Dict[str, Dict[str, float]] = {}
    for key in ens_config.getKeylistFromVarType(
        EnkfVarType.PARAMETER + EnkfVarType.EXT_PARAMETER  # type: ignore
    ):
        node = ens_config[key]
        type_ = node.getImplementationType()

        if type_ == ErtImplType.FIELD:
            if node.getUseForwardInit() and not fs.field_has_data(key, iens):
                continue
            _generate_field_parameter_file(
                fs,
                iens,
                key,
                Path(node.get_enkf_outfile()),
                Path(run_path),
            )
            continue

        if type_ == ErtImplType.GEN_KW:
            _generate_gen_kw_parameter_file(
                fs,
                iens,
                node.getKeywordModelConfig(),
                node.get_enkf_outfile(),
                Path(run_path),
                exports,
            )
            continue

        if type_ == ErtImplType.EXT_PARAM:
            _generate_ext_parameter_file(
                fs, iens, node.getKey(), node.get_enkf_outfile(), Path(run_path)
            )
            continue
        if type_ == ErtImplType.SURFACE:
            if node.getUseForwardInit() and not fs.has_surface(node.getKey(), iens):
                continue
            _generate_surface_file(
                fs, iens, node.getKey(), node.get_enkf_outfile(), Path(run_path)
            )
            continue

        raise NotImplementedError

    _value_export_txt(run_path, export_base_name, exports)
    _value_export_json(run_path, export_base_name, exports)


def field_transform(data: npt.ArrayLike, transform_name: str) -> Any:
    if not transform_name:
        return data

    def f(x: float) -> float:  # pylint: disable=too-many-return-statements
        if transform_name in ("LN", "LOG"):
            return math.log(x, math.e)
        if transform_name == "LN0":
            return math.log(x, math.e) + 0.000001
        if transform_name == "LOG10":
            return math.log(x, 10)
        if transform_name == "EXP":
            return math.exp(x)
        if transform_name == "EXP0":
            return math.exp(x) + 0.000001
        if transform_name == "POW10":
            return math.pow(x, 10)
        if transform_name == "TRUNC_POW10":
            return math.pow(max(x, 0.001), 10)
        return x

    vfunc = np.vectorize(f)

    return vfunc(data)


class ObservationConfigError(ConfigValidationError):
    def __init__(self, errors: str, config_file: Optional[str] = None) -> None:
        self.config_file = config_file
        self.errors = errors
        super().__init__(
            (
                f"Parsing observations config file `{self.config_file}` "
                f"resulted in the errors: {self.errors}"
            )
            if self.config_file
            else f"{self.errors}"
        )


class EnKFMain:
    def __init__(self, config: "ErtConfig", read_only: bool = False) -> None:
        self.ert_config = config
        self._update_configuration: Optional[UpdateConfiguration] = None

        self._observations = EnkfObs(  # type: ignore
            config.model_config.history_source,
            config.model_config.time_map,
            config.ensemble_config.refcase,
            config.ensemble_config,
        )
        if config.model_config.obs_config_file:
            if (
                os.path.isfile(config.model_config.obs_config_file)
                and os.path.getsize(config.model_config.obs_config_file) == 0
            ):
                raise ObservationConfigError(
                    f"Empty observations file: "
                    f"{config.model_config.obs_config_file}"
                )

            if self._observations.error:
                raise ObservationConfigError(
                    f"Incorrect observations file: "
                    f"{config.model_config.obs_config_file}"
                    f": {self._observations.error}",
                    config_file=config.model_config.obs_config_file,
                )
            try:
                self._observations.load(
                    config.model_config.obs_config_file,
                    config.analysis_config.get_std_cutoff(),
                )
            except (ValueError, IndexError) as err:
                raise ObservationConfigError(
                    str(err),
                    config_file=config.model_config.obs_config_file,
                ) from err

        self._ensemble_size = self.ert_config.model_config.num_realizations
        self._runpaths = Runpaths(
            jobname_format=self.getModelConfig().jobname_format_string,
            runpath_format=self.getModelConfig().runpath_format_string,
            filename=self.getModelConfig().runpath_file,
            substitute=self.get_context().substitute_real_iter,
        )

        # Set up RNG
        config_seed = self.resConfig().random_seed
        if config_seed is None:
            seed_seq = np.random.SeedSequence()
            logger.info(
                "To repeat this experiment, "
                "add the following random seed to your config file:"
            )
            logger.info(f"RANDOM_SEED {seed_seq.entropy}")
        else:
            seed: Union[int, Sequence[int]]
            try:
                seed = int(config_seed)
            except ValueError:
                seed = [ord(x) for x in config_seed]
            seed_seq = np.random.SeedSequence(seed)
        self._global_seed = seed_seq
        self._shared_rng = np.random.default_rng(seed_seq)

    @property
    def update_configuration(self) -> UpdateConfiguration:
        if not self._update_configuration:
            global_update_step = [
                {
                    "name": "ALL_ACTIVE",
                    "observations": self._observation_keys,
                    "parameters": self._parameter_keys,
                }
            ]
            self._update_configuration = UpdateConfiguration(
                update_steps=global_update_step
            )
        return self._update_configuration

    @update_configuration.setter
    def update_configuration(self, user_config: Any) -> None:
        config = UpdateConfiguration(update_steps=user_config)
        config.context_validate(self._observation_keys, self._parameter_keys)
        self._update_configuration = config

    @property
    def _observation_keys(self) -> List[str]:
        return list(self._observations.getMatchingKeys("*"))

    @property
    def _parameter_keys(self) -> List[str]:
        return self.ensembleConfig().parameters

    @property
    def runpaths(self) -> Runpaths:
        return self._runpaths

    @property
    def runpath_list_filename(self) -> os.PathLike[str]:
        return self._runpaths.runpath_list_filename

    def getLocalConfig(self) -> "UpdateConfiguration":
        return self.update_configuration

    def loadFromForwardModel(
        self, realization: List[bool], iteration: int, fs: EnsembleAccessor
    ) -> int:
        """Returns the number of loaded realizations"""
        run_context = self.ensemble_context(fs, realization, iteration)
        nr_loaded = fs.load_from_run_path(
            self.getEnsembleSize(),
            self.ensembleConfig(),
            run_context.run_args,
            run_context.mask,
        )
        fs.sync()
        return nr_loaded

    def ensemble_context(
        self, case: EnsembleAccessor, active_realizations: List[bool], iteration: int
    ) -> RunContext:
        """This loads an existing case from storage
        and creates run information for that case"""
        self.addDataKW("<ERT-CASE>", case.name)
        self.addDataKW("<ERTCASE>", case.name)
        return RunContext(
            sim_fs=case,
            path_format=self.getModelConfig().jobname_format_string,
            format_string=self.getModelConfig().runpath_format_string,
            runpath_file=Path(self.getModelConfig().runpath_file),
            initial_mask=active_realizations,
            global_substitutions=dict(self.get_context()),
            iteration=iteration,
        )

    def write_runpath_list(
        self, iterations: List[int], realizations: List[int]
    ) -> None:
        self.runpaths.write_runpath_list(iterations, realizations)

    def get_queue_config(self) -> QueueConfig:
        return self.resConfig().queue_config

    def get_num_cpu(self) -> int:
        return self.ert_config.preferred_num_cpu()

    def __repr__(self) -> str:
        return f"EnKFMain(size: {self.getEnsembleSize()}, config: {self.ert_config})"

    def getEnsembleSize(self) -> int:
        return self._ensemble_size

    def ensembleConfig(self) -> EnsembleConfig:
        return self.resConfig().ensemble_config

    def analysisConfig(self) -> AnalysisConfig:
        return self.resConfig().analysis_config

    def getModelConfig(self) -> ModelConfig:
        return self.ert_config.model_config

    def resConfig(self) -> "ErtConfig":
        return self.ert_config

    def get_context(self) -> SubstitutionList:
        return self.ert_config.substitution_list

    def addDataKW(self, key: str, value: str) -> None:
        self.get_context().addItem(key, value)

    def getObservations(self) -> EnkfObs:
        return self._observations

    def have_observations(self) -> bool:
        return len(self._observations) > 0

    def getHistoryLength(self) -> int:
        return self.resConfig().model_config.get_history_num_steps()

    def sample_prior(
        self,
        ensemble: EnsembleAccessor,
        active_realizations: List[int],
        parameters: Optional[List[str]] = None,
    ) -> None:
        """This function is responsible for getting the prior into storage,
        in the case of GEN_KW we sample the data and store it, and if INIT_FILES
        are used without FORWARD_INIT we load files and store them. If FORWARD_INIT
        is set the state is set to INITIALIZED, but no parameters are saved to storage
        until after the forward model has completed.
        """
        # pylint: disable=too-many-nested-blocks
        # (this is a real code smell that we mute for now)
        if parameters is None:
            parameters = self._parameter_keys

        for parameter in parameters:
            config_node = self.ensembleConfig().getNode(parameter)
            if config_node.getUseForwardInit():
                continue
            impl_type = config_node.getImplementationType()
            if impl_type == ErtImplType.FIELD:
                for _, realization_nr in enumerate(active_realizations):
                    init_file = config_node.get_init_file_fmt()
                    if "%d" in init_file:
                        init_file = init_file % realization_nr
                    grid_file = self.ensembleConfig().grid_file
                    assert grid_file is not None
                    grid = xtgeo.grid_from_file(grid_file)
                    try:
                        props = xtgeo.gridproperty_from_file(
                            init_file, name=parameter, grid=grid
                        )
                    except PermissionError as err:
                        context_message = (
                            f"Failed to open init file for parameter {parameter!r}"
                        )
                        raise RuntimeError(context_message) from err

                    data = props.values1d.data
                    field_config = config_node.getFieldModelConfig()
                    trans = field_config.get_init_transform_name()
                    data_transformed = field_transform(data, trans)
                    if not ensemble.field_has_info(parameter):
                        ensemble.save_field_info(
                            parameter,
                            grid_file,
                            Path(config_node.get_enkf_outfile()).suffix[1:],
                            field_config.get_output_transform_name(),
                            field_config.get_truncation_mode(),
                            field_config.get_truncation_min(),
                            field_config.get_truncation_max(),
                            field_config.get_nx(),
                            field_config.get_ny(),
                            field_config.get_nz(),
                        )
                    ensemble.save_field_data(
                        parameter, realization_nr, data_transformed
                    )

            elif impl_type == ErtImplType.GEN_KW:
                gen_kw_config = config_node.getKeywordModelConfig()
                keys = list(gen_kw_config)
                if config_node.get_init_file_fmt():
                    logging.info(
                        f"Reading from init file {config_node.get_init_file_fmt()}"
                        + f" for {parameter}"
                    )
                    parameter_values = gen_kw_config.values_from_files(
                        active_realizations,
                        config_node.get_init_file_fmt(),
                        keys,
                    )
                else:
                    logging.info(f"Sampling parameter {parameter}")
                    parameter_values = gen_kw_config.sample_values(
                        parameter,
                        keys,
                        str(self._global_seed.entropy),
                        active_realizations,
                        self.getEnsembleSize(),
                    )

                ensemble.save_gen_kw(
                    parameter_name=parameter,
                    parameter_keys=keys,
                    parameter_transfer_functions=gen_kw_config.get_priors(),
                    realizations=active_realizations,
                    data=parameter_values,
                )
            elif impl_type == ErtImplType.SURFACE:
                for realization_nr in active_realizations:
                    init_file = config_node.get_init_file_fmt()
                    if "%d" in init_file:
                        init_file = init_file % realization_nr
                    ensemble.save_surface_file(
                        config_node.getKey(), realization_nr, init_file
                    )
            else:
                raise NotImplementedError(f"{impl_type} is not supported")
        for realization_nr in active_realizations:
            ensemble.update_realization_state(
                realization_nr,
                [
                    RealizationStateEnum.STATE_UNDEFINED,
                    RealizationStateEnum.STATE_LOAD_FAILURE,
                ],
                RealizationStateEnum.STATE_INITIALIZED,
            )

        ensemble.sync()

    def rng(self) -> np.random.Generator:
        "Will return the random number generator used for updates."
        return self._shared_rng

    def createRunPath(self, run_context: RunContext) -> None:
        for iens, run_arg in enumerate(run_context):
            if run_context.is_active(iens):
                os.makedirs(
                    run_arg.runpath,
                    exist_ok=True,
                )

                for source_file, target_file in self.ert_config.ert_templates:
                    target_file = run_context.substituter.substitute_real_iter(
                        target_file, run_arg.iens, run_context.iteration
                    )
                    result = run_context.substituter.substitute_real_iter(
                        Path(source_file).read_text("utf-8"),
                        run_arg.iens,
                        run_context.iteration,
                    )
                    target = Path(run_arg.runpath) / target_file
                    if not target.parent.exists():
                        os.makedirs(
                            target.parent,
                            exist_ok=True,
                        )
                    target.write_text(result)

                ert_config = self.resConfig()
                model_config = ert_config.model_config
                _generate_parameter_files(
                    ert_config.ensemble_config,
                    model_config.gen_kw_export_name,
                    run_arg.runpath,
                    run_arg.iens,
                    run_context.sim_fs,
                )

                with open(
                    Path(run_arg.runpath) / "jobs.json", mode="w", encoding="utf-8"
                ) as fptr:
                    forward_model_output = ert_config.forward_model_data_to_json(
                        ert_config.forward_model_list,
                        run_arg.get_run_id(),
                        run_arg.iens,
                        run_context.iteration,
                        run_context.substituter,
                        ert_config.env_vars,
                    )

                    json.dump(forward_model_output, fptr)

        run_context.runpaths.write_runpath_list(
            [run_context.iteration], run_context.active_realizations
        )

    def runWorkflows(
        self,
        runtime: int,
        storage: "Optional[StorageAccessor]" = None,
    ) -> None:
        for workflow in self.ert_config.hooked_workflows[runtime]:
            workflow.run(self, storage)


__all__ = ["trans_func"]
