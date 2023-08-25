from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, no_type_check, overload

from ecl.summary import EclSum
from sortedcontainers import SortedList

from ert.field_utils import get_shape
from ert.validation import rangestring_to_list

from ._option_dict import option_dict
from ._str_to_bool import str_to_bool
from .field import Field
from .gen_data_config import GenDataConfig
from .gen_kw_config import GenKwConfig, PriorDict
from .parameter_config import ParameterConfig
from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
    ErrorInfo,
    MaybeWithContext,
)
from .response_config import ResponseConfig
from .summary_config import SummaryConfig
from .surface_config import SurfaceConfig

logger = logging.getLogger(__name__)


@overload
def _get_abs_path(file: None) -> None:
    pass


@overload
def _get_abs_path(file: str) -> str:
    pass


def _get_abs_path(file: Optional[str]) -> Optional[str]:
    if file is not None:
        file = os.path.realpath(file)
    return file


class EnsembleConfig:
    @staticmethod
    def _load_refcase(refcase_file: Optional[str]) -> Optional[EclSum]:
        if refcase_file is None:
            return None

        refcase_filepath = Path(refcase_file)
        refcase_file = str(refcase_filepath.parent / refcase_filepath.stem)

        if not os.path.exists(refcase_file + ".UNSMRY"):
            raise ConfigValidationError(
                f"Cannot find UNSMRY file for refcase provided! {refcase_file}.UNSMRY"
            )

        if not os.path.exists(refcase_file + ".SMSPEC"):
            raise ConfigValidationError(
                f"Cannot find SMSPEC file for refcase provided! {refcase_file}.SMSPEC"
            )

        # defaults for loading refcase - necessary for using the function
        # exposed in python part of ecl
        refcase_load_args = {
            "load_case": refcase_file,
            "join_string": ":",
            "include_restart": True,
            "lazy_load": False,
            "file_options": 0,
        }
        return EclSum(**refcase_load_args)

    def __init__(  # noqa: 501 pylint: disable=too-many-arguments, too-many-branches, too-many-statements
        self,
        grid_file: Optional[str] = None,
        ref_case_file: Optional[str] = None,
        gen_data_list: Optional[List[List[str]]] = None,
        gen_kw_list: Optional[List[List[str]]] = None,
        surface_list: Optional[List[SurfaceConfig]] = None,
        summary_list: Optional[List[List[str]]] = None,
        field_list: Optional[List[Field]] = None,
        ecl_base: Optional[str] = None,
    ) -> None:
        _gen_kw_list = [] if gen_kw_list is None else gen_kw_list
        _gen_data_list = [] if gen_data_list is None else gen_data_list
        _surface_list = [] if surface_list is None else surface_list
        _field_list = [] if field_list is None else field_list
        _summary_list = [] if summary_list is None else summary_list

        self._validate_gen_kw_list(_gen_kw_list)

        self._grid_file = _get_abs_path(grid_file)
        self._refcase_file = _get_abs_path(ref_case_file)
        self.refcase: Optional[EclSum] = self._load_refcase(self._refcase_file)
        self.parameter_configs: Dict[str, ParameterConfig] = {}
        self.response_configs: Dict[str, ResponseConfig] = {}

        for gene_data in _gen_data_list:
            self.addNode(self.gen_data_node(gene_data))

        for gen_kw in _gen_kw_list:
            gen_kw_key = gen_kw[0]

            if gen_kw_key == "PRED":
                warnings.warn(
                    ConfigWarning.with_context(
                        "GEN_KW PRED used to hold a special meaning and be "
                        "excluded from being updated.\n If the intention was "
                        "to exclude this from updates, please use the "
                        "DisableParametersUpdate workflow though the "
                        "DISABLE_PARAMETERS key instead.\n",
                        gen_kw[0],
                    ),
                )

            options = option_dict(gen_kw, 4)
            forward_init = str_to_bool(options.get("FORWARD_INIT", "FALSE"))
            init_file = _get_abs_path(options.get("INIT_FILES"))

            if len(gen_kw) == 2:
                parameter_file = _get_abs_path(gen_kw[1])
                template_file = None
                output_file = None
            else:
                output_file = gen_kw[2]
                parameter_file = _get_abs_path(gen_kw[3])

                template_file = _get_abs_path(gen_kw[1])
                if not os.path.isfile(template_file):
                    raise ConfigValidationError.with_context(
                        f"No such template file: {template_file}", gen_kw[1]
                    )
            if not os.path.isfile(parameter_file):
                raise ConfigValidationError.with_context(
                    f"No such parameter file: {parameter_file}", gen_kw[3]
                )

            transfer_function_definitions: List[str] = []
            with open(parameter_file, "r", encoding="utf-8") as file:
                for item in file:
                    item = item.rsplit("--")[0]  # remove comments
                    if item.strip():  # only lines with content
                        transfer_function_definitions.append(item)

            kw_node = GenKwConfig(
                name=gen_kw_key,
                forward_init=forward_init,
                template_file=template_file,
                output_file=output_file,
                forward_init_file=init_file,
                transfer_function_definitions=transfer_function_definitions,
            )

            self._check_config_node(kw_node, parameter_file)
            self.addNode(kw_node)

        for surface in _surface_list:
            self.addNode(surface)

        if ecl_base:
            ecl_base = ecl_base.replace("%d", "<IENS>")
            summary_keys = [item for sublist in _summary_list for item in sublist]
            self.add_summary_full(ecl_base, summary_keys, self.refcase)

        for field in _field_list:
            self.addNode(field)

    @classmethod
    def _validate_gen_kw_list(cls, gen_kw_list: List[List[str]]) -> None:
        errors = []

        def find_first_gen_kw_arg(kw_id: str, matching: str) -> Optional[str]:
            all_arglists = [arglist for arglist in gen_kw_list if arglist[0] == kw_id]

            # Example all_arglists:
            # [["SIGMA", "sigma.tmpl", "coarse.sigma", "sigma.dist"]]
            # It is expected to be of length 1
            if len(all_arglists) > 1:
                raise ConfigValidationError.with_context(
                    f"Found two GEN_KW {kw_id} declarations", kw_id
                )

            return next(
                (arg for arg in all_arglists[0] if matching.lower() in arg.lower()),
                None,
            )

        gen_kw_id_list = list({x[0] for x in gen_kw_list})

        for kw_id in gen_kw_id_list:
            use_fwd_init_token = find_first_gen_kw_arg(kw_id, "FORWARD_INIT:TRUE")

            if use_fwd_init_token is not None:
                errors.append(
                    ErrorInfo(
                        "Loading GEN_KW from files created by the forward "
                        "model is not supported.",
                    ).set_context(use_fwd_init_token)
                )

            init_files_token = find_first_gen_kw_arg(kw_id, "INIT_FILES:")

            if init_files_token is not None and "%" not in init_files_token:
                errors.append(
                    ErrorInfo(
                        "Loading GEN_KW from files requires %d in file format"
                    ).set_context(init_files_token)
                )
        if errors:
            raise ConfigValidationError.from_collected(errors)

    @staticmethod
    def gen_data_node(gen_data: List[str]) -> GenDataConfig:
        options = option_dict(gen_data, 1)
        name = gen_data[0]
        res_file = options.get("RESULT_FILE")

        if res_file is None:
            raise ConfigValidationError.with_context(
                f"Missing or unsupported RESULT_FILE for GEN_DATA key {name!r}", name
            )

        report_steps = rangestring_to_list(options.get("REPORT_STEPS", ""))
        report_steps = SortedList(report_steps) if report_steps else None
        if os.path.isabs(res_file):
            result_file_context = next(
                x for x in gen_data if x.startswith("RESULT_FILE:")
            )
            raise ConfigValidationError.with_context(
                f"The RESULT_FILE:{res_file} setting for {name} is "
                f"invalid - must be a relative path",
                result_file_context,
            )

        if report_steps is None and "%d" in res_file:
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message="RESULT_FILES using %d must have REPORT_STEPS:xxxx"
                    " defined. Several report steps separated with ',' "
                    "and ranges with '-' can be listed",
                ).set_context_keyword(gen_data)
            )

        if report_steps is not None and "%d" not in res_file:
            result_file_context = next(
                x for x in gen_data if x.startswith("RESULT_FILE:")
            )
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message=f"When configuring REPORT_STEPS:{report_steps} "
                    "RESULT_FILES must be configured using %d"
                ).set_context_keyword(result_file_context)
            )
        gdc = GenDataConfig(name=name, input_file=res_file, report_steps=report_steps)
        return gdc

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> EnsembleConfig:
        grid_file_path = config_dict.get(ConfigKeys.GRID)
        refcase_file_path = config_dict.get(ConfigKeys.REFCASE)
        gen_data_list = config_dict.get(ConfigKeys.GEN_DATA, [])
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE, [])
        summary_list = config_dict.get(ConfigKeys.SUMMARY, [])
        field_list = config_dict.get(ConfigKeys.FIELD, [])
        dims = None
        if grid_file_path is not None:
            dims = get_shape(grid_file_path)

        def make_field(field_list: List[str]) -> Field:
            if grid_file_path is None:
                raise ConfigValidationError.with_context(
                    "In order to use the FIELD keyword, a GRID must be supplied.",
                    field_list,
                )
            if dims is None:
                raise ConfigValidationError.with_context(
                    f"Grid file {grid_file_path} did not contain dimensions",
                    grid_file_path,
                )
            return Field.from_config_list(grid_file_path, dims, field_list)

        return cls(
            grid_file=grid_file_path,
            ref_case_file=refcase_file_path,
            gen_data_list=gen_data_list,
            gen_kw_list=gen_kw_list,
            surface_list=[SurfaceConfig.from_config_list(s) for s in surface_list],
            summary_list=summary_list,
            field_list=[make_field(f) for f in field_list],
            ecl_base=config_dict.get("ECLBASE"),
        )

    def _node_info(self, object_type: Type[Any]) -> str:
        key_list = self.getKeylistFromImplType(object_type)
        return f"{object_type}: " f"{[self[key] for key in key_list]}, "

    def __repr__(self) -> str:
        return (
            "EnsembleConfig(config_dict={"
            + self._node_info(GenDataConfig)
            + self._node_info(GenKwConfig)
            + self._node_info(SurfaceConfig)
            + self._node_info(SummaryConfig)
            + self._node_info(Field)
            + f"{ConfigKeys.GRID}: {self._grid_file},"
            + f"{ConfigKeys.REFCASE}: {self._refcase_file}"
            + "}"
        )

    def __getitem__(self, key: str) -> Union[ParameterConfig, ResponseConfig]:
        if key in self.parameter_configs:
            return self.parameter_configs[key]
        elif key in self.response_configs:
            return self.response_configs[key]
        else:
            raise KeyError(f"The key:{key} is not in the ensemble configuration")

    def getNodeGenData(self, key: str) -> GenDataConfig:
        gen_node = self.response_configs[key]
        assert isinstance(gen_node, GenDataConfig)
        return gen_node

    def hasNodeGenData(self, key: str) -> bool:
        return key in self.response_configs and isinstance(
            self.response_configs[key], GenDataConfig
        )

    def getNode(self, key: str) -> Union[ParameterConfig, ResponseConfig]:
        return self[key]

    def add_summary_full(
        self, ecl_base: str, key_list: List[str], refcase: Optional[EclSum]
    ) -> None:
        optional_keys = []
        for key in key_list:
            if "*" in key and refcase:
                optional_keys.extend(list(refcase.keys(pattern=key)))
            else:
                optional_keys.append(key)
        self.addNode(
            SummaryConfig(
                name="summary",
                input_file=ecl_base,
                keys=optional_keys,
                refcase=refcase,
            )
        )

    @staticmethod
    def _check_config_node(node: GenKwConfig, context: MaybeWithContext) -> None:
        errors = []

        def _check_non_negative_parameter(param: str, prior: PriorDict) -> None:
            key = prior["key"]
            dist = prior["function"]
            param_val = prior["parameters"][param]
            if param_val < 0:
                errors.append(
                    ErrorInfo(
                        f"Negative {param} {param_val!r}"
                        f" for {dist} distributed parameter {key!r}",
                    ).set_context(context)
                )

        for prior in node.get_priors():
            if prior["function"] == "LOGNORMAL":
                _check_non_negative_parameter("MEAN", prior)
                _check_non_negative_parameter("STD", prior)
            elif prior["function"] in ["NORMAL", "TRUNCATED_NORMAL"]:
                _check_non_negative_parameter("STD", prior)
        if errors:
            raise ConfigValidationError.from_collected(errors)

    def check_unique_node(self, key: str) -> None:
        if key in self:
            raise ConfigValidationError(
                f"Config node with key {key!r} already present in ensemble config"
            )

    def addNode(self, config_node: Union[ParameterConfig, ResponseConfig]) -> None:
        assert config_node is not None
        self.check_unique_node(config_node.name)
        if isinstance(config_node, ParameterConfig):
            self.parameter_configs[config_node.name] = config_node
        else:
            self.response_configs[config_node.name] = config_node

    def getKeylistFromImplType(self, node_type: Type[Any]) -> List[str]:
        mylist = []

        for key in self.keys:
            if isinstance(self[key], node_type):
                mylist.append(key)

        return mylist

    def get_keylist_gen_kw(self) -> List[str]:
        return self.getKeylistFromImplType(GenKwConfig)

    def get_keylist_gen_data(self) -> List[str]:
        return self.getKeylistFromImplType(GenDataConfig)

    @property
    def grid_file(self) -> Optional[str]:
        return self._grid_file

    @property
    def parameters(self) -> List[str]:
        return list(self.parameter_configs)

    @property
    def responses(self) -> List[str]:
        return list(self.response_configs)

    @property
    def keys(self) -> List[str]:
        return self.parameters + self.responses

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnsembleConfig):
            return False

        if self.keys != other.keys:
            return False

        for par in self.keys:
            if par in self and par in other:
                if self[par] != other[par]:
                    return False
            else:
                return False

        if (
            self._grid_file != other._grid_file
            or self._refcase_file != other._refcase_file
        ):
            return False

        return True

    def get_summary_keys(self) -> List[str]:
        if "summary" in self:
            summary = self["summary"]
            if isinstance(summary, SummaryConfig):
                return sorted(set(summary.keys))
        return []

    @property
    def parameter_configuration(self) -> List[ParameterConfig]:
        return list(self.parameter_configs.values())
