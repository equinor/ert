from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    no_type_check,
    overload,
)

import xtgeo
from ecl.summary import EclSum
from sortedcontainers import SortedList

from ert.field_utils import Shape, get_shape
from ert.validation import rangestring_to_list

from .field import TRANSFORM_FUNCTIONS, Field
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
    WarningInfo,
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


def _option_dict(option_list: Sequence[str], offset: int) -> Dict[str, str]:
    """Gets the list of options given to a keywords such as GEN_DATA.

    The first step of parsing will separate a line such as

      GEN_DATA NAME INPUT_FORMAT:ASCII RESULT_FILE:file.txt REPORT_STEPS:3

    into

    >>> opts = ["NAME", "INPUT_FORMAT:ASCII", "RESULT_FILE:file.txt", "REPORT_STEPS:3"]

    From there, _option_dict can be used to get a dictionary of the options:

    >>> _option_dict(opts, 1)
    {'INPUT_FORMAT': 'ASCII', 'RESULT_FILE': 'file.txt', 'REPORT_STEPS': '3'}

    Errors are reported to the log, and erroring fields ignored:

    >>> import sys
    >>> logger.addHandler(logging.StreamHandler(sys.stdout))
    >>> _option_dict(opts + [":T"], 1)
    Ignoring argument :T not properly formatted should be of type ARG:VAL
    {'INPUT_FORMAT': 'ASCII', 'RESULT_FILE': 'file.txt', 'REPORT_STEPS': '3'}

    """
    option_dict = {}
    for option_pair in option_list[offset:]:
        if not isinstance(option_pair, str):
            logger.warning(
                f"Ignoring unsupported option pair{option_pair} "
                f"of type {type(option_pair)}"
            )
            continue

        if len(option_pair.split(":")) == 2:
            key, val = option_pair.split(":")
            if val != "" and key != "":
                option_dict[key] = val
            else:
                logger.warning(
                    f"Ignoring argument {option_pair}"
                    " not properly formatted should be of type ARG:VAL"
                )
    return option_dict


def _str_to_bool(txt: str) -> bool:
    """This function converts text to boolean values according to the rules of
    the FORWARD_INIT keyword.

    The rules for str_to_bool is keep for backwards compatability

    First, any upper/lower case true/false value is converted to the corresponding
    boolean value:

    >>> _str_to_bool("TRUE")
    True
    >>> _str_to_bool("true")
    True
    >>> _str_to_bool("True")
    True
    >>> _str_to_bool("FALSE")
    False
    >>> _str_to_bool("false")
    False
    >>> _str_to_bool("False")
    False

    Any text which is not correctly identified as true or false returns False, but
    with a failure message written to the log:

    >>> _str_to_bool("fail")
    Failed to parse fail as bool! Using FORWARD_INIT:FALSE
    False
    """
    if txt.lower() == "true":
        return True
    elif txt.lower() == "false":
        return False
    else:
        logger.error(f"Failed to parse {txt} as bool! Using FORWARD_INIT:FALSE")
        return False


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

    def __init__(  # pylint: disable=too-many-arguments, too-many-branches
        self,
        grid_file: Optional[str] = None,
        ref_case_file: Optional[str] = None,
        gen_data_list: Optional[List[List[str]]] = None,
        gen_kw_list: Optional[List[List[str]]] = None,
        surface_list: Optional[List[List[str]]] = None,
        summary_list: Optional[List[List[str]]] = None,
        field_list: Optional[List[List[str]]] = None,
        ecl_base: Optional[str] = None,
    ) -> None:
        _gen_kw_list = [] if gen_kw_list is None else gen_kw_list
        _gen_data_list = [] if gen_data_list is None else gen_data_list
        _surface_list = [] if surface_list is None else surface_list
        _field_list = [] if field_list is None else field_list
        _summary_list = [] if summary_list is None else summary_list

        self._grid_file = grid_file
        self._refcase_file = ref_case_file
        self.refcase: Optional[EclSum] = self._load_refcase(ref_case_file)
        self.parameter_configs: Dict[str, ParameterConfig] = {}
        self.response_configs: Dict[str, ResponseConfig] = {}

        for gene_data in _gen_data_list:
            self.addNode(self.gen_data_node(gene_data))

        for gen_kw in _gen_kw_list:
            gen_kw_key = gen_kw[0]

            if gen_kw_key == "PRED":
                warnings.warn(
                    ConfigWarning(
                        WarningInfo(
                            "GEN_KW PRED used to hold a special meaning and be "
                            "excluded from being updated.\n If the intention was "
                            "to exclude this from updates, please use the "
                            "DisableParametersUpdate workflow though the "
                            "DISABLE_PARAMETERS key instead.\n fRef. GEN_KW "
                            "{gen_kw[0]} {gen_kw[1]} {gen_kw[2]} {gen_kw[3]}"
                        ).set_context(gen_kw[0])
                    ),
                    category=ConfigWarning,
                )

            options = _option_dict(gen_kw, 4)
            forward_init = _str_to_bool(options.get("FORWARD_INIT", "FALSE"))
            init_file = options.get("INIT_FILES")
            if init_file is not None:
                init_file = os.path.abspath(init_file)

            parameter_file = _get_abs_path(gen_kw[3])
            if not os.path.isfile(parameter_file):
                raise ConfigValidationError(f"No such parameter file: {parameter_file}")

            template_file = _get_abs_path(gen_kw[1])
            if not os.path.isfile(template_file):
                raise ConfigValidationError(f"No such template file: {template_file}")

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
                output_file=gen_kw[2],
                forward_init_file=init_file,
                transfer_function_definitions=transfer_function_definitions,
            )

            self._check_config_node(kw_node, gen_kw[3])
            self.addNode(kw_node)

        for surface in _surface_list:
            self.addNode(self.get_surface_node(surface))

        if ecl_base:
            ecl_base = ecl_base.replace("%d", "<IENS>")
            summary_keys = [item for sublist in _summary_list for item in sublist]
            self.add_summary_full(ecl_base, summary_keys, self.refcase)

        for field in _field_list:
            if self.grid_file is None:
                raise ConfigValidationError(
                    "In order to use the FIELD keyword, a GRID must be supplied."
                )
            dims = get_shape(self.grid_file)
            if dims is None:
                raise ConfigValidationError(
                    f"Grid file {self.grid_file} did not contain dimensions"
                )
            self.addNode(self.get_field_node(field, self.grid_file, dims))

    @staticmethod
    def gen_data_node(gen_data: List[str]) -> GenDataConfig:
        options = _option_dict(gen_data, 1)
        name = gen_data[0]
        res_file = options.get("RESULT_FILE")

        if res_file is None:
            raise ConfigValidationError(
                f"Missing or unsupported RESULT_FILE for GEN_DATA key {name!r}"
            )

        report_steps = rangestring_to_list(options.get("REPORT_STEPS", ""))

        if os.path.isabs(res_file) or "%d" not in res_file:
            result_file_context = next(
                x for x in gen_data if x.startswith("RESULT_FILE:")
            )
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message=f"The RESULT_FILE:{res_file} setting for {name} is "
                    f"invalid - must have an embedded %d and be a relative path",
                ).set_context(result_file_context)
            )

        if not report_steps:
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message="The GEN_DATA keywords must have REPORT_STEPS:xxxx"
                    " defined. Several report steps separated with ',' "
                    "and ranges with '-' can be listed",
                ).set_context_keyword(gen_data)
            )

        gdc = GenDataConfig(
            name=name, input_file=res_file, report_steps=SortedList(report_steps)
        )
        return gdc

    @staticmethod
    def get_surface_node(surface: List[str]) -> SurfaceConfig:
        options = _option_dict(surface, 1)
        name = surface[0]
        init_file = options.get("INIT_FILES")
        out_file = options.get("OUTPUT_FILE")
        base_surface = options.get("BASE_SURFACE")
        forward_init = _str_to_bool(options.get("FORWARD_INIT", "FALSE"))
        errors = []
        if not out_file:
            errors.append("Missing required OUTPUT_FILE")
        if not init_file:
            errors.append("Missing required INIT_FILES")
        elif not forward_init and "%d" not in init_file:
            errors.append("Must give file name with %d with FORWARD_INIT:FALSE")
        if not base_surface:
            errors.append("Missing required BASE_SURFACE")
        elif not Path(base_surface).exists():
            errors.append(f"BASE_SURFACE:{base_surface} not found")
        if errors:
            raise ConfigValidationError(
                f"SURFACE {name} incorrectly configured: {';'.join(errors)}"
            )
        surf = xtgeo.surface_from_file(base_surface, fformat="irap_ascii")
        return SurfaceConfig(
            ncol=surf.ncol,
            nrow=surf.nrow,
            xori=surf.xori,
            yori=surf.yori,
            xinc=surf.xinc,
            yinc=surf.yinc,
            rotation=surf.rotation,
            yflip=surf.yflip,
            name=name,
            forward_init=forward_init,
            forward_init_file=init_file,  # type: ignore
            output_file=Path(out_file),  # type: ignore
            base_surface_path=base_surface,  # type: ignore
        )

    @staticmethod
    def get_field_node(
        field: Sequence[str], grid_file: str, dimensions: Shape
    ) -> Field:
        name = field[0]
        out_file = Path(field[2])
        options = _option_dict(field, 2)
        init_transform = options.get("INIT_TRANSFORM")
        forward_init = _str_to_bool(options.get("FORWARD_INIT", "FALSE"))
        output_transform = options.get("OUTPUT_TRANSFORM")
        input_transform = options.get("INPUT_TRANSFORM")
        min_ = options.get("MIN")
        max_ = options.get("MAX")
        init_files = options.get("INIT_FILES")

        if input_transform:
            warnings.warn(
                ConfigWarning(
                    WarningInfo(
                        f"Got INPUT_TRANSFORM for FIELD: {name}, "
                        f"this has no effect and can be removed"
                    ).set_context(name)
                ),
                category=ConfigWarning,
            )
        if init_transform and init_transform not in TRANSFORM_FUNCTIONS:
            raise ConfigValidationError(
                f"FIELD INIT_TRANSFORM:{init_transform} is an invalid function"
            )
        if output_transform and output_transform not in TRANSFORM_FUNCTIONS:
            raise ConfigValidationError(
                f"FIELD OUTPUT_TRANSFORM:{output_transform} is an invalid function"
            )

        if init_files is None:
            raise ConfigValidationError("Missing required INIT_FILES")

        return Field(
            name=name,
            nx=dimensions.nx,
            ny=dimensions.ny,
            nz=dimensions.nz,
            file_format=out_file.suffix[1:],
            output_transformation=output_transform,
            input_transformation=init_transform,
            truncation_max=float(max_) if max_ is not None else None,
            truncation_min=float(min_) if min_ is not None else None,
            forward_init=forward_init,
            forward_init_file=init_files,
            output_file=out_file,
            grid_file=grid_file,
        )

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> EnsembleConfig:
        grid_file_path = _get_abs_path(config_dict.get(ConfigKeys.GRID))
        refcase_file_path = _get_abs_path(config_dict.get(ConfigKeys.REFCASE))
        gen_data_list = config_dict.get(ConfigKeys.GEN_DATA, [])
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE, [])
        summary_list = config_dict.get(ConfigKeys.SUMMARY, [])
        field_list = config_dict.get(ConfigKeys.FIELD, [])

        return cls(
            grid_file=grid_file_path,
            ref_case_file=refcase_file_path,
            gen_data_list=gen_data_list,
            gen_kw_list=gen_kw_list,
            surface_list=surface_list,
            summary_list=summary_list,
            field_list=field_list,
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
    def get_refcase_file(self) -> Optional[str]:
        return self._refcase_file

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
