from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import xtgeo
from ecl.summary import EclSum

from ert import _clib
from ert._c_wrappers.enkf import GenKwConfig
from ert._c_wrappers.enkf.config.field_config import TRANSFORM_FUNCTIONS, Field
from ert._c_wrappers.enkf.config.gen_data_config import GenDataConfig
from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
from ert._c_wrappers.enkf.config.response_config import ResponseConfig
from ert._c_wrappers.enkf.config.summary_config import SummaryConfig
from ert._c_wrappers.enkf.config.surface_config import SurfaceConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert.parsing import ConfigValidationError, ConfigWarning, ErrorInfo
from ert.storage.field_utils.field_utils import Shape, get_shape
from ert.validation import rangestring_to_list

logger = logging.getLogger(__name__)


def _get_abs_path(file):
    if file is not None:
        file = os.path.realpath(file)
    return file


def _option_dict(option_list: List[str], offset: int) -> Dict[str, str]:
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

    def __init__(
        self,
        grid_file: Optional[str] = None,
        ref_case_file: Optional[str] = None,
        gen_data_list: Optional[List] = None,
        gen_kw_list: Optional[List] = None,
        surface_list: Optional[List] = None,
        summary_list: Optional[List] = None,
        field_list=None,
        ecl_base: Optional[str] = None,
    ):
        gen_kw_list = [] if gen_kw_list is None else gen_kw_list
        gen_data_list = [] if gen_data_list is None else gen_data_list
        surface_list = [] if surface_list is None else surface_list
        field_list = [] if field_list is None else field_list

        self._grid_file = grid_file
        self._refcase_file = ref_case_file
        self.refcase: Optional[EclSum] = self._load_refcase(ref_case_file)
        self.parameter_configs = {}
        self.response_configs = {}

        for gene_data in gen_data_list:
            self.addNode(self.gen_data_node(gene_data))

        for gen_kw in gen_kw_list:
            gen_kw_key = gen_kw[0]

            if gen_kw_key == "PRED":
                warnings.warn(
                    "GEN_KW PRED used to hold a special meaning and be excluded from "
                    "being updated.\n"
                    "If the intention was to exclude this from updates, please "
                    "use the DisableParametersUpdate workflow instead.\n"
                    f"Ref. GEN_KW {gen_kw[0]} {gen_kw[1]} {gen_kw[2]} {gen_kw[3]}",
                    category=ConfigWarning,
                )

            options = _option_dict(gen_kw, 4)
            forward_init = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "FALSE"))
            init_file = options.get(ConfigKeys.INIT_FILES, None)
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
                parameter_file=parameter_file,
                forward_init_file=init_file,
                transfer_function_definitions=transfer_function_definitions,
            )

            self._check_config_node(kw_node, parameter_file)
            self.addNode(kw_node)

        for surface in surface_list:
            self.addNode(self.get_surface_node(surface))

        if ecl_base:
            ecl_base = ecl_base.replace("%d", "<IENS>")
            summary_keys = [item for sublist in summary_list for item in sublist]
            self.add_summary_full(ecl_base, summary_keys, self.refcase)

        for field in field_list:
            if self.grid_file is None:
                raise ConfigValidationError(
                    "In order to use the FIELD keyword, a GRID must be supplied."
                )
            dims = get_shape(grid_file)
            self.addNode(self.get_field_node(field, grid_file, dims))

    @staticmethod
    def gen_data_node(gen_data: List[str]) -> Optional[GenDataConfig]:
        options = _option_dict(gen_data, 1)
        name = gen_data[0]
        res_file = options.get(ConfigKeys.RESULT_FILE)

        if res_file is None:
            raise ConfigValidationError(
                f"Missing or unsupported RESULT_FILE for GEN_DATA key {name!r}"
            )

        report_steps = rangestring_to_list(options.get(ConfigKeys.REPORT_STEPS, ""))

        if os.path.isabs(res_file) or "%d" not in res_file:
            logger.error(
                f"The RESULT_FILE:{res_file} setting for {name} is invalid - "
                "must have an embedded %d - and be a relative path"
            )
        elif not report_steps:
            logger.error(
                "The GEN_DATA keywords must have a REPORT_STEPS:xxxx defined"
                "Several report steps separated with ',' and ranges with '-'"
                "can be listed"
            )
        else:
            gdc = GenDataConfig(
                name=name, input_file=res_file, report_steps=report_steps
            )
            return gdc

    @staticmethod
    def get_surface_node(surface: List[str]) -> SurfaceConfig:
        options = _option_dict(surface, 1)
        name = surface[0]
        init_file = options.get(ConfigKeys.INIT_FILES)
        out_file = options.get("OUTPUT_FILE")
        base_surface = options.get(ConfigKeys.BASE_SURFACE_KEY)
        forward_init = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "FALSE"))
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
            errors = ";".join(errors)
            raise ConfigValidationError(
                f"SURFACE {name} incorrectly configured: {errors}"
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
            forward_init_file=init_file,
            output_file=Path(out_file),
            base_surface_path=base_surface,
        )

    @staticmethod
    def get_field_node(
        field: Union[dict, list], grid_file: str, dimensions: Shape
    ) -> Field:
        name = field[0]
        out_file = Path(field[2])
        options = _option_dict(field, 2)
        init_transform = options.get(ConfigKeys.INIT_TRANSFORM)
        forward_init = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "FALSE"))
        output_transform = options.get(ConfigKeys.OUTPUT_TRANSFORM)
        input_transform = options.get(ConfigKeys.INPUT_TRANSFORM)
        min_ = options.get(ConfigKeys.MIN_KEY)
        max_ = options.get(ConfigKeys.MAX_KEY)
        init_files = options.get(ConfigKeys.INIT_FILES)

        if input_transform:
            warnings.warn(
                f"Got INPUT_TRANSFORM for FIELD: {name}, "
                f"this has no effect and can be removed",
                category=ConfigWarning,
            )
        if init_transform and init_transform not in TRANSFORM_FUNCTIONS:
            raise ValueError(
                f"FIELD INIT_TRANSFORM:{init_transform} is an invalid function"
            )
        if output_transform and output_transform not in TRANSFORM_FUNCTIONS:
            raise ValueError(
                f"FIELD OUTPUT_TRANSFORM:{output_transform} is an invalid function"
            )

        if min_ is not None and not isinstance(min_, float):
            min_ = float(min_)
        if max_ is not None and not isinstance(max_, float):
            max_ = float(max_)
        return Field(
            name=name,
            nx=dimensions.nx,
            ny=dimensions.ny,
            nz=dimensions.nz,
            file_format=out_file.suffix[1:],
            output_transformation=output_transform,
            input_transformation=init_transform,
            truncation_max=max_,
            truncation_min=min_,
            forward_init=forward_init,
            forward_init_file=init_files,
            output_file=out_file,
            grid_file=grid_file,
        )

    @classmethod
    def from_dict(cls, config_dict) -> EnsembleConfig:
        grid_file_path = _get_abs_path(config_dict.get(ConfigKeys.GRID))
        refcase_file_path = _get_abs_path(config_dict.get(ConfigKeys.REFCASE))
        gen_data_list = config_dict.get(ConfigKeys.GEN_DATA, [])
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE_KEY, [])
        summary_list = config_dict.get(ConfigKeys.SUMMARY, [])
        field_list = config_dict.get(ConfigKeys.FIELD_KEY, [])

        ens_config = cls(
            grid_file=grid_file_path,
            ref_case_file=refcase_file_path,
            gen_data_list=gen_data_list,
            gen_kw_list=gen_kw_list,
            surface_list=surface_list,
            summary_list=summary_list,
            field_list=field_list,
            ecl_base=config_dict.get("ECLBASE"),
        )

        return ens_config

    def _node_info(self, object_type: object) -> str:
        key_list = self.getKeylistFromImplType(object_type)
        return f"{object_type}: " f"{[self[key] for key in key_list]}, "

    def __repr__(self):
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

    def getNode(
        self, key: str
    ) -> Union[ParameterConfig, EnsembleConfig,]:
        return self[key]

    def add_summary_full(self, ecl_base, key_list, refcase) -> SummaryConfig:
        optional_keys = []
        for key in key_list:
            optional_keys.extend(
                _clib.ensemble_config.get_summary_key_list(key, refcase)
                if refcase
                else key_list
            )
        self.addNode(
            SummaryConfig(
                name="summary",
                input_file=ecl_base,
                keys=optional_keys,
                refcase=refcase,
            )
        )

    @staticmethod
    def _check_config_node(node: GenKwConfig, parameter_file: str):
        errors = []

        def _check_non_negative_parameter(param: str):
            key = prior["key"]
            dist = prior["function"]
            param_val = prior["parameters"][param]
            if param_val < 0:
                errors.append(
                    f"Negative {param} {param_val!r}"
                    f" for {dist} distributed parameter {key!r}"
                )

        for prior in node.get_priors():
            if prior["function"] == "LOGNORMAL":
                _check_non_negative_parameter("MEAN")
                _check_non_negative_parameter("STD")
            elif prior["function"] in ["NORMAL", "TRUNCATED_NORMAL"]:
                _check_non_negative_parameter("STD")
        if errors:
            raise ConfigValidationError(
                config_file=parameter_file,
                errors=[
                    ErrorInfo(message=str(e), filename=parameter_file) for e in errors
                ],
            )

    def check_unique_node(self, key: str):
        if key in self:
            raise ConfigValidationError(
                f"Config node with key {key!r} already present in ensemble config"
            )

    def addNode(
        self,
        config_node: Union[
            ParameterConfig,
            ResponseConfig,
        ],
    ):
        assert config_node is not None
        self.check_unique_node(config_node.name)
        if isinstance(config_node, ParameterConfig):
            self.parameter_configs[config_node.name] = config_node
        else:
            self.response_configs[config_node.name] = config_node

    def getKeylistFromImplType(self, node_type: object):
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
    def keys(self):
        return self.parameters + self.responses

    def __contains__(self, key):
        return key in self.keys

    def __eq__(self, other: EnsembleConfig):
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
        if "summary" in self and isinstance(self["summary"], SummaryConfig):
            return sorted(set(self["summary"].keys))
        return []

    @property
    def parameter_configuration(self) -> List[ParameterConfig]:
        return list(self.parameter_configs.values())
