import logging
import os
from typing import Dict, List, Optional, Union

from cwrap import BaseCClass
from ecl.ecl_util import EclFileEnum, get_file_type
from ecl.grid import EclGrid
from ecl.summary import EclSum
from ecl.util.util import StringList

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.config.rangestring import rangestring_to_list
from ert._c_wrappers.enkf.config import EnkfConfigNode
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.enums import EnkfVarType, ErtImplType, GenDataFileType

logger = logging.getLogger(__name__)


def _get_abs_path(file):
    if file is not None:
        file = os.path.realpath(file)
    return file


def _get_filename(file):
    if file is not None:
        file = os.path.basename(file)
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


class EnsembleConfig(BaseCClass):
    TYPE_NAME = "ens_config"
    _alloc_full = ResPrototype("void* ensemble_config_alloc_full(char*)", bind=False)
    _free = ResPrototype("void ensemble_config_free( ens_config )")
    _has_key = ResPrototype("bool ensemble_config_has_key( ens_config , char* )")
    _get_node = ResPrototype(
        "enkf_config_node_ref ensemble_config_get_node( ens_config , char*)"
    )
    _alloc_keylist = ResPrototype(
        "stringlist_obj ensemble_config_alloc_keylist( ens_config )"
    )
    _alloc_keylist_from_impl_type = ResPrototype(
        "stringlist_obj ensemble_config_alloc_keylist_from_impl_type(ens_config, \
                                                                     ert_impl_type_enum)"  # noqa
    )
    _add_node = ResPrototype(
        "void ensemble_config_add_node( ens_config , enkf_config_node )"
    )
    _add_summary_full = ResPrototype(
        "void ensemble_config_init_SUMMARY_full(ens_config, char*, ecl_sum)"
    )

    @staticmethod
    def _load_grid(grid_file: Optional[str]) -> Optional[EclGrid]:
        if grid_file is None:
            return None
        ecl_grid_file_types = [
            EclFileEnum.ECL_GRID_FILE,
            EclFileEnum.ECL_EGRID_FILE,
        ]
        if get_file_type(grid_file) not in ecl_grid_file_types:
            raise ValueError(f"grid file {grid_file} does not have expected type")
        return EclGrid.load_from_file(grid_file)

    @staticmethod
    def _load_refcase(refcase_file: Optional[str]) -> Optional[EclSum]:
        """
        If the user has not given a refcase_file it will be
        impossible to use wildcards when expanding summary variables.
        """
        if refcase_file is None:
            return None
        # defaults for loading refcase - necessary for using the function
        # exposed in python part of ecl
        refcase_load_args = {
            "load_case": refcase_file,
            "join_string": ":",
            "include_restart": True,
            "lazy_load": True,
            "file_options": 0,
        }
        return EclSum(**refcase_load_args)

    def __init__(
        self,
        grid_file: Optional[str] = None,
        ref_case_file: Optional[str] = None,
        tag_format: str = "<%s>",
        gen_data_list: Optional[List] = None,
        gen_kw_list: Optional[List] = None,
        surface_list: Optional[List] = None,
        summary_list: Optional[List] = None,
        schedule_file: Optional[List] = None,
        field_list=None,
    ):
        gen_kw_list = [] if gen_kw_list is None else gen_kw_list
        gen_data_list = [] if gen_data_list is None else gen_data_list
        surface_list = [] if surface_list is None else surface_list
        summary_list = [] if summary_list is None else summary_list
        schedule_file = [] if schedule_file is None else schedule_file
        field_list = [] if field_list is None else field_list

        self._grid_file = grid_file
        self._refcase_file = ref_case_file
        self.grid: Optional[EclGrid] = self._load_grid(grid_file)
        self.refcase: Optional[EclSum] = self._load_refcase(ref_case_file)
        self._gen_kw_tag_format = tag_format
        c_ptr = self._alloc_full(self._gen_kw_tag_format)

        if c_ptr is None:
            raise ValueError("Failed to construct EnsembleConfig instance")
        super().__init__(c_ptr)

        for gene_data in gen_data_list:
            node = self.gen_data_node(gene_data)
            if node is not None:
                self.addNode(node)

        for gen_kw in gen_kw_list:
            gen_kw_node = self.gen_kw_node(gen_kw, tag_format)
            self.addNode(gen_kw_node)

        for surface in surface_list:
            surface_node = self.get_surface_node(surface)
            self.addNode(surface_node)

        for key in summary_list:
            if isinstance(key, list):
                for kkey in key:
                    self.add_summary_full(kkey, self.refcase)
            else:
                self.add_summary_full(key, self.refcase)

        for field in field_list:
            field_node = self.get_field_node(field, self.grid)
            self.addNode(field_node)
        if schedule_file:
            schedule_file_node = self._get_schedule_file_node(
                schedule_file, self._gen_kw_tag_format
            )
            self.addNode(schedule_file_node)

    @staticmethod
    def gen_data_node(gen_data: List[str]) -> Optional[EnkfConfigNode]:
        options = _option_dict(gen_data, 1)
        name = gen_data[0]
        res_file = options.get(ConfigKeys.RESULT_FILE)
        input_format_str = options.get(ConfigKeys.INPUT_FORMAT)
        if input_format_str != "ASCII":
            logger.error("The only supported INPUT_FORMAT is ASCII")
            return None
        report_steps_str = options.get(ConfigKeys.REPORT_STEPS, "")
        report_steps = rangestring_to_list(report_steps_str)

        return EnkfConfigNode.create_gen_data_full(
            name,
            res_file,
            GenDataFileType.ASCII,
            report_steps,
        )

    @staticmethod
    def gen_kw_node(gen_kw: List[str], tag_format: str) -> EnkfConfigNode:
        options = _option_dict(gen_kw, 4)
        name = gen_kw[0]
        tmpl_path = _get_abs_path(gen_kw[1])
        out_file = _get_filename(_get_abs_path(gen_kw[2]))
        param_file_path = _get_abs_path(gen_kw[3])
        forward_init = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "FALSE"))
        init_files = _get_abs_path(options.get(ConfigKeys.INIT_FILES))
        return EnkfConfigNode.create_gen_kw(
            name,
            tmpl_path,
            out_file,
            param_file_path,
            forward_init,
            init_files,
            tag_format,
        )

    @staticmethod
    def get_surface_node(surface: List[str]) -> EnkfConfigNode:
        options = _option_dict(surface, 1)
        name = surface[0]
        init_file = options.get(ConfigKeys.INIT_FILES)
        out_file = options.get("OUTPUT_FILE")
        base_surface = options.get(ConfigKeys.BASE_SURFACE_KEY)
        forward_init = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "FALSE"))

        return EnkfConfigNode.create_surface(
            name,
            init_file,
            out_file,
            base_surface,
            forward_init,
        )

    @staticmethod
    def get_field_node(
        field: Union[dict, list], grid: Optional[EclGrid]
    ) -> EnkfConfigNode:
        name = field[0]
        var_type = field[1]
        out_file = field[2]
        enkf_infile = None
        options = _option_dict(field, 2)
        if var_type == ConfigKeys.GENERAL_KEY:
            enkf_infile = field[3]
        forward_init = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "FALSE"))
        init_transform = options.get(ConfigKeys.INIT_TRANSFORM)
        output_transform = options.get(ConfigKeys.OUTPUT_TRANSFORM)
        input_transform = options.get(ConfigKeys.INPUT_TRANSFORM)
        min_ = options.get(ConfigKeys.MIN_KEY)
        max_ = options.get(ConfigKeys.MAX_KEY)
        init_files = options.get(ConfigKeys.INIT_FILES)
        return EnkfConfigNode.create_field(
            name,
            var_type,
            grid,
            out_file,
            enkf_infile,
            forward_init,
            init_transform,
            output_transform,
            input_transform,
            min_,
            max_,
            init_files,
        )

    @staticmethod
    def _get_schedule_file_node(
        schedule_file: Union[dict, list], tag_format: str
    ) -> EnkfConfigNode:
        file_path = _get_abs_path(schedule_file[0])
        file_name = _get_filename(file_path)
        options = _option_dict(schedule_file, 1)
        parameter = options.get(ConfigKeys.PARAMETER_KEY)
        init_files = options.get(ConfigKeys.INIT_FILES)

        return EnkfConfigNode.create_gen_kw(
            ConfigKeys.PRED_KEY,
            file_path,
            file_name,
            parameter,
            False,
            init_files,
            tag_format,
        )

    @classmethod
    def from_dict(cls, config_dict):
        grid_file_path = _get_abs_path(config_dict.get(ConfigKeys.GRID))
        refcase_file_path = _get_abs_path(config_dict.get(ConfigKeys.REFCASE))
        tag_format = config_dict.get(ConfigKeys.GEN_KW_TAG_FORMAT, "<%s>")
        gen_data_list = config_dict.get(ConfigKeys.GEN_DATA, [])
        gen_kw_list = config_dict.get(ConfigKeys.GEN_KW, [])
        surface_list = config_dict.get(ConfigKeys.SURFACE_KEY, [])
        summary_list = config_dict.get(ConfigKeys.SUMMARY, [])
        schedule_file = config_dict.get(ConfigKeys.SCHEDULE_PREDICTION_FILE, [])
        field_list = config_dict.get(ConfigKeys.FIELD_KEY, [])

        ens_config = cls(
            grid_file=grid_file_path,
            ref_case_file=refcase_file_path,
            tag_format=tag_format,
            gen_data_list=gen_data_list,
            gen_kw_list=gen_kw_list,
            surface_list=surface_list,
            summary_list=summary_list,
            schedule_file=schedule_file,
            field_list=field_list,
        )

        return ens_config

    def check_forward_init_nodes(self) -> List[EnkfConfigNode]:
        forward_init_nodes = []
        for config_key in self.alloc_keylist():
            config_node = self[config_key]
            if config_node.getUseForwardInit():
                forward_init_nodes.append(config_node)

        return forward_init_nodes

    def _node_info(self, node: str) -> str:
        impl_type = ErtImplType.from_string(node)
        key_list = self.getKeylistFromImplType(impl_type)
        return f"{node}: " f"{[self.getNode(key) for key in key_list]}, "

    def __repr__(self):
        if not self._address():
            return "<EnsembleConfig()>"
        return (
            "EnsembleConfig(config_dict={"
            + self._node_info(ConfigKeys.GEN_KW)
            + self._node_info(ConfigKeys.GEN_DATA)
            + self._node_info(ConfigKeys.SURFACE_KEY)
            + self._node_info(ConfigKeys.SUMMARY)
            + self._node_info(ConfigKeys.FIELD_KEY)
            + f"{ConfigKeys.GRID}: {self._grid_file},"
            + f"{ConfigKeys.REFCASE}: {self._refcase_file}"
            + "}"
        )

    def __getitem__(self, key: str) -> EnkfConfigNode:
        if key in self:
            return self._get_node(key).setParent(self)
        else:
            raise KeyError(f"The key:{key} is not in the ensemble configuration")

    def getNode(self, key: str) -> EnkfConfigNode:
        return self[key]

    def alloc_keylist(self) -> StringList:
        return self._alloc_keylist()

    def add_summary_full(self, key, refcase) -> EnkfConfigNode:
        return self._add_summary_full(key, refcase)

    def addNode(self, config_node: EnkfConfigNode):
        assert isinstance(config_node, EnkfConfigNode)
        assert config_node is not None
        key = config_node.getKey()
        if key in self:
            raise ConfigValidationError(
                f"Enkf config node with key {key} already present in ensemble config"
            )
        self._add_node(config_node)
        config_node.convertToCReference(self)

    def getKeylistFromVarType(self, var_mask: EnkfVarType) -> List[str]:
        assert isinstance(var_mask, EnkfVarType)
        return _clib.ensemble_config.ensemble_config_keylist_from_var_type(
            self, int(var_mask)
        )

    def getKeylistFromImplType(self, ert_impl_type) -> List[str]:
        assert isinstance(ert_impl_type, ErtImplType)
        return list(self._alloc_keylist_from_impl_type(ert_impl_type))

    @property
    def grid_file(self) -> Optional[str]:
        return self._grid_file

    @property
    def get_refcase_file(self) -> Optional[str]:
        return self._refcase_file

    @property
    def parameters(self) -> List[str]:
        return self.getKeylistFromVarType(EnkfVarType.PARAMETER)

    def __contains__(self, key):
        return self._has_key(key)

    def free(self):
        self._free()

    def __eq__(self, other):
        self_param_list = set(self.alloc_keylist())
        other_param_list = set(other.alloc_keylist())
        if self_param_list != other_param_list:
            return False

        for a in self_param_list:
            if a in self and a in other:
                if self.getNode(a) != other.getNode(a):
                    return False
            else:
                return False

        if (
            self._grid_file != other._grid_file
            or self._refcase_file != other._refcase_file
        ):
            return False

        return True

    def __ne__(self, other):
        return not self == other

    def have_forward_init(self) -> bool:
        return _clib.ensemble_config.have_forward_init(self)
