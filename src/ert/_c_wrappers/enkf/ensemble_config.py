import logging
import os
from typing import Any, Dict, List, Optional, Union

from cwrap import BaseCClass
from ecl.ecl_util import EclFileEnum, get_file_type
from ecl.grid import EclGrid
from ecl.summary import EclSum
from ecl.util.util import StringList

from ert import _clib
from ert._c_wrappers import ResPrototype
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


def _option_dict(option_list: list, offset: int) -> Dict[str, str]:
    option_dict = {}
    for option_pair in option_list[offset:]:
        if len(option_pair.split(":")) == 2:
            key, val = option_pair.split(":")
            if val != "" and key != "":
                option_dict[key] = val
    return option_dict


def _str_to_bool(txt: str) -> bool:
    if txt.lower() in ["true", "1"]:
        return True
    elif txt.lower() in ["false", "0"]:
        return False
    else:
        logger.error(f"Failed to parse {txt} as bool! Using FORWARD_INIT:FALSE")
        return False


class EnsembleConfig(BaseCClass):
    TYPE_NAME = "ens_config"
    _alloc_full = ResPrototype("void* ensemble_config_alloc_full(char*)", bind=False)
    _free = ResPrototype("void ensemble_config_free( ens_config )")
    _has_key = ResPrototype("bool ensemble_config_has_key( ens_config , char* )")
    _size = ResPrototype("int ensemble_config_get_size( ens_config)")
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
    _get_trans_table = ResPrototype("void* ensemble_config_get_trans_table(ens_config)")
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

    def _load_refcase(self, refcase_file: Optional[str]) -> Optional[EclSum]:
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
        gen_data_list=None,
        gen_kw_list=None,
        surface_list=None,
        summary_list=None,
        schedule_file_list=None,
        field_list=None,
    ):
        if gen_kw_list is None:
            gen_kw_list = []
        if gen_data_list is None:
            gen_data_list = []
        if surface_list is None:
            surface_list = []
        if summary_list is None:
            summary_list = []
        if schedule_file_list is None:
            schedule_file_list = []
        if field_list is None:
            field_list = []

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
            field_node = self.get_field_node(field, self.grid, self._get_trans_table())
            self.addNode(field_node)

        for schedule_file in schedule_file_list:
            schedule_file_node = self._get_schedule_file_node(
                schedule_file, self._gen_kw_tag_format
            )
            self.addNode(schedule_file_node)

    @staticmethod
    def gen_data_node(gen_data: Union[dict, list]) -> Optional[EnkfConfigNode]:
        if isinstance(gen_data, dict):
            name = gen_data.get(ConfigKeys.NAME)
            res_file = gen_data.get(ConfigKeys.RESULT_FILE)
            input_format = gen_data.get(ConfigKeys.INPUT_FORMAT)
            report_steps = gen_data.get(ConfigKeys.REPORT_STEPS)
        else:
            options = _option_dict(gen_data, 1)
            name = gen_data[0]
            res_file = options.get(ConfigKeys.RESULT_FILE)
            input_format_str = options.get(ConfigKeys.INPUT_FORMAT)
            if input_format_str not in ["ASCII", "ASCII_TEMPLATE"]:
                return None
            input_format = GenDataFileType.from_string(input_format_str)
            report_steps_str = options.get(ConfigKeys.REPORT_STEPS, "")
            report_steps = rangestring_to_list(report_steps_str)

        return EnkfConfigNode.create_gen_data_full(
            name,
            res_file,
            input_format,
            report_steps,
        )

    @staticmethod
    def gen_kw_node(gen_kw: Union[dict, list], tag_format: str) -> EnkfConfigNode:
        if isinstance(gen_kw, dict):
            name = gen_kw.get(ConfigKeys.NAME)
            tmpl_path = _get_abs_path(gen_kw.get(ConfigKeys.TEMPLATE))
            out_file = _get_filename(gen_kw.get(ConfigKeys.OUT_FILE))
            param_file_path = _get_abs_path(gen_kw.get(ConfigKeys.PARAMETER_FILE))
            forward_init = gen_kw.get(ConfigKeys.FORWARD_INIT)
            init_files = _get_abs_path(gen_kw.get(ConfigKeys.INIT_FILES))
        else:
            options = _option_dict(gen_kw, 4)
            name = gen_kw[0]
            tmpl_path = _get_abs_path(gen_kw[1])
            out_file = _get_filename(_get_abs_path(gen_kw[2]))
            param_file_path = _get_abs_path(gen_kw[3])
            forward_init = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "0"))
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
    def get_surface_node(surface: Union[dict, list]) -> EnkfConfigNode:
        if isinstance(surface, dict):
            name = surface.get(ConfigKeys.NAME)
            init_file = surface.get(ConfigKeys.INIT_FILES)
            out_file = surface.get(ConfigKeys.OUT_FILE)
            base_surface = surface.get(ConfigKeys.BASE_SURFACE_KEY)
            forward_int = surface.get(ConfigKeys.FORWARD_INIT)
        else:
            options = _option_dict(surface, 1)
            name = surface[0]
            init_file = options.get(ConfigKeys.INIT_FILES)
            out_file = options.get("OUTPUT_FILE")
            base_surface = options.get(ConfigKeys.BASE_SURFACE_KEY)
            forward_int = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "0"))

        return EnkfConfigNode.create_surface(
            name,
            init_file,
            out_file,
            base_surface,
            forward_int,
        )

    @staticmethod
    def get_field_node(
        field: Union[dict, list], grid: Optional[EclGrid], trans_table: Any
    ) -> EnkfConfigNode:
        if isinstance(field, dict):
            name = field.get(ConfigKeys.NAME)
            var_type = field.get(ConfigKeys.VAR_TYPE)
            out_file = field.get(ConfigKeys.OUT_FILE)
            enkf_infile = field.get(ConfigKeys.ENKF_INFILE)
            forward_init = field.get(ConfigKeys.FORWARD_INIT)
            init_transform = field.get(ConfigKeys.INIT_TRANSFORM)
            output_transform = field.get(ConfigKeys.OUTPUT_TRANSFORM)
            input_transform = field.get(ConfigKeys.INPUT_TRANSFORM)
            min_ = field.get(ConfigKeys.MIN_KEY)
            max_ = field.get(ConfigKeys.MAX_KEY)
            init_files = field.get(ConfigKeys.INIT_FILES)
        else:
            name = field[0]
            var_type = field[1]
            out_file = field[2]
            enkf_infile = None
            options = _option_dict(field, 2)
            if var_type == ConfigKeys.GENERAL_KEY:
                enkf_infile = field[3]
            forward_init = _str_to_bool(options.get(ConfigKeys.FORWARD_INIT, "0"))
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
            trans_table,
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

    @classmethod
    def _get_schedule_file_node(
        cls, schedule_file: Union[dict, list, str], tag_format: str
    ) -> EnkfConfigNode:
        if isinstance(schedule_file, dict):
            file_path = schedule_file.get(ConfigKeys.TEMPLATE)
            file_name = _get_filename(file_path)
            parameter = schedule_file.get(ConfigKeys.PARAMETER_KEY)
            init_files = schedule_file.get(ConfigKeys.INIT_FILES)
        else:
            file_path = schedule_file[0]
            if isinstance(schedule_file, str):
                file_path = schedule_file
            file_name = _get_filename(file_path)
            options = _option_dict(file_path, 1)
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
        if isinstance(summary_list, str):
            summary_list = [summary_list]
        schedule_file_list = config_dict.get(ConfigKeys.SCHEDULE_PREDICTION_FILE, [])
        if isinstance(schedule_file_list, str):
            schedule_file_list = [[schedule_file_list]]
        field_list = config_dict.get(ConfigKeys.FIELD_KEY, [])

        ens_config = cls(
            grid_file=grid_file_path,
            ref_case_file=refcase_file_path,
            tag_format=tag_format,
            gen_data_list=gen_data_list,
            gen_kw_list=gen_kw_list,
            surface_list=surface_list,
            summary_list=summary_list,
            schedule_file_list=schedule_file_list,
            field_list=field_list,
        )

        return ens_config

    def __len__(self):
        return self._size()

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
