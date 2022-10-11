import sys
from typing import TYPE_CHECKING

from cwrap import BaseCClass

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.data.ext_param import ExtParam
from ert._c_wrappers.enkf.data.field import Field
from ert._c_wrappers.enkf.data.gen_data import GenData
from ert._c_wrappers.enkf.data.gen_kw import GenKw
from ert._c_wrappers.enkf.data.summary import Summary
from ert._c_wrappers.enkf.enkf_fs import EnkfFs
from ert._c_wrappers.enkf.enums import ErtImplType
from ert._c_wrappers.enkf.node_id import NodeId

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.config import EnkfConfigNode


class EnkfNode(BaseCClass):
    TYPE_NAME = "enkf_node"
    _alloc = ResPrototype("void* enkf_node_alloc(enkf_config_node)", bind=False)
    _free = ResPrototype("void  enkf_node_free(enkf_node)")
    _get_name = ResPrototype("char* enkf_node_get_key(enkf_node)")
    _value_ptr = ResPrototype("void* enkf_node_value_ptr(enkf_node)")
    _try_load = ResPrototype("bool  enkf_node_try_load(enkf_node, enkf_fs, node_id)")
    _store = ResPrototype("bool  enkf_node_store(enkf_node, enkf_fs, node_id)")
    _has_data = ResPrototype("bool  enkf_node_has_data(enkf_node, enkf_fs, node_id)")
    _get_impl_type = ResPrototype(
        "ert_impl_type_enum enkf_node_get_impl_type(enkf_node)"
    )

    def __init__(self, config_node: "EnkfConfigNode"):
        c_pointer = self._alloc(config_node)

        if c_pointer:
            super().__init__(c_pointer, config_node, True)
        else:
            raise ValueError("Unable to create EnkfNode from given config node.")

    @classmethod
    def exportMany(
        cls,
        config_node: "EnkfConfigNode",
        file_format: str,
        fs: EnkfFs,
        iens_list,
        report_step=0,
        file_type=None,
        arg=None,
    ) -> None:
        node = EnkfNode(config_node)
        for iens in iens_list:
            filename = file_format % iens
            node_id = NodeId(report_step, iens)
            if node.tryLoad(fs, node_id):
                if node.export(filename, file_type=file_type, arg=arg):
                    print(f"{config_node.getKey()}[{iens:03d}] -> {filename}")
            else:
                sys.stderr.write(
                    f"** ERROR: Could not load realisation:{iens} - export failed"
                )

    def export(self, filename, file_type=None, arg=None):
        impl_type = self.getImplType()
        if impl_type == ErtImplType.FIELD:
            field_node = self.asField()
            return field_node.export(filename, file_type=file_type, init_file=arg)
        else:
            raise NotImplementedError("The export method is only implemented for field")

    def has_data(self, fs: EnkfFs, node_id: NodeId) -> bool:
        return self._has_data(fs, node_id)

    def valuePointer(self):
        return self._value_ptr()

    def getImplType(self) -> ErtImplType:
        return self._get_impl_type()

    def asGenData(self) -> GenData:
        impl_type = self.getImplType()
        assert impl_type == ErtImplType.GEN_DATA

        return GenData.createCReference(self.valuePointer(), self)

    def asGenKw(self) -> GenKw:
        impl_type = self.getImplType()
        assert impl_type == ErtImplType.GEN_KW

        return GenKw.createCReference(self.valuePointer(), self)

    def asField(self) -> Field:
        impl_type = self.getImplType()
        assert impl_type == ErtImplType.FIELD

        return Field.createCReference(self.valuePointer(), self)

    def as_summary(self) -> Summary:
        impl_type = self.getImplType()
        assert impl_type == ErtImplType.SUMMARY

        return Summary.createCReference(self.valuePointer(), self)

    def as_ext_param(self) -> ExtParam:
        impl_type = self.getImplType()
        assert impl_type == ErtImplType.EXT_PARAM

        return ExtParam.createCReference(self.valuePointer(), self)

    def tryLoad(self, fs: EnkfFs, node_id: NodeId) -> bool:
        if not isinstance(fs, EnkfFs):
            raise TypeError(f"fs must be an EnkfFs, not {type(fs)}")
        if not isinstance(node_id, NodeId):
            raise TypeError(f"node_id must be a NodeId, not {type(node_id)}")

        return self._try_load(fs, node_id)

    def name(self) -> str:
        return self._get_name()

    def load(self, fs: EnkfFs, node_id: NodeId):
        if not self.tryLoad(fs, node_id):
            raise Exception(
                f"Could not load node: {self.name()} iens: {node_id.iens} "
                f"report: {node_id.report_step}"
            )

    def save(self, fs: EnkfFs, node_id: NodeId):
        assert isinstance(fs, EnkfFs)
        assert isinstance(node_id, NodeId)

        return self._store(fs, node_id)

    def free(self):
        self._free()

    def __repr__(self):
        return f'EnkfNode(name = "{self.name()}") {self._ad_str()}'

    def forward_init(self, run_path: str, iens: int) -> bool:
        return _clib.enkf_node.forward_init(self, run_path, iens)
