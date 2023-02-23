import os

import pytest

from ert._c_wrappers.enkf import NodeId
from ert._c_wrappers.enkf.config import FieldTypeEnum
from ert._c_wrappers.enkf.data import EnkfNode


def test_field_type_enum(snake_oil_field_example):
    ert = snake_oil_field_example
    ens_config = ert.ensembleConfig()
    fc = ens_config["PERMX"].getFieldModelConfig()
    assert fc.get_type() == FieldTypeEnum.ECLIPSE_PARAMETER


def test_field_basics(snake_oil_field_example):
    ert = snake_oil_field_example
    ens_config = ert.ensembleConfig()
    fc = ens_config["PERMX"].getFieldModelConfig()
    grid = fc.get_grid()

    assert repr(fc).startswith("FieldConfig(type")
    assert (fc.get_nx(), fc.get_ny(), fc.get_nz()) == (10, 10, 5)
    assert (grid.getNX(), grid.getNY(), grid.getNZ()) == (10, 10, 5)
    assert fc.get_truncation_mode() == 0
    assert fc.get_truncation_min() == -1.0
    assert fc.get_truncation_max() == -1.0
    assert fc.get_init_transform_name() is None
    assert fc.get_output_transform_name() is None


def test_field_export(snake_oil_field_example):
    ert = snake_oil_field_example
    fs_manager = ert.storage_manager
    ens_config = ert.ensembleConfig()
    config_node = ens_config["PERMX"]
    data_node = EnkfNode(config_node)
    node_id = NodeId(0, 0)
    fs = fs_manager.current_case
    data_node.tryLoad(fs, node_id)

    data_node.export("export/with/path/PERMX.grdecl")
    assert os.path.isfile("export/with/path/PERMX.grdecl")


def test_field_export_many(snake_oil_field_example):
    ert = snake_oil_field_example
    fs_manager = ert.storage_manager
    ens_config = ert.ensembleConfig()
    config_node = ens_config["PERMX"]

    fs = fs_manager.current_case
    ert.sample_prior(fs, list(range(ert.getEnsembleSize())))
    # Filename without embedded %d - TypeError
    with pytest.raises(TypeError):
        EnkfNode.exportMany(config_node, "export/with/path/PERMX.grdecl", fs, [0, 2, 4])

    EnkfNode.exportMany(config_node, "export/with/path/PERMX_%d.grdecl", fs, [0, 2, 4])
    assert os.path.isfile("export/with/path/PERMX_0.grdecl")
    assert os.path.isfile("export/with/path/PERMX_2.grdecl")
    assert os.path.isfile("export/with/path/PERMX_4.grdecl")
