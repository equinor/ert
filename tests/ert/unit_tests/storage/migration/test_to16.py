import os
from pathlib import Path

from ert.storage.migration.to16 import delete_mask_file, migrate_field_param


def test_that_field_migration_removes_mask_file(use_tmpdir: Path):
    original_field_param = {
        "COND": {
            "type": "field",
            "name": "COND",
            "forward_init": True,
            "update": True,
            "ertbox_params": {
                "nx": 10,
                "ny": 10,
                "nz": 1,
                "xlength": 10.0,
                "ylength": 10.0,
                "xinc": 1.0,
                "yinc": 1.0,
                "rotation_angle": 0.0,
                "origin": [6.123233998228043e-16, 10.0],
            },
            "file_format": "bgrdecl",
            "mask_file": "grid_mask.npy",
            "output_transformation": None,
            "input_transformation": None,
            "truncation_min": None,
            "truncation_max": None,
            "forward_init_file": "cond.bgrdecl",
            "output_file": "cond.bgrdecl",
            "grid_file": "CASE.EGRID",
        }
    }
    Path("grid_mask.npy").write_bytes(os.urandom(100))  # Create a dummy mask file

    migrated_field_param = migrate_field_param(original_field_param)
    assert "mask_file" not in migrated_field_param["COND"]
    delete_mask_file(Path("."))  # Clean up the dummy mask file
    assert not Path("grid_mask.npy").exists()
