from cwrap import BaseCEnum


class EnKFFSType(BaseCEnum):
    TYPE_NAME = "enkf_fs_type_enum"
    INVALID_DRIVER_ID = None
    BLOCK_FS_DRIVER_ID = None


EnKFFSType.addEnum("INVALID_DRIVER_ID", 0)
EnKFFSType.addEnum("BLOCK_FS_DRIVER_ID", 3001)
