from cwrap import BaseCEnum


class ErtImplType(BaseCEnum):
    TYPE_NAME = "ert_impl_type_enum"
    FIELD = None
    GEN_KW = None
    SUMMARY = None
    GEN_DATA = None
    SURFACE = None
    EXT_PARAM = None


ErtImplType.addEnum("FIELD", 104)
ErtImplType.addEnum("GEN_KW", 107)
ErtImplType.addEnum("SUMMARY", 110)
ErtImplType.addEnum("GEN_DATA", 113)
ErtImplType.addEnum("SURFACE", 114)
ErtImplType.addEnum("EXT_PARAM", 116)
