from ert.enum_shim import StrEnum


class ResponseDataInitialLayout(StrEnum):
    """
    Represents how data from a forward model run is organized initially within
    each realization folder.
    """

    ONE_FILE_PER_NAME = "ONE_FILE_PER_NAME"  # ex: gen data files
    ONE_FILE_WITH_ALL_NAMES = "ONE_FILE_WITH_ALL_NAMES"  # ex: summary files


class ResponseTypes(StrEnum):
    """
    Represents response types internal to ert.
    """

    GEN_DATA = "GEN_DATA"
    SUMMARY = "SUMMARY"
