from ert.enum_shim import StrEnum


class ResponseTypes(StrEnum):
    """
    Represents response types internal to ert.
    """

    gen_data = "gen_data"
    summary = "summary"
