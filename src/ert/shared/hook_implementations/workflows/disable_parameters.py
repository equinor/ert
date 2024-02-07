from ert import ErtScript


class DisableParametersUpdate(ErtScript):
    """The DISABLE_PARAMETERS workflow has been removed and replaced with setting an
    option directly on the parameter for example:
    GEN_KW MY_PARAMETER_NAME DISTRIBUTIONS FILE UPDATE:FALSE
    """

    def run(self, disable_parameters: str) -> None:
        pass
