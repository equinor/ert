from ert import ErtScript


class DisableParametersUpdate(ErtScript):
    """The DISABLE_PARAMETERS workflow disables parameters,
    so they are excluded from the update step. The job takes a list
    of parameters as input:

    DISABLE_PARAMETERS "PARAMETER_1, PARAMETER_2"

    The parameters that are given as arguments will be removed from the
    update. Note that if giving more than one parameter as input the list
    must be enclosed in quotes.

    Note that unknown parameter names will be silently ignored.

    This workflow is recommended to be run as a PRE_FIRST_UPDATE hook
    """

    def run(self, disable_parameters):
        ert = self.ert()
        disable_parameters = disable_parameters.split(",")
        disable_parameters = [val.strip() for val in disable_parameters]
        altered_update_step = [
            {
                "name": "DISABLED_PARAMETERS",
                "observations": ert._observation_keys,
                "parameters": [
                    key for key in ert._parameter_keys if key not in disable_parameters
                ],
            }
        ]
        ert.update_configuration = altered_update_step
