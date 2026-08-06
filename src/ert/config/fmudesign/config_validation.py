"""
Module for validation of config (typically read from Excel).
"""

import copy
import numbers
from typing import Any


def validate_configuration(
    config: dict[str, Any], verbosity: int = 0
) -> dict[str, Any]:
    """Main function for config validation.

    This function is responsible for:
        - Checking that required keys exist
        - Checking that values are set to valid types
        - Setting default values if keys are not set

    """
    config = copy.deepcopy(config)

    if config["designtype"] != "onebyone":
        raise ValueError(
            "Generation of DesignMatrix only implemented for type 'onebyone', "
            f"not {config['designtype']}"
        )

    if "repeats" not in config:
        raise LookupError('"repeats" must be specified in general input sheet')

    key = "correlation_iterations"
    if key not in config:
        if verbosity > 0:
            print(f"{key!r} not set in general input sheet. Setting to default 0.")
            print("  - When set to 0, Iman Conover is used to induce correlations.")
            print(
                "  - When set to a positive integer N, Iman Conover is followed by N iterations\n"  # noqa: E501
                "    of random permutations (swaps). This leads to results that are never worse, and often better.\n"  # noqa: E501
                "    It is especially useful for skewed distributions like lognormal and high dimensional problems."  # noqa: E501
            )
            print(
                f"  If desired correlation does not match observed, try setting {key!r}=999 or higher."  # noqa: E501
            )
        config[key] = 0
    else:
        try:
            config[key] = int(config[key])
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"{key!r} must be a non-negative integer, got: {config[key]}"
            ) from err

    key = "distribution_seed"
    if key not in config:
        raise ValueError(
            "You did not specify a value for 'distribution_seed', which is used to "
            "seed the random number generator that draws from distributions in Monte "
            "Carlo sensitivities.\n"
            "- Specify a number (e.g. a 6 digit integer) to seed the random number "
            "generator and obtain reproducible results.\n"
            "- Specify None if you do not want to seed the random number generator. "
            "Your analysis will not be reproducible."
        )
    if not (isinstance(config[key], numbers.Integral) or (config[key] is None)):
        raise ValueError(
            f"{key!r} must be a non-negative integer or None, got: {config[key]}"
        )

    # 'seeds' here is 'rms_seeds' in the input. It can be either:
    # - 'default' => gives seed numbers 1000, 1001, 1002, ...
    # - 'None'    => seed number not added
    # - a path to a file
    key = "seeds"
    if key not in config:
        msg = '"rms_seeds" must be specified in general input sheet\n'
        msg += ' - Set to "None", "default" or path to a file.'
        raise LookupError(msg)
    is_default = config[key] == "default"
    is_none = config[key] is None
    is_list = isinstance(config[key], list) and config[key]
    if not any([is_default, is_none, is_list]):
        msg = f"'rms_seeds' must be 'None', 'default' or a list, got: {config[key]}"
        raise ValueError(msg)

    return config
