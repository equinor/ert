# -*- coding: utf-8 -*-
import configsuite
import six

from copy import deepcopy

from configsuite import MetaKeys as MK
from configsuite import types


@configsuite.validator_msg("Minimum length of index list must be > 1 for PCA")
def _min_length(value):
    return len(value) > 1


@configsuite.validator_msg("Minimum value of index must be >= 0")
def _min_value(value):
    return value >= 0


_num_convert_msg = "Will go through the input and try to convert to list of int"
@configsuite.transformation_msg(_num_convert_msg)
def _to_int_list(value):
    value = deepcopy(value)
    if isinstance(value, six.integer_types):
        return [value]
    elif isinstance(value, (list, tuple)):
        value = ",".join([str(x) for x in value])
    return _realize_list(value)


@configsuite.transformation_msg("Convert ranges and singeltons into list")
def _realize_list(input_string):
    """If input_string is not a string, input_string will be returned. If input_string
    is a string it is assumed to contain comma separated elements. Each element is
    assumed to be either a singelton or a range. A singleton is a single number,
    i.e.  7 or 14. A range is a lower and upper element of the range separated by a
    single '-'. When provided with a string we will either return a list containing the
    union of all the singeltons and the ranges, or raise a TypeError or ValueError if
    it fails in the process. _realize_list('1,2,4-7,14-15') -> [1, 2, 4, 5, 6, 7, 14, 15]
    """
    real_list = []
    for elem in input_string.split(","):
        bounds = elem.split("-")
        if len(bounds) == 1:
            if "-" in elem:
                raise ValueError("Did not expect '-' in singleton")
            real_list.append(int(elem))
        elif len(bounds) == 2:
            if elem.count("-") != 1:
                raise ValueError("Did expect single '-' in range")
            lower_bound = int(bounds[0])
            upper_bound = int(bounds[1]) + 1

            if lower_bound > upper_bound:
                err_msg = "Lower bound of range expected to be smaller then upper bound"
                raise ValueError(err_msg)

            real_list += range(lower_bound, upper_bound)
        else:
            raise ValueError("Expected at most one '-' in an element")

    return real_list


_num_convert_msg = "Create UPDATE_KEYS from CALCULATE_KEYS as it was not specified"
@configsuite.transformation_msg(_num_convert_msg)
def _expand_input(input_value):
    expanded_values = deepcopy(input_value)
    if "CALCULATE_KEYS" in expanded_values and "UPDATE_KEYS" not in expanded_values:
        if "index" in expanded_values["CALCULATE_KEYS"]:
            expanded_values.update(
                {"UPDATE_KEYS": {
                     "keys":
                         [{"key": key, "index": expanded_values["CALCULATE_KEYS"]["index"]} for key in expanded_values["CALCULATE_KEYS"]["keys"]]}
                }
            )
        else:
            expanded_values.update(
                {"UPDATE_KEYS": {
                    "keys":
                        [{"key": key} for key in expanded_values["CALCULATE_KEYS"]["keys"]]}
                }
            )
    return expanded_values


@configsuite.validator_msg("Threshold must be higher than 0 and lower than 1")
def _min_max_value(value):
    return 0.0 < value < 1.0


def build_schema():
    return {
            MK.Type: types.NamedDict,
            MK.Description: "Keys and index lists from all scaled keys",
            MK.LayerTransformation: _expand_input,
            MK.Content: {
                "CALCULATE_KEYS": {
                    MK.Required: True,
                    MK.Type: types.NamedDict,
                    MK.Content: {
                        "keys": {
                            MK.Required: True,
                            MK.Type: types.List,
                            MK.Content: {
                                MK.Item: {
                                MK.Type: types.String,
                                },
                            },
                        },
                        "index": {
                            MK.Required: False,
                            MK.LayerTransformation: _to_int_list,
                            MK.Type: types.List,
                            MK.ElementValidators: (_min_length,),
                            MK.Content: {
                                MK.Item: {
                                    MK.Type: types.Integer,
                                    MK.ElementValidators: (_min_value,),
                                },
                            },
                        },
                        "threshold": {
                            MK.Required: False,
                            MK.Type: types.Number,
                            MK.ElementValidators: (_min_max_value,),
                        },
                        "std_cutoff": {
                            MK.Required: False,
                            MK.Type: types.Number,
                        },
                        "alpha": {
                            MK.Required: False,
                            MK.Type: types.Number,
                        },
                    },
                },
                "UPDATE_KEYS": {
                    MK.Required: False,
                    MK.Type: types.NamedDict,
                    MK.Content: {
                        "keys": {
                            MK.Required: False,
                            MK.Type: types.List,
                            MK.Content: {
                                MK.Item: {
                                    MK.Type: types.NamedDict,
                                    MK.Content: {
                                        "key": {
                                            MK.Required: True,
                                            MK.Type: types.String,
                                        },
                                        "index": {
                                            MK.Required: False,
                                            MK.Type: types.List,
                                            MK.LayerTransformation: _to_int_list,
                                            MK.Content: {
                                                MK.Item: {
                                                    MK.Type: types.Integer,
                                                    MK.ElementValidators: (_min_value,),
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }


def get_default_values():
    default_values = {
        "CALCULATE_KEYS": {
            "threshold": 0.95,
            "std_cutoff": 1e-6,
            "alpha": 3.0,
        },
        "UPDATE_KEYS": {}
    }
    return default_values
