#!/usr/bin/env python
import os
import sys
import re
from argparse import ArgumentParser, ArgumentTypeError
from ert_gui import run_cli
from ert_gui import ERT
from ert_gui.ide.keywords.definitions import (
    RangeStringArgument,
    ProperNameArgument,
    ProperNameFormatArgument,
    NumberListStringArgument,
)
from ert_gui.simulation.models.multiple_data_assimilation import (
    MultipleDataAssimilation,
)


def strip_error_message_and_raise_exception(validated):
    error = validated.message()
    error = re.sub(r"\<[^>]*\>", " ", error)
    raise ArgumentTypeError(error)


def valid_file(fname):
    if not os.path.isfile(fname):
        raise ArgumentTypeError("File was not found: {}".format(fname))
    return fname


def valid_realizations(user_input):
    validator = RangeStringArgument()
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_weights(user_input):
    validator = NumberListStringArgument()
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_name_format(user_input):
    validator = ProperNameFormatArgument()
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_name(user_input):
    validator = ProperNameArgument()
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_name_not_default(user_input):
    if user_input == "default":
        msg = (
            "Target file system and source file system can not be the same. "
            "They were both: <default>."
        )
        raise ArgumentTypeError(msg)
    valid_name(user_input)
    return user_input


def range_limited_int(user_input):
    try:
        i = int(user_input)
    except ValueError:
        raise ArgumentTypeError("Must be a int")
    if 0 < i < 100:
        return i
    raise ArgumentTypeError("Range must be in range 1 - 99")


def runGui(args):
    os.execvp("python", ["python"] + ["-m", "ert_gui.gert_main"] + [args.config])


def get_ert_parser(parser=None):
    if parser is None:
        parser = ArgumentParser(description="ERT - Ensemble Reservoir Tool")
    subparsers = parser.add_subparsers(
        title="Available user entries",
        description="ERT can be accessed through a GUI or CLI interface. Include "
        "one of the following arguments to change between the "
        "interfaces. Note that different sub commands may require "
        "different additional arguments. See the help section for "
        "each sub command for more details.",
        help="Available sub commands",
        dest="mode",
    )

    config_help = "Ert configuration file"

    # gui_parser
    gui_parser = subparsers.add_parser(
        "gui",
        description="Opens an independent graphical user interface for "
        "the user to interact with ERT.",
    )
    gui_parser.set_defaults(func=runGui)
    gui_parser.add_argument("config", type=valid_file, help=config_help)

    # test_run_parser
    test_run_description = "Run 'test_run' in cli"
    test_run_parser = subparsers.add_parser(
        "test_run", help=test_run_description, description=test_run_description
    )
    test_run_parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output", default=False
    )
    test_run_parser.set_defaults(func=run_cli)
    test_run_parser.add_argument("config", type=valid_file, help=config_help)

    # ensemble_experiment_parser
    ensemble_experiment_description = (
        "Run simulations in cli without performing any updates on the parameters."
    )
    ensemble_experiment_parser = subparsers.add_parser(
        "ensemble_experiment",
        description=ensemble_experiment_description,
        help=ensemble_experiment_description,
    )
    ensemble_experiment_parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output", default=False
    )
    ensemble_experiment_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform simulations."
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform simulations "
        "while realizations 10,11, 12,...,49 will be excluded",
    )
    ensemble_experiment_parser.set_defaults(func=run_cli)
    ensemble_experiment_parser.add_argument("config", type=valid_file, help=config_help)

    # ensemble_smoother_parser
    ensemble_smoother_description = (
        "Run simulations in cli while performing one update"
        " on the parameters by using the ensemble smoother algorithm"
    )
    ensemble_smoother_parser = subparsers.add_parser(
        "ensemble_smoother",
        description=ensemble_smoother_description,
        help=ensemble_smoother_description,
    )
    ensemble_smoother_parser.add_argument(
        "--target-case",
        type=valid_name_not_default,
        required=True,
        help="This is the name of the case where the results for the "
        "updated parameters will be stored",
    )
    ensemble_smoother_parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output", default=False
    )
    ensemble_smoother_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform simulations."
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform simulations "
        "while realizations 10,11, 12,...,49 will be excluded",
    )
    ensemble_smoother_parser.set_defaults(func=run_cli)
    ensemble_smoother_parser.add_argument("config", type=valid_file, help=config_help)

    # es_mda_parser
    es_mda_description = "Run 'es_mda' in cli"
    es_mda_parser = subparsers.add_parser(
        "es_mda", description=es_mda_description, help=es_mda_description
    )
    es_mda_parser.add_argument(
        "--target-case",
        type=valid_name_format,
        help="The es_mda creates multiple cases for the different "
        "iterations. The case names will follow the specified format. "
        "For example, 'Target case format: iter_%%d' will generate "
        "cases with the names iter_0, iter_1, iter_2, iter_3, ....",
    )
    es_mda_parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output", default=False
    )
    es_mda_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform simulations."
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform simulations "
        "while realizations 10,11, 12,...,49 will be excluded",
    )
    es_mda_parser.add_argument(
        "--weights",
        type=valid_weights,
        default=MultipleDataAssimilation.default_weights,
        help="Example Custom Relative Weights: '8,4,2,1'. This means Multiple Data "
        "Assimilation Ensemble Smoother will half the weight applied to the "
        "Observation Errors from one iteration to the next across 4 iterations.",
    )
    es_mda_parser.set_defaults(func=run_cli)
    es_mda_parser.add_argument("config", type=valid_file, help=config_help)

    return parser


def ert_parser(parser, argv):
    return get_ert_parser(parser).parse_args(argv)


def main():
    args = ert_parser(None, sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
