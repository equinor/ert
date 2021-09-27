import logging
import os
import sys
import re
import yaml
from argparse import ArgumentParser, ArgumentTypeError
from contextlib import contextmanager

from ert_logging import LOGGING_CONFIG
from ert_shared import clear_global_state
from ert_shared.cli.main import run_cli, ErtCliError
from ert_shared.storage.main import run_server
from ert_shared.cli import (
    ENSEMBLE_SMOOTHER_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    TEST_RUN_MODE,
    WORKFLOW_MODE,
)
from ert_shared.ide.keywords.definitions import (
    RangeStringArgument,
    ProperNameArgument,
    ProperNameFormatArgument,
    NumberListStringArgument,
    IntegerArgument,
)
from ert_shared.models.multiple_data_assimilation import MultipleDataAssimilation
from ert_shared.plugins.plugin_manager import ErtPluginContext
from ert_shared.plugins import launch_visualization_plugin
from ert_shared.feature_toggling import FeatureToggling, feature_enabled
from ert_shared.storage.command import add_parser_options as ert_api_add_parser_options
import ert_shared


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


def valid_iter_num(user_input):
    validator = IntegerArgument(from_value=0)
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_num_iterations(user_input):
    validator = IntegerArgument(from_value=1)
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def attemp_int_conversion(val: str) -> int:
    try:
        return int(val)
    except ValueError:
        raise ArgumentTypeError(f"{val} is not a valid integer")


def convert_port(val: str) -> int:
    val = attemp_int_conversion(val)
    if not (0 <= val <= 65535):
        raise ArgumentTypeError(f"{val} is not in valid port range 0-65535")
    return val


def valid_port_range(user_input: str) -> range:
    if "-" not in user_input:
        raise ArgumentTypeError("Port range must contain two integers separated by '-'")
    a, b = user_input.split("-")

    a, b = convert_port(a), convert_port(b)

    if b < a:
        raise ArgumentTypeError(f"Invalid port range [{a},{b}], {b} is < {a}")

    return range(a, b + 1)


def range_limited_int(user_input):
    try:
        i = int(user_input)
    except ValueError:
        raise ArgumentTypeError("Must be a int")
    if 0 < i < 100:
        return i
    raise ArgumentTypeError("Range must be in range 1 - 99")


def run_gui_wrapper(args):
    from ert_gui.gert_main import run_gui

    run_gui(args)


def get_ert_parser(parser=None):
    if parser is None:
        parser = ArgumentParser(description="ERT - Ensemble Reservoir Tool")

    parser.add_argument(
        "--version",
        action="version",
        version="{}".format(ert_shared.__version__),
    )

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
    subparsers.required = True

    config_help = "ERT configuration file"

    # gui_parser
    gui_parser = subparsers.add_parser(
        "gui",
        description="Opens an independent graphical user interface for "
        "the user to interact with ERT.",
    )
    gui_parser.set_defaults(func=run_gui_wrapper)
    gui_parser.add_argument("config", type=valid_file, help=config_help)
    gui_parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output.", default=False
    )
    FeatureToggling.add_feature_toggling_args(gui_parser)
    gui_url_or_bind = gui_parser.add_mutually_exclusive_group()
    gui_url_or_bind.add_argument(
        "--storage-api-url",
        type=str,
        help="Storage API URL. If not provided, ERT will start a server for you.",
    )
    gui_url_or_bind.add_argument(
        "--storage-api-bind",
        type=str,
        help="Bind the Storage API to this server socket.",
        default="127.0.0.1:0",
    )

    # ert_api
    ert_api_parser = subparsers.add_parser(
        "api",
        description="Expose ERT data through an HTTP server",
    )
    ert_api_parser.set_defaults(func=run_server)
    ert_api_add_parser_options(ert_api_parser)

    ert_vis_parser = subparsers.add_parser(
        "vis",
        description="Launch webviz-driven visualization tool.",
    )
    ert_vis_parser.set_defaults(func=launch_visualization_plugin)
    ert_vis_parser.add_argument("--name", "-n", type=str, default="Webviz-ERT")
    ert_vis_parser.add_argument(
        "--project",
        "-p",
        type=str,
        help="Path to folder running ert storage server",
        default=os.getcwd(),
    )
    ert_vis_parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output.", default=False
    )

    # test_run_parser
    test_run_description = "Run '{}' in cli".format(TEST_RUN_MODE)
    test_run_parser = subparsers.add_parser(
        TEST_RUN_MODE, help=test_run_description, description=test_run_description
    )

    # ensemble_experiment_parser
    ensemble_experiment_description = (
        "Run simulations in cli without performing any updates on the parameters."
    )
    ensemble_experiment_parser = subparsers.add_parser(
        ENSEMBLE_EXPERIMENT_MODE,
        description=ensemble_experiment_description,
        help=ensemble_experiment_description,
    )
    ensemble_experiment_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform simulations. "
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform simulations "
        "while realizations 10,11, 12,...,49 will be excluded.",
    )
    ensemble_experiment_parser.add_argument(
        "--current-case",
        type=valid_name,
        required=False,
        help="Name of the case where the results for the simulation "
        "using the prior parameters will be stored.",
    )

    ensemble_experiment_parser.add_argument(
        "--iter-num",
        type=valid_iter_num,
        default=0,
        required=False,
        help="Specification of which iteration number is about to be made. "
        "Use iter-num to avoid recomputing the priors.",
    )

    # ensemble_smoother_parser
    ensemble_smoother_description = (
        "Run simulations in cli while performing one update"
        " on the parameters by using the ensemble smoother algorithm."
    )
    ensemble_smoother_parser = subparsers.add_parser(
        ENSEMBLE_SMOOTHER_MODE,
        description=ensemble_smoother_description,
        help=ensemble_smoother_description,
    )
    ensemble_smoother_parser.add_argument(
        "--target-case",
        type=valid_name,
        required=True,
        help="Name of the case where the results for the "
        "updated parameters will be stored.",
    )
    ensemble_smoother_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform simulations."
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform simulations "
        "while realizations 10,11, 12,...,49 will be excluded",
    )
    ensemble_smoother_parser.add_argument(
        "--current-case",
        type=valid_name,
        required=False,
        help="Name of the case where the results for the simulation "
        "using the prior parameters will be stored.",
    )

    # iterative_ensemble_smoother_parser
    iterative_ensemble_smoother_description = (
        "Run simulations in cli while performing updates"
        " on the parameters using the iterative ensemble smoother algorithm."
    )
    iterative_ensemble_smoother_parser = subparsers.add_parser(
        ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
        description=iterative_ensemble_smoother_description,
        help=iterative_ensemble_smoother_description,
    )
    iterative_ensemble_smoother_parser.add_argument(
        "--target-case",
        type=valid_name_format,
        required=True,
        help="The iterative ensemble smoother creates multiple cases for the different "
        "iterations. The case names will follow the specified format. "
        "For example, 'Target case format: iter_%%d' will generate "
        "cases with the names iter_0, iter_1, iter_2, iter_3, ....",
    )
    iterative_ensemble_smoother_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform simulations."
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform simulations "
        "while realizations 10,11, 12,...,49 will be excluded",
    )
    iterative_ensemble_smoother_parser.add_argument(
        "--current-case",
        type=valid_name,
        required=False,
        help="Name of the case where the results for the simulation "
        "using the prior parameters will be stored.",
    )
    iterative_ensemble_smoother_parser.add_argument(
        "--num-iterations",
        type=valid_num_iterations,
        required=False,
        help="The number of iterations to run.",
    )

    # es_mda_parser
    es_mda_description = "Run '{}' in cli".format(ES_MDA_MODE)
    es_mda_parser = subparsers.add_parser(
        ES_MDA_MODE, description=es_mda_description, help=es_mda_description
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
    es_mda_parser.add_argument(
        "--current-case",
        type=valid_name,
        required=False,
        help="Name of the case where the results for the simulation "
        "using the prior parameters will be stored.",
    )
    es_mda_parser.add_argument(
        "--start-iteration",
        default="0",
        type=valid_iter_num,
        required=False,
        help="Which iteration the evaluation should start from. "
        "Requires cases previous to the specified iteration to exist.",
    )

    workflow_description = "Executes the workflow given"
    workflow_parser = subparsers.add_parser(
        WORKFLOW_MODE, help=workflow_description, description=workflow_description
    )
    workflow_parser.add_argument(help="Name of workflow", dest="name")

    # Common arguments/defaults for all non-gui modes
    for cli_parser in [
        test_run_parser,
        ensemble_experiment_parser,
        ensemble_smoother_parser,
        iterative_ensemble_smoother_parser,
        es_mda_parser,
        workflow_parser,
    ]:
        cli_parser.set_defaults(func=run_cli)
        cli_parser.add_argument(
            "--verbose", action="store_true", help="Show verbose output.", default=False
        )
        cli_parser.add_argument(
            "--color-always",
            action="store_true",
            help="Force coloring of monitor output, which is automatically"
            + " disabled if the output stream is not a terminal.",
            default=False,
        )
        cli_parser.add_argument(
            "--disable-monitoring",
            action="store_true",
            help="Disable monitoring.",
            default=False,
        )
        cli_parser.add_argument(
            "--port-range",
            type=valid_port_range,
            required=False,
            help="Port range [a,b] to be used by the evaluator. Format: a-b",
        )
        cli_parser.add_argument("config", type=valid_file, help=config_help)

        FeatureToggling.add_feature_toggling_args(cli_parser)

    return parser


def ert_parser(parser, argv):
    return get_ert_parser(parser).parse_args(argv)


@contextmanager
def start_ert_server():
    monitor = None
    if FeatureToggling.is_enabled("new-storage"):
        from ert_shared.storage.server_monitor import ServerMonitor

        monitor = ServerMonitor.get_instance()
        monitor.start()

    try:
        yield
    finally:
        if monitor is not None:
            monitor.shutdown()


def main():
    with open(LOGGING_CONFIG, encoding="utf-8") as conf_file:
        logging.config.dictConfig(yaml.safe_load(conf_file))
    import locale

    locale.setlocale(locale.LC_NUMERIC, "C")

    args = ert_parser(None, sys.argv[1:])
    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.setLevel("DEBUG")
    FeatureToggling.update_from_args(args)
    try:
        with start_ert_server(), ErtPluginContext() as context:
            context.plugin_manager.add_logging_handle_to_root(logging.getLogger())
            logger.info("Running ert with {}".format(str(args)))
            args.func(args)
    except ErtCliError as err:
        logger.exception(str(err))
        sys.exit(str(err))
    except:
        logger.exception("ert crashed unexpectedly")
        sys.exit("ert crashed unexpectedly")
    clear_global_state()
