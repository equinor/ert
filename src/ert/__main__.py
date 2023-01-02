from __future__ import annotations

import logging
import logging.config
import os
import re
import sys
from argparse import ArgumentParser, ArgumentTypeError
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Sequence, Union

import yaml
from ecl import set_abort_handler

import ert.shared
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import ResConfig
from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
    WORKFLOW_MODE,
)
from ert.cli.main import ErtCliError, ErtTimeoutError, run_cli
from ert.logging import LOGGING_CONFIG
from ert.logging._log_util_abort import _log_util_abort
from ert.namespace import Namespace
from ert.services import Storage, WebvizErt
from ert.shared.feature_toggling import FeatureToggling
from ert.shared.ide.keywords.data.validation_status import ValidationStatus
from ert.shared.ide.keywords.definitions import (
    IntegerArgument,
    NumberListStringArgument,
    ProperNameArgument,
    ProperNameFormatArgument,
    RangeStringArgument,
)
from ert.shared.models.multiple_data_assimilation import MultipleDataAssimilation
from ert.shared.plugins.plugin_manager import ErtPluginContext
from ert.shared.storage.command import add_parser_options as ert_api_add_parser_options


def run_ert_storage(args: Namespace) -> None:
    kwargs = {"res_config": args.config, "verbose": True}

    if args.database_url is not None:
        kwargs["database_url"] = args.database_url

    with Storage.start_server(**kwargs) as server:
        server.wait()


def run_webviz_ert(args: Namespace) -> None:
    try:
        # pylint: disable=unused-import,import-outside-toplevel
        import webviz_ert  # type: ignore
    except ImportError as err:
        raise ValueError(
            "Running `ert vis` requires that webviz_ert is installed"
        ) from err

    kwargs: Dict[str, Any] = {"verbose": args.verbose}
    if args.config:
        res_config = ResConfig(args.config)
        os.chdir(res_config.config_path)
        ens_path = res_config.ens_path

        # Changing current working directory means we need to
        # only use the base name of the config file path
        kwargs["res_config"] = os.path.basename(args.config)
        kwargs["project"] = os.path.abspath(ens_path)

    if args.database_url is not None:
        kwargs["database_url"] = args.database_url

    with Storage.init_service(**kwargs) as storage:
        storage.wait_until_ready()
        print(
            """
-----------------------------------------------------------

Starting up Webviz-ERT. This might take more than a minute.

-----------------------------------------------------------
"""
        )
        webviz_kwargs = {
            "experimental_mode": args.experimental_mode,
            "verbose": args.verbose,
            "title": kwargs.get("res_config", "ERT - Visualization tool"),
            "project": kwargs.get("project", os.getcwd()),
        }
        with WebvizErt.start_server(**webviz_kwargs) as webviz_ert_server:
            webviz_ert_server.wait()


def strip_error_message_and_raise_exception(validated: ValidationStatus) -> None:
    error = validated.message()
    error = re.sub(r"\<[^>]*\>", " ", error)
    raise ArgumentTypeError(error)


def valid_file(fname: str) -> str:
    if not os.path.isfile(fname):
        raise ArgumentTypeError(f"File was not found: {fname}")
    if not os.access(fname, os.R_OK):
        raise ArgumentTypeError(f"We do not have read permissions for file: {fname}")
    return fname


def valid_realizations(user_input: str) -> str:
    validator = RangeStringArgument()
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_weights(user_input: str) -> str:
    validator = NumberListStringArgument()
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_name_format(user_input: str) -> str:
    validator = ProperNameFormatArgument()
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_name(user_input: str) -> str:
    validator = ProperNameArgument()
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_iter_num(user_input: str) -> str:
    validator = IntegerArgument(from_value=0)
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def valid_num_iterations(user_input: str) -> str:
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
    int_val = attemp_int_conversion(val)
    if not 0 <= int_val <= 65535:
        raise ArgumentTypeError(f"{int_val} is not in valid port range 0-65535")
    return int_val


def valid_port_range(user_input: str) -> range:
    if "-" not in user_input:
        raise ArgumentTypeError("Port range must contain two integers separated by '-'")
    a, b = user_input.split("-")

    port_a, port_b = convert_port(a), convert_port(b)

    if port_b < port_a:
        raise ArgumentTypeError(f"Invalid port range [{a},{b}], {b} is < {a}")

    return range(port_a, port_b + 1)


def range_limited_int(user_input: str) -> int:
    try:
        i = int(user_input)
    except ValueError:
        raise ArgumentTypeError("Must be a int")
    if 0 < i < 100:
        return i
    raise ArgumentTypeError("Range must be in range 1 - 99")


def run_gui_wrapper(args: Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from ert.gui.gert_main import run_gui

    run_gui(args)


# pylint: disable=too-many-statements
def get_ert_parser(parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(description="ERT - Ensemble Reservoir Tool")

    parser.add_argument(
        "--version",
        action="version",
        version=f"{ert.shared.__version__}",
    )

    parser.add_argument(
        "--logdir",
        required=False,
        type=str,
        default="./logs",
        help="Directory where ERT will store the logs. Default is ./logs",
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

    # ert_api
    ert_api_parser = subparsers.add_parser(
        "api",
        description="Expose ERT data through an HTTP server",
    )
    ert_api_parser.set_defaults(func=run_ert_storage)
    ert_api_add_parser_options(ert_api_parser)

    ert_vis_parser = subparsers.add_parser(
        "vis",
        description="Launch webviz-driven visualization tool.",
    )
    ert_vis_parser.set_defaults(func=run_webviz_ert)
    ert_vis_parser.add_argument("--name", "-n", type=str, default="Webviz-ERT")
    ert_vis_parser.add_argument(
        "--experimental-mode",
        action="store_true",
        help="Feature flag for enabling experimental plugins",
    )
    ert_api_add_parser_options(ert_vis_parser)  # ert vis shares args with ert api

    # test_run_parser
    test_run_description = f"Run '{TEST_RUN_MODE}' in cli"
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
        "For example, if 'Number of realizations:50 and active realizations are 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform simulations "
        "while realizations 10,11, 12,...,49 will be excluded.",
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
    es_mda_description = f"Run '{ES_MDA_MODE}' in cli"
    es_mda_parser = subparsers.add_parser(
        ES_MDA_MODE, description=es_mda_description, help=es_mda_description
    )
    es_mda_parser.add_argument(
        "--target-case",
        type=valid_name_format,
        help="The es_mda creates multiple cases for the different "
        "iterations. The case names will follow the specified format. "
        "For example, 'Target case format: iter-%%d' will generate "
        "cases with the names iter-0, iter-1, iter-2, iter-3, ....",
    )
    es_mda_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform simulations."
        "For example, if 'Number of realizations:50 and active realizations are 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform simulations "
        "while realizations 10,11, 12,...,49 will be excluded.",
    )
    es_mda_parser.add_argument(
        "--weights",
        type=valid_weights,
        default=MultipleDataAssimilation.default_weights,
        help="Example custom relative weights: '8,4,2,1'. This means multiple data "
        "assimilation ensemble smoother will half the weight applied to the "
        "observation errors from one iteration to the next across 4 iterations.",
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


def ert_parser(parser: Optional[ArgumentParser], args: Sequence[str]) -> Namespace:
    return get_ert_parser(parser).parse_args(args, namespace=Namespace())


@contextmanager
def start_ert_server(mode: str) -> Generator[None, None, None]:
    if mode in ("api", "vis") or not FeatureToggling.is_enabled("new-storage"):
        yield
        return

    with Storage.start_server():
        yield


def log_process_usage() -> None:
    try:
        # pylint: disable=import-outside-toplevel
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)

        if sys.platform == "darwin":
            # macOS apparently outputs the maxrss value as bytes rather than
            # kilobytes as on Linux.
            #
            # https://stackoverflow.com/questions/59913657/strange-values-of-get-rusage-maxrss-on-macos-and-linux
            rss_scale = 1000
        else:
            rss_scale = 1

        maxrss = usage.ru_maxrss // rss_scale

        usage_dict: Dict[str, Union[int, float]] = {
            "User time": usage.ru_utime,
            "System time": usage.ru_stime,
            "File system inputs": usage.ru_inblock,
            "File system outputs": usage.ru_oublock,
            "Socket messages sent": usage.ru_msgsnd,
            "Socket messages Received": usage.ru_msgrcv,
            "Signals received": usage.ru_nsignals,
            "Swaps": usage.ru_nswap,
            "Peak memory use (kB)": maxrss,
        }
        logging.info(f"Peak memory use: {maxrss} kB", extra=usage_dict)
    # pylint: disable=broad-except
    except Exception as exc:
        logging.warning(
            f"Exception while trying to log ERT process resource usage: {exc}"
        )


def main() -> None:
    # pylint: disable=import-outside-toplevel
    import locale

    locale.setlocale(locale.LC_NUMERIC, "C")

    args = ert_parser(None, sys.argv[1:])

    log_dir = os.path.abspath(args.logdir)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except PermissionError as err:
        sys.exit(str(err))

    os.environ["ERT_LOG_DIR"] = log_dir

    with open(LOGGING_CONFIG, encoding="utf-8") as conf_file:
        logging.config.dictConfig(yaml.safe_load(conf_file))
    set_abort_handler(_log_util_abort)

    logger = logging.getLogger(__name__)
    if args.verbose:
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    FeatureToggling.update_from_args(args)
    try:
        with start_ert_server(args.mode), ErtPluginContext() as context:
            context.plugin_manager.add_logging_handle_to_root(logging.getLogger())
            logger.info(f"Running ert with {args}")
            if FeatureToggling.is_enabled("experiment-server"):
                if args.mode != ENSEMBLE_EXPERIMENT_MODE:
                    raise NotImplementedError(
                        f"experiment-server can only run '{ENSEMBLE_EXPERIMENT_MODE}'"
                    )
            args.func(args)
    except (ErtCliError, ErtTimeoutError) as err:
        logger.exception(str(err))
        sys.exit(str(err))
    except ConfigValidationError as err:
        errMsg = f"Error(s) in configuration file {err.config_file}: {err.errors}"
        logger.exception(errMsg)
        sys.exit(errMsg)
    except BaseException as err:  # pylint: disable=broad-except
        logger.exception(f'ERT crashed unexpectedly with "{err}"')

        logfiles = set()  # Use set to avoid duplicates...
        for loghandler in logging.getLogger().handlers:
            if isinstance(loghandler, logging.FileHandler):
                logfiles.add(loghandler.baseFilename)

        msg = f'ERT crashed unexpectedly with "{err}".\nSee logfile(s) for details:'
        msg += "\n   " + "\n   ".join(logfiles)

        sys.exit(msg)
    finally:
        log_process_usage()
        os.environ.pop("ERT_LOG_DIR")


if __name__ == "__main__":
    main()
