from __future__ import annotations

import argparse
import locale
import logging
import logging.config
import multiprocessing
import os
import re
import resource
import sys
import warnings
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import yaml
from opentelemetry.trace import Status, StatusCode

import ert.shared
from _ert.threading import set_signal_handler
from ert.cli.main import ErtCliError, run_cli
from ert.config import ConfigValidationError, ErtConfig, lint_file
from ert.logging import LOGGING_CONFIG
from ert.mode_definitions import (
    ENIF_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    TEST_RUN_MODE,
    WORKFLOW_MODE,
)
from ert.namespace import Namespace
from ert.plugins import ErtPluginContext, ErtPluginManager
from ert.run_models.multiple_data_assimilation import MultipleDataAssimilation
from ert.services import StorageService, WebvizErt
from ert.shared.storage.command import add_parser_options as ert_api_add_parser_options
from ert.storage import ErtStorageException
from ert.trace import trace, tracer
from ert.validation import (
    IntegerArgument,
    NumberListStringArgument,
    ProperNameArgument,
    ProperNameFormatArgument,
    RangeStringArgument,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


def run_ert_storage(args: Namespace, _: ErtPluginManager | None = None) -> None:
    with StorageService.start_server(
        verbose=True,
        project=ErtConfig.from_file(args.config).ens_path,
        parent_pid=os.getpid(),
    ) as server:
        server.wait()


def run_webviz_ert(args: Namespace, _: ErtPluginManager | None = None) -> None:
    try:
        import webviz_ert  # type: ignore  # noqa
    except ImportError as err:
        raise ValueError(
            "Running `ert vis` requires that webviz_ert is installed"
        ) from err

    kwargs: dict[str, Any] = {"verbose": args.verbose}
    ert_config = ErtConfig.with_plugins().from_file(args.config)
    os.chdir(ert_config.config_path)
    ens_path = ert_config.ens_path

    # Changing current working directory means we need to
    # only use the base name of the config file path
    kwargs["ert_config"] = os.path.basename(args.config)
    kwargs["project"] = os.path.abspath(ens_path)
    with StorageService.init_service(project=os.path.abspath(ens_path)) as storage:
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
            "title": kwargs.get("ert_config", "ERT - Visualization tool"),
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


def valid_ensemble(user_input: str) -> str | UUID:
    if user_input.startswith("UUID="):
        return UUID(user_input[5:])
    return valid_name(user_input)


def valid_num_iterations(user_input: str) -> str:
    validator = IntegerArgument(from_value=1)
    validated = validator.validate(user_input)
    if validated.failed():
        strip_error_message_and_raise_exception(validated)
    return user_input


def _attempt_int_conversion(val: str) -> int:
    try:
        return int(val)
    except ValueError as e:
        raise ArgumentTypeError(f"{val} is not a valid integer") from e


def convert_port(val: str) -> int:
    int_val = _attempt_int_conversion(val)
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


def run_gui_wrapper(args: Namespace, ert_plugin_manager: ErtPluginManager) -> None:
    # Importing ert.gui on-demand saves ~0.5 seconds off `from ert import __main__`
    from ert.gui.main import run_gui  # noqa: PLC0415

    run_gui(args, ert_plugin_manager)


def run_lint_wrapper(args: Namespace, _: ErtPluginManager) -> None:
    lint_file(args.config)


class DeprecatedAction(argparse.Action):
    def __init__(self, alternative_option: str | None = None, **kwargs: Any) -> None:
        self.alternative_option: str | None = alternative_option
        super().__init__(**kwargs)

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        alternative_msg: str = (
            f"Use {self.alternative_option} instead." if self.alternative_option else ""
        )
        warnings.warn(
            f"{option_string} is deprecated and will be removed in "
            f"future versions. {alternative_msg}",
            stacklevel=1,
        )
        setattr(namespace, self.dest, values)


def get_ert_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
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

    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Start ERT in read-only mode",
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

    # lint_parser
    lint_parser = subparsers.add_parser(
        "lint",
        description="Find and print errors in existing .ert configuration, including "
        "errors related to files used in the ert config.",
    )
    lint_parser.set_defaults(func=run_lint_wrapper)
    lint_parser.add_argument("config", type=valid_file, help=config_help)
    lint_parser.add_argument(
        "--verbose", action="store_true", help="Show verbose output.", default=False
    )

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
    test_run_parser.add_argument(
        "--current-case",
        type=valid_name,
        default="default",
        action=DeprecatedAction,
        dest="current_ensemble",
        help="Deprecated: This argument is deprecated and will be "
        "removed in future versions. Use --current-ensemble instead.",
    )
    test_run_parser.add_argument(
        "--current-ensemble",
        type=valid_name,
        default="default",
        help="Name of the ensemble where the results for the experiment "
        "using the prior parameters will be stored.",
    )

    # ensemble_experiment_parser
    ensemble_experiment_description = (
        "Run experiments in cli without performing any updates on the parameters."
    )
    ensemble_experiment_parser = subparsers.add_parser(
        ENSEMBLE_EXPERIMENT_MODE,
        description=ensemble_experiment_description,
        help=ensemble_experiment_description,
    )
    ensemble_experiment_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform experiments. "
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform experiments "
        "while realizations 10,11, 12,...,49 will be excluded.",
    )
    ensemble_experiment_parser.add_argument(
        "--current-case",
        type=valid_ensemble,
        default="default",
        action=DeprecatedAction,
        dest="current_ensemble",
        help="Deprecated: This argument is deprecated and will be "
        "removed in future versions. Use --current-ensemble instead.",
    )
    ensemble_experiment_parser.add_argument(
        "--current-ensemble",
        type=valid_ensemble,
        default="default",
        help="Name of the ensemble where the results for the experiment "
        "using the prior parameters will be stored.",
    )
    ensemble_experiment_parser.add_argument(
        "--experiment-name",
        type=str,
        default="ensemble-experiment",
        help="Name of the experiment",
    )

    # ensemble_smoother_parser
    ensemble_smoother_description = (
        "Run experiments in cli while performing one update"
        " on the parameters by using the ensemble smoother algorithm."
    )
    ensemble_smoother_parser = subparsers.add_parser(
        ENSEMBLE_SMOOTHER_MODE,
        description=ensemble_smoother_description,
        help=ensemble_smoother_description,
    )
    ensemble_smoother_parser.add_argument(
        "--current-case",
        type=valid_name,
        action=DeprecatedAction,
        alternative_option="--current-ensemble",
        dest="current_ensemble",
        default="default",
        help="Deprecated: This argument is deprecated and has no effect.",
    )
    ensemble_smoother_parser.add_argument(
        "--current-ensemble",
        type=valid_name,
        default="default",
        help="This argument is deprecated and has no effect.",
    )
    ensemble_smoother_parser.add_argument(
        "--target-case",
        type=valid_name_format,
        default="iter-%d",
        action=DeprecatedAction,
        alternative_option="--target-ensemble",
        dest="target_ensemble",
        help="Deprecated: This argument is deprecated and will be "
        "removed in future versions. Use --target-ensemble instead.",
    )
    ensemble_smoother_parser.add_argument(
        "--target-ensemble",
        type=valid_name_format,
        default="iter-%d",
        dest="target_ensemble",
        help="Name of the ensemble where the results for the "
        "updated parameters will be stored.",
    )
    ensemble_smoother_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform experiments."
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform experiments "
        "while realizations 10,11, 12,...,49 will be excluded",
    )
    ensemble_smoother_parser.add_argument(
        "--experiment-name",
        type=str,
        default="es",
        help="Name of the experiment",
    )

    # enif_parser
    enif_description = (
        "Run experiments in cli while performing one update"
        " on the parameters by using the ensemble information filter algorithm."
    )
    enif_parser = subparsers.add_parser(
        ENIF_MODE,
        description=enif_description,
        help=enif_description,
    )
    enif_parser.add_argument(
        "--target-ensemble",
        type=valid_name_format,
        default="iter-%d",
        dest="target_ensemble",
        help="Name of the ensemble where the results for the "
        "updated parameters will be stored.",
    )
    enif_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform experiments."
        "For example, if 'Number of realizations:50 and Active realizations is 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform experiments "
        "while realizations 10,11, 12,...,49 will be excluded",
    )
    enif_parser.add_argument(
        "--experiment-name",
        type=str,
        default="enif",
        help="Name of the experiment",
    )

    # es_mda_parser
    es_mda_description = f"Run '{ES_MDA_MODE}' in cli"
    es_mda_parser = subparsers.add_parser(
        ES_MDA_MODE, description=es_mda_description, help=es_mda_description
    )
    es_mda_parser.add_argument(
        "--target-case",
        type=valid_name_format,
        action=DeprecatedAction,
        alternative_option="--target-ensemble",
        dest="target_ensemble",
        help="Deprecated: This argument is deprecated and will be "
        "removed in future versions. Use --target-ensemble instead.",
    )
    es_mda_parser.add_argument(
        "--target-ensemble",
        type=valid_name_format,
        help="The es_mda creates multiple ensembles for the different "
        "iterations. The ensemble names will follow the specified format. "
        "For example, 'Target ensemble format: iter-%%d' will generate "
        "ensembles with the names iter-0, iter-1, iter-2, iter-3, ....",
    )
    es_mda_parser.add_argument(
        "--realizations",
        type=valid_realizations,
        help="These are the realizations that will be used to perform experiments."
        "For example, if 'Number of realizations:50 and active realizations are 0-9', "
        "then only realizations 0,1,2,3,...,9 will be used to perform experiments "
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
        "--restart-case",
        type=valid_name,
        default=None,
        action=DeprecatedAction,
        dest="restart_ensemble_id",
        help="Deprecated: This argument is deprecated and will be "
        "removed in future versions. Use --restart-ensemble instead.",
    )
    es_mda_parser.add_argument(
        "--restart-ensemble",
        type=valid_name,
        default=None,
        dest="restart_ensemble_id",
        help="Deprecated: This argument is deprecated and will be "
        "removed in future versions. Use --restart-ensemble-id instead.",
    )
    es_mda_parser.add_argument(
        "--restart-ensemble-id",
        type=valid_name,  # validate UUID
        default=None,
        dest="restart_ensemble_id",
        help="UUID of the ensemble where the results for the experiment "
        "using the prior parameters will be stored. Iteration number is read "
        "from this ensemble. If provided this will be a restart a run",
    )
    es_mda_parser.add_argument(
        "--experiment-name",
        type=valid_name,
        default="es-mda",
        dest="experiment_name",
        help="Name of the experiment",
    )

    workflow_description = "Executes the workflow given"
    workflow_parser = subparsers.add_parser(
        WORKFLOW_MODE, help=workflow_description, description=workflow_description
    )
    workflow_parser.add_argument(help="Name of workflow", dest="name")
    workflow_parser.add_argument(
        "--ensemble", help="Which ensemble to use", default=None
    )

    # Common arguments/defaults for all non-gui modes
    for cli_parser in [
        test_run_parser,
        ensemble_experiment_parser,
        ensemble_smoother_parser,
        enif_parser,
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
            help="Monitoring will continuously print the status of the realisations"
            + " classified into Waiting, Pending, Running, Failed, Finished"
            + " and Unknown.",
            default=False,
        )
        cli_parser.add_argument(
            "--port-range",
            type=valid_port_range,
            required=False,
            help="Port range [a,b] to be used by the evaluator. Format: a-b",
        )
        cli_parser.add_argument("config", type=valid_file, help=config_help)

    return parser


def ert_parser(parser: ArgumentParser | None, args: Sequence[str]) -> Namespace:
    return get_ert_parser(parser).parse_args(
        args,
        namespace=Namespace(),
    )


def log_process_usage() -> None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss = ert.shared.status.utils.get_ert_memory_usage()

        usage_dict: dict[str, int | float] = {
            "User time": usage.ru_utime,
            "System time": usage.ru_stime,
            "File system inputs": usage.ru_inblock,
            "File system outputs": usage.ru_oublock,
            "Socket messages sent": usage.ru_msgsnd,
            "Socket messages Received": usage.ru_msgrcv,
            "Signals received": usage.ru_nsignals,
            "Swaps": usage.ru_nswap,
            "Peak memory use (KB)": max_rss,
        }
        logger.info(f"Ert process usage: {usage_dict}")
    except Exception as exc:
        logger.warning(
            f"Exception while trying to log ERT process resource usage: {exc}"
        )


@tracer.start_as_current_span("ert.application.start")
def main() -> None:
    span = trace.get_current_span()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    locale.setlocale(locale.LC_NUMERIC, "C")

    # Have ErtThread re-raise uncaught exceptions on main thread
    set_signal_handler()

    args = ert_parser(None, sys.argv[1:])

    log_dir = os.path.abspath(args.logdir)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except PermissionError as err:
        sys.exit(str(err))

    os.environ["ERT_LOG_DIR"] = log_dir

    with open(LOGGING_CONFIG, encoding="utf-8") as conf_file:
        config_dict = yaml.safe_load(conf_file)
        for handler_name, handler_config in config_dict["handlers"].items():
            if handler_name == "file":
                handler_config["filename"] = "ert-log.txt"
            if "ert.logging.TimestampedFileHandler" in handler_config.values():
                handler_config["config_filename"] = args.config
        try:
            logging.config.dictConfig(config_dict)
        except ValueError as err:
            if "handler 'file'" in str(err):
                exit_msg = (
                    f"Could not configure log handler for files. "
                    f"Check if you have write-access to the logs-directory ({log_dir})."
                )
            else:
                exit_msg = str(err)
            os.environ.pop("ERT_LOG_DIR")
            sys.exit(exit_msg)

    logger = logging.getLogger(__name__)
    if args.verbose:
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)
    try:
        with ErtPluginContext(logger=logging.getLogger()) as context:
            logger.info(f"Running ert with {args} in {os.getcwd()}")
            args.func(args, context.plugin_manager)
    except (ErtCliError, ErtStorageException) as err:
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(err)
        logger.debug(str(err))
        sys.exit(str(err))
    except ConfigValidationError as err:
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(err)
        err_msg = err.cli_message()
        logger.debug(err_msg)
        sys.exit(err_msg)
    except BaseException as err:
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(err)
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
    if (
        sys.platform == "linux"
        and multiprocessing.get_start_method(allow_none=True) != "forkserver"
    ):
        multiprocessing.set_start_method("forkserver")
    main()
