"""
This module is responsible for running the 'fmudesign' CLI tool.

It contains argumenting parsing logic, some argument validation and high-level
functions that delegate to lower-level functions for creating design matrices.

There are two main sub-commands:
    $ fmudesign init  =>   Create example/demo configuration file for the user
    $ fmudesign run   =>   Run a configuration file and produce design matrix

Without arguments (init / run), the CLI will execute 'run' to be backwards
compatible. For more information, look at the code or execute

    $ fmudesign --help

"""

import argparse
import dataclasses
import functools
import shutil
import sys
import traceback
import warnings
from argparse import ArgumentParser, Namespace, _SubParsersAction
from importlib.resources import as_file, files
from pathlib import Path

from packaging.version import Version

import semeio
from semeio.fmudesign import DesignMatrix, excel_to_dict


@dataclasses.dataclass
class Example:
    # Examples used in the 'fmudesign init' subcommand

    filename: str
    description: str
    # Auxiliary files that the main file depend on (external params, seeds, etc.)
    other_files: list[str] = dataclasses.field(default_factory=list)


EXAMPLES = [
    Example(
        "fmudesign_ex_montecarlo.xlsx",
        description=(
            "Shows all statistical parameter distributions, "
            "how to correlate samples, etc."
        ),
    ),
    Example(
        "fmudesign_ex_onebyone.xlsx",
        description="The example file used in fmu-coursedocs. Extensively documented.",
    ),
    Example(
        "ex1_onebyone_rms_repeat.xlsx",
        description="One by one sensitivities with repeating RMS seeds",
    ),
    Example(
        "ex2_correlations.xlsx",
        description=(
            "Sensitivities with group of (correlated) parameters "
            "sampled from distributions"
        ),
        other_files=["ex2_doe1.xlsx"],
    ),
    Example(
        "ex3_velocities.xlsx",
        description="Testing different velocity models with uncertainty",
    ),
    Example(
        "ex4_background_parameters.xlsx",
        description="Sensitivities with background parameters",
        other_files=["ex4_doe1.xlsx"],
    ),
    Example(
        "ex5_single_reference.xlsx",
        description="Sensitivities with a single reference realisation",
    ),
    Example(
        "ex6_singlereference_and_seed.xlsx",
        description=(
            "Sensitivities with a single reference realisation and seed sensitivity"
        ),
    ),
    Example(
        "ex7_background_no_seed.xlsx",
        description="Sensitivities with background but without RMS seed",
        other_files=["ex7_doe1.xlsx"],
    ),
    Example("ex8_mc_with_correls.xlsx", description="Full Monte Carlo sensitivity"),
]


def get_parser() -> tuple[ArgumentParser, _SubParsersAction]:
    """Create argument parser and return (parser, subparsers)."""

    # =============== MAIN PARSER ===============
    parser = argparse.ArgumentParser(
        description="Generates design matrices to be used with ERT",
        epilog=(
            f"""
example usage:
  $ fmudesign --help
  $ fmudesign init --help
  $ fmudesign run --help
  $ fmudesign init {next(iter(EXAMPLES)).filename}
  $ fmudesign run {next(iter(EXAMPLES)).filename} output_example.xlsx

getting help:
  - Documentation: https://equinor.github.io/fmu-tools/fmudesign.html
  - Issue tracker: https://github.com/equinor/semeio/issues"""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"fmudesign {Version(semeio.__version__)}",
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # =============== SUBCOMMAND: run ===============
    description = "Generate a design matrix from a config file."
    epilog = """example usage:
  $ fmudesign --help
  $ fmudesign run input_config_example.xlsx
  $ fmudesign run input_config_example.xlsx output_example.xlsx """

    parser_run = subparsers.add_parser(
        "run",
        help=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog=epilog,
        description=description,
    )

    parser_run.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    parser_run.add_argument(
        "config",
        type=str,
        help="Input design matrix filename in Excel format",
    )
    parser_run.add_argument(
        "destination",
        type=str,
        nargs="?",
        help=(
            "Destination filename for design matrix "
            "(default: generateddesignmatrix.xlsx)"
        ),
        default="generateddesignmatrix.xlsx",
    )
    parser_run.add_argument(
        "--designinput",
        type=str,
        help=(
            "Alternative sheetname for the worksheet designinput (default: designinput)"
        ),
        default="designinput",
    )
    parser_run.add_argument(
        "--defaultvalues",
        type=str,
        help=(
            "Alternative sheetname for worksheet defaultvalues (default: defaultvalues)"
        ),
        default="defaultvalues",
    )
    parser_run.add_argument(
        "--general_input",
        type=str,
        help=(
            "Alternative sheetname for the worksheet general_input"
            "(default: general_input)"
        ),
        default="general_input",
    )
    parser_run.add_argument(
        "-v",
        "--verbose",
        action="count",
        help=(
            "Verbosity of terminal output and plotting, "
            "run with increased verbosity level -v -v to include more information"
        ),
        default=0,
    )
    func = functools.partial(subcommand_run, parser=parser_run)
    parser_run.set_defaults(func=func)

    # =============== SUBCOMMAND: init ===============
    description = "Initialize a demo file to get started with fmudesign."
    epilog = "available demo files:\n"
    ljust = max(len(f.filename) for f in EXAMPLES)
    for example in EXAMPLES:
        epilog += f"  - {example.filename.ljust(ljust)} : {example.description}\n"

    ex_filename = next(iter(EXAMPLES)).filename
    epilog += "\nexample usage:\n"
    epilog += f"  $ fmudesign init {ex_filename}\n"
    epilog += f"  $ fmudesign run {ex_filename}"

    parser_init = subparsers.add_parser(
        "init",
        help=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog=epilog,
        description=description,
    )
    parser_init.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    parser_init.add_argument(
        "file", type=str, nargs="?", help="Name of demo file to create."
    )

    func = functools.partial(subcommand_init, parser=parser_init)
    parser_init.set_defaults(func=func)

    return parser, subparsers


def subcommand_run(args: Namespace, parser: ArgumentParser) -> None:
    """Handles the 'run' subcommand."""

    # Check if defaults were changed
    for sheet in ["designinput", "defaultvalues", "general_input"]:
        default = parser.get_default(sheet)
        custom = getattr(args, sheet)
        if default != custom:
            print(f"Worksheet changed from default: {default!r} -> {custom!r}")

    # Check existence of config file
    if not Path(args.config).is_file():
        raise OSError(f"Input file {args.config} does not exist")

    # Check if destination exists
    if Path(args.config).resolve() == Path(args.destination).resolve():
        raise OSError(
            f'Identical name "{args.config}" have been provided for the input'
            "file and the output file"
        )

    # Parse Excel config file to dict-of-dict configuration
    print(f"Reading file: {args.config!r}")
    config = excel_to_dict(
        args.config,
        gen_input_sheet=args.general_input,
        design_input_sheet=args.designinput,
        default_val_sheet=args.defaultvalues,
    )

    # If destination is 'analysis/generateddesignmatrix.xlsx', then plots
    # will be saved to 'analysis/generateddesignmatrix/<SENSNAME>/<VARNAME>.png'
    output_dir = Path(args.destination).parent / Path(args.destination).stem
    design = DesignMatrix(verbosity=args.verbose, output_dir=output_dir)

    design.generate(config)
    design.to_xlsx(args.destination)


def subcommand_init(args: Namespace, parser: ArgumentParser) -> None:
    """Handles the 'init' subcommand."""
    EXAMPLES_DIR = files("semeio.fmudesign.examples")

    # Verify that all examples in EXAMPLES_DIR exist on disk
    for example in EXAMPLES:
        assert (EXAMPLES_DIR / example.filename).is_file()

    # No files were provided
    if not args.file:
        parser.print_help()
        sys.exit(0)

    filename = args.file.strip()
    valid_names = {ex.filename for ex in EXAMPLES}
    if filename not in valid_names:
        print(f"Error on {filename!r}. Not found among: {valid_names}")
        sys.exit(1)

    if Path(filename).exists():
        print(f"Error on {filename!r}. Already exists.")
        sys.exit(1)

    with as_file(EXAMPLES_DIR / filename) as source_path:
        shutil.copy(source_path, filename)
        print(f"Created file {filename!r}.")

    examples_by_filename = {example.filename: example for example in EXAMPLES}
    for other_file in examples_by_filename[filename].other_files:
        with as_file(EXAMPLES_DIR / other_file) as source_path:
            shutil.copy(source_path, other_file)
            print(f"  Created auxiliary file {other_file!r}.")

    sys.exit(0)


def main() -> None:
    """semeio.fmudesign is a command line utility for generating design matrices

    Wrapper for the the semeio.fmudesign module"""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser, _subparsers = get_parser()

    # Backwards compatibility. If not a known command, assume "run"
    valid_cmds = ("run", "init", "-h", "--help", "-v", "--version")
    if len(sys.argv) > 1 and sys.argv[1] not in valid_cmds:
        sys.argv.insert(1, "run")

    args = parser.parse_args()

    # No subcommand was provided
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(
            "\n \n",
            "fmudesign failed. Read the error message above and fix the input file.\n",
            " - Documentation:           https://equinor.github.io/fmu-tools/fmudesign.html\n",
            " - Course docs:             https://fmu-docs.equinor.com/docs/fmu-coursedocs/fmu-howto/sensitivities/index.html \n",  # noqa: E501
            " - Issues/feature requests: https://github.com/equinor/semeio/issues\n",
            "If you believe this error is a bug or are unable to fix it, create an issue or contact the scout team \n",  # noqa: E501
        )
        sys.exit(1)  # Exit with a non-zero status code (required for smoke tests!)

    print(
        "\n",
        f"Thank you for using fmudesign {Version(semeio.__version__).base_version}\n",
        " - Documentation:           https://equinor.github.io/fmu-tools/fmudesign.html\n",
        " - Course docs:             https://fmu-docs.equinor.com/docs/fmu-coursedocs/fmu-howto/sensitivities/index.html\n",
        " - Issues/feature requests: https://github.com/equinor/semeio/issues\n",
    )


if __name__ == "__main__":
    main()
