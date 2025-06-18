import argparse
import sys

try:
    from ert.shared.version import __version__ as everest_version
except ImportError:
    everest_version = "0.0.0"
from ert.trace import tracer
from everest.bin.config_branch_script import config_branch_entry
from everest.bin.everconfigdump_script import config_dump_entry
from everest.bin.everest_script import everest_entry
from everest.bin.everexport_script import everexport_entry
from everest.bin.everlint_script import lint_entry
from everest.bin.kill_script import kill_entry
from everest.bin.monitor_script import monitor_entry
from everest.bin.visualization_script import visualization_entry


def _build_args_parser() -> argparse.ArgumentParser:
    """Build arg parser"""
    arg_parser = argparse.ArgumentParser(
        description="Tool for performing reservoir management optimization",
        usage=(
            "everest <command> [<args>]\n\n"
            "The most commonly used everest commands are:\n"
            f"{EverestMain.methods_help()}\n\n"
            "Run everest <command> --help for more information on a command"
        ),
    )
    arg_parser.add_argument("command", help="Subcommand to run")
    arg_parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {everest_version}",
    )
    return arg_parser


class EverestMain:
    def __init__(self, args: list[str]) -> None:
        parser = _build_args_parser()
        # Parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        parsed_args = parser.parse_args(args[1:2])
        if not hasattr(self, parsed_args.command):
            parser.error("Unrecognized command")

        # Use dispatch pattern to invoke method with same name
        getattr(self, parsed_args.command)(args[2:])

    @classmethod
    def methods_help(cls) -> str:
        """Return documentation of the public methods in this class"""
        pubmets = [m for m in dir(cls) if not m.startswith("_")]
        pubmets.remove("methods_help")  # Current method should not show up in desc
        maxlen = max(len(m) for m in pubmets)
        docstrs = [getattr(cls, m).__doc__ for m in pubmets]
        doclist = [
            m.ljust(maxlen + 1) + d for m, d in zip(pubmets, docstrs, strict=False)
        ]
        return "\n".join(doclist)

    def run(self, args: list[str]) -> None:
        """Start an optimization case base on given config file"""
        everest_entry(args)

    def monitor(self, args: list[str]) -> None:
        """Monitor a running optimization case base on given config file"""
        monitor_entry(args)

    def kill(self, args: list[str]) -> None:
        """Kill a running optimization case base on given config file"""
        kill_entry(args)

    def gui(self, _: list[str]) -> None:
        """Start the graphical user interface (Removed)"""
        print(
            "The gui command has been removed. "
            "Please use the run command with the --gui option instead."
        )

    def export(self, args: list[str]) -> None:
        """Export data from a completed optimization case"""
        everexport_entry(args)

    def lint(self, args: list[str]) -> None:
        """Validate a config file"""
        lint_entry(args)

    def render(self, args: list[str]) -> None:
        """Display the configuration data loaded from a config file"""
        config_dump_entry(args)

    def branch(self, args: list[str]) -> None:
        """Construct possible restart config file"""
        config_branch_entry(args)

    def results(self, args: list[str]) -> None:
        """Start everest visualization plugin"""
        visualization_entry(args)


@tracer.start_as_current_span("everest.application.start")
def start_everest(args: list[str] | None = None) -> None:
    """Main entry point for the everest application"""
    args = args or sys.argv
    EverestMain(args)
