#! /usr/bin/env python

import logging
import sys
from argparse import ArgumentParser

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from everest.util import version_info
from ieverest import IEverest


def ieverest_entry(args=None):
    """Entry point for running the graphical version of everest"""
    if args is None:
        args = sys.argv

    QApplication.setAttribute(
        Qt.AA_X11InitThreads
    )  # required in order to use threads later on
    app = QApplication(args)
    app.setOrganizationName("Equinor/TNO")
    app.setApplicationName("IEverest")

    parser = _build_args_parser()
    options, _ = parser.parse_known_args(args)
    if options.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Remove the null handler if set:
        logging.getLogger().removeHandler(logging.NullHandler())

    logging.info(version_info())

    _keep_in_scope_until_exit = IEverest(config_file=options.config_file)
    sys.exit(app.exec_())


def _build_args_parser():
    """Build arg parser"""
    arg_parser = ArgumentParser(
        description="Everest graphical user interface",
        usage="everest gui [<config_file>]",
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Display debug information"
    )
    arg_parser.add_argument(
        "config_file",
        nargs="?",
        default=None,
        help="Start IEverest with the given configuration file",
    )
    return arg_parser


if __name__ == "__main__":
    ieverest_entry()
