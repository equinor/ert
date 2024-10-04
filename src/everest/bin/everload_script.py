#!/usr/bin/env python

import argparse
import datetime
import logging
import os
import shutil
from functools import partial

from ert import LibresFacade
from ert.config import ErtConfig
from ert.storage import open_storage
from everest import MetaDataColumnNames as MDCN
from everest import export
from everest.config import EverestConfig
from everest.config.export_config import ExportConfig
from everest.simulator.everest_to_ert import everest_to_ert_config
from everest.util import version_info


def everload_entry(args=None):
    parser = _build_args_parser()
    options = parser.parse_args(args)
    if options.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Remove the null handler if set:
        logging.getLogger().removeHandler(logging.NullHandler())
    logging.info(version_info())

    config: EverestConfig = options.config_file

    if options.batches is not None:
        batch_list = [int(item) for item in options.batches]
        if config.export is None:
            config.export = ExportConfig(batches=batch_list)
        else:
            config.export.batches = batch_list

    # The case must have run before
    out_dir = config.output_dir
    if not os.path.isdir(out_dir):
        raise RuntimeError("This case was never run, cannot internalize data")

    # The simulation directory should be available
    # At the moment we check only if the simulation folder exists. In the future
    # we may consider carrying out some more thorough sanity check on the folder
    # before proceding with the internalization
    sim_dir = config.simulation_dir
    if not os.path.isdir(sim_dir):
        raise RuntimeError(
            (
                "The simulation directory '{}' cannot be found, "
                "cannot internalize data"
            ).format(sim_dir)
        )

    # Warn the user and ask for confirmation
    storage_path = config.storage_dir
    backup_path = None
    if not os.path.isdir(storage_path):
        storage_path = None
    elif not options.overwrite:
        backup_path = storage_path + datetime.datetime.utcnow().strftime(
            "__%Y-%m-%d_%H.%M.%S.%f"
        )

    if not options.silent and not user_confirms(sim_dir, storage_path, backup_path):
        return

    reload_data(config, backup_path=backup_path)


def _build_args_parser():
    """Build arg parser"""
    arg_parser = argparse.ArgumentParser(
        description="Load Eclipse data from an existing simulation folder",
        usage="""everest load <config_file>""",
    )

    def batch(batch_str, parser=arg_parser):
        batch_str = "{}".format(
            batch_str.strip()
        )  # Because isnumeric only works on unicode strings in py27
        if not batch_str.isnumeric() or (batch_str[0] == "0" and len(batch_str) > 1):
            parser.error("Invalid batch given: '{}'".format(batch_str))
        return int(batch_str)

    arg_parser.add_argument(
        "config_file",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )
    arg_parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Backup/overwrite current internal storage without asking",
    )
    arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the internal storage instead of backing it up",
    )
    arg_parser.add_argument(
        "-b",
        "--batches",
        nargs="+",
        type=batch,
        help="List of batches to be internalized",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Display debug information in the terminal",
    )

    return arg_parser


def user_confirms(simulation_path, storage_path=None, backup_path=None):
    print("\n*************************************************************")
    print("*** This operation can take several minutes or even hours ***")
    print("*************************************************************\n")
    print("The Everest internal storage will be populated using data from")
    print("  {}".format(simulation_path))
    if storage_path is not None:
        if backup_path is None:
            print("WARNING: the current internal storage will be deleted")
        else:
            print("The current internal storage will be backed up in")
            print("  {}".format(backup_path))
    while True:
        text = input("Are you sure you want to proceed? (y/n) ")
        if not text:
            continue
        if text[0] in ("n", "N"):
            return False
        if text[0] in ("y", "Y"):
            return True


def reload_data(ever_config: EverestConfig, backup_path=None):
    """Load data from a completed optimization into ert storage

    If @batch_ids are given, only the specified batches are internalized
    If a @backup_path is specified, the current internal storage will be copied
    to the given path instead of being deleted.
    """
    # The ErtConfig constructor is picky, these sections can produce errors, but
    # we don't need them for re-internalizing the data

    ever_config.forward_model = None
    ever_config.install_jobs = None
    ever_config.install_workflow_jobs = None
    ever_config.install_data = None
    ever_config.install_templates = None

    # prepare the ErtConfig object
    ert_config_dict = everest_to_ert_config(
        ever_config, site_config=ErtConfig.read_site_config()
    )
    ert_config = ErtConfig.with_plugins().from_dict(config_dict=ert_config_dict)

    # load information about batches from previous run
    df = export(ever_config, export_ecl=False)
    groups = df.groupby(by=MDCN.BATCH)

    # backup or delete the previous internal storage
    if backup_path:
        shutil.move(ever_config.storage_dir, backup_path)
    else:
        shutil.rmtree(ever_config.storage_dir)

    # internalize one batch at a time
    for batch_id, group in groups:
        _internalize_batch(ert_config, batch_id, group)


def _internalize_batch(ert_config, batch_id, batch_data):
    facade = LibresFacade(ert_config)
    case_name = "batch_{}".format(batch_id)
    batch_size = batch_data.shape[0]
    with open_storage(facade.enspath, "w") as storage:
        experiment = storage.get_experiment_by_name(f"experiment_{case_name}")
        ensemble = experiment.get_ensemble_by_name(case_name)
        # Everest artificially inflates the ensemble size as it is not possible to
        # add after the fact, therefore a batch is much smaller than the overall
        # ensemble size
        realizations = [True] * batch_size + [False] * (
            facade.get_ensemble_size() - batch_size
        )
        facade.load_from_forward_model(ensemble, realizations, 0)


if __name__ == "__main__":
    everload_entry()
