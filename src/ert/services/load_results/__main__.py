"""_ert_internal.load_results

This script internalises data from a given RUNPATH into ERT Storage and is ran
as the last job in a forward model.

NOTE: No parts of ert should ever import from this script. This script is meant
to be executed directly using 'python -m _ert_internal.load_results'

"""

from __future__ import annotations

import logging
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from uuid import UUID

from ert.storage import open_single_realization

logger = logging.getLogger(__name__)
from ert.config import ParameterConfig, ResponseConfig, SummaryConfig

if TYPE_CHECKING:
    from ert.storage import Realization


def _load_parameters_into_storage(
    runpath: str,
    real: Realization,
    parameter_configuration: Iterable[ParameterConfig],
) -> bool:
    if real.ensemble.iteration != 0:
        return True

    success = True
    for config in parameter_configuration:
        if not config.forward_init:
            continue
        try:
            start_time = time.perf_counter()
            logger.info(f"Starting to load parameter: {config.name}")
            ds = config.read_from_runpath(Path(runpath), real.index)
            logger.info(
                f"Loaded {config.name}",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
            start_time = time.perf_counter()
            real.save_parameters(config.name, ds)
            logger.info(
                f"Saved {config.name} to storage",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
        except Exception as exc:
            logger.exception(
                f"Loading parameter '{config.name}' failed: {exc}", exc_info=exc
            )
            success = False
    return success


def _load_responses_into_storage(
    runpath: str, real: Realization, response_configs: Iterable[ResponseConfig]
) -> bool:
    success = True
    for config in response_configs:
        if isinstance(config, SummaryConfig) and not config.keys:
            # Nothing to load, should not be handled here, should never be
            # added in the first place
            continue

        try:
            start_time = time.perf_counter()
            logger.info(f"Starting to load response: {config.name}")
            ds = config.read_from_file(runpath, real.index)
            logger.info(
                f"Loaded {config.name}",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
            start_time = time.perf_counter()
            real.save_response(config.name, ds)
            logger.info(
                f"Saved {config.name} to storage",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
        except Exception as exc:
            logger.exception(
                f"Loading response '{config.name}' failed: {exc}", exc_info=exc
            )
            success = False
    return success


def load_results(
    real: Realization,
    runpath: str,
    parameters: Iterable[ParameterConfig],
    responses: Iterable[ResponseConfig],
) -> bool:
    # We only read parameters after the prior, after that, ERT
    # handles parameters
    try:
        param_success = _load_parameters_into_storage(
            runpath,
            real,
            parameters,
        )
        resp_success = _load_responses_into_storage(
            runpath,
            real,
            responses,
        )
        return param_success and resp_success

    except Exception as exc:
        logging.exception(
            f"Failed to load results for realization {real.index}", exc_info=exc
        )
        return False


def parse_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("--storage-path", type=Path)
    ap.add_argument("--ensemble", type=UUID)
    ap.add_argument("--index", type=int)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"{args.storage_path=}")
    real = open_single_realization(
        args.storage_path, args.ensemble, args.index, mode="w"
    )
    exp = real.experiment
    if not load_results(
        real,
        os.getcwd(),
        exp.parameter_configuration.values(),
        exp.response_configuration.values(),
    ):
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    main()
