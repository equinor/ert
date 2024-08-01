import argparse
import sys
from typing import Any, List, Optional, Sequence

import pandas as pd
from fmu import ensemble

from ert import ErtScript

DESCRIPTION = """
CSV_EXPORT2 will export selected Eclipse summary vectors to a CSV file.
The vector selection is independent of the ``SUMMARY`` keywords in the
ert config file.

The CSV file will look like:

======== ==== =========== ==== ======
ENSEMBLE REAL DATE        FOPR FOPT
======== ==== =========== ==== ======
iter-0   0    2020-01-01  800  0
iter-0   0    2020-02-01  1000 365000
iter-0   1    2020-01-01  700  0
iter-0   1    2020-01-01  1100 401500
======== ==== =========== ==== ======

The time frequency must be chosen. If ``raw``, the original timesteps from
Eclipse is chosen, and it will be individual pr. realization. If ``daily``,
``weekly``, ``monthly``  or  ``yearly`` is chosen, only data at those dates are
given for all realization. Rate data (e.g.  FOPR) is valid for the given dates,
but can not be summed up to cumulative data when time interpolation. Cumulative
columns (f.ex. FOPT) are time-interpolated linearly. See the `documentation on
fmu-ensemble
<https://equinor.github.io/fmu-ensemble/usage.html#rate-handling-in-eclipse-summary-vectors>`_
for more details on rate handling.

Columns are selected by a list of strings, where wildcards characters ``?``
(matches exactly one character) and ``*`` (matches zero or more characters) can
be used to select multiple columns.

Column count more than 1000 gives increased probability for problems downstream,
depending on which applications are put into use. Column count depends on the
combination of wildcards used in this workflow and the actual vectors that are
requested in the Eclipse DATA file. A wildcard like ``W*`` can in certain cases
(e.g. Eclipse simulations with 100+ wells) produce thousands of vectors, and can
then be replaced by something more explicit like ``WOPT* WGPT* WWPT*``.
"""  # noqa

EXAMPLES = """
Example
-------

Add a file named e.g. ``ert/bin/workflows/QC_CSVEXPORT2`` with the contents::

  MAKE_DIRECTORY <CASEDIR>/share/summary/
  EXPORT_RUNPATH * | *
  CSV_EXPORT2 <RUNPATH_FILE> <CASEDIR>/share/summary/<CASE>.csv monthly F* W* TCPU TIMESTEP

(where ``<CASEDIR>`` typically points to ``/scratch/..``). Adjust all three
lines to your needs.

``EXPORT_RUNPATH`` in the workflow file is added to ensure all realizations and
all iterations are included in the RUNPATH file.  If you have rerun only a
subset of your ensemble, the RUNPATH file will only contain those unless this
statement is included.

Add to your ERT config to have the workflow automatically executed on successful
runs::

  LOAD_WORKFLOW ../bin/workflows/QC_CSVEXPORT2
  HOOK_WORKFLOW QC_CSVEXPORT2 POST_SIMULATION

"""  # noqa


def csv_exporter(
    runpathfile: str,
    time_index: str,
    outputfile: str,
    column_keys: Optional[List[str]] = None,
) -> None:
    """Export CSV data (summary and parameters) from an EnsembleSet

    The EnsembleSet is described by a runpathfile which must exists
    and point to realizations"""
    ensemble_set = ensemble.EnsembleSet(
        name="ERT EnsembleSet for CSV_EXPORT2", runpathfile=runpathfile
    )
    try:
        summary = ensemble_set.load_smry(time_index=time_index, column_keys=column_keys)
        parameters = ensemble_set.parameters
    except KeyError as exc:
        raise UserWarning("No data found") from exc

    if not parameters.empty:
        pd.merge(summary, parameters).to_csv(outputfile, index=False)
    else:
        summary.to_csv(outputfile, index=False)


def main(args: Sequence[str]) -> None:
    parser = csv_export_parser()
    namespace = parser.parse_args(args)

    csv_exporter(
        runpathfile=namespace.runpathfile,
        time_index=namespace.time_index,
        outputfile=namespace.outputfile,
        column_keys=namespace.column_keys,
    )

    print(f"{namespace.time_index} csv-export written to {namespace.outputfile}")


def csv_export_parser() -> argparse.ArgumentParser:
    """Setup parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runpathfile",
        type=str,
        help=(
            "Path to ERT RUNPATH-file, "
            "usually the ERT magic variable <RUNPATH_FILE> can be used"
        ),
    )
    parser.add_argument(
        "outputfile",
        type=str,
        help="Path to CSV file to be written. The directory pointed to must exist.",
    )
    parser.add_argument(
        "time_index",
        type=str,
        default="monthly",
        help=(
            "Time interval specifier for the output. "
            "This argument is passed on to fmu-ensemble, "
            "supported specifiers are 'raw', 'daily', 'weekly', 'monthly' and 'yearly'"
        ),
    )
    parser.add_argument(
        "column_keys", nargs="+", default=None, help="List of summary vector wildcards"
    )
    return parser


class CsvExport2Job(ErtScript):
    description = DESCRIPTION
    examples = EXAMPLES
    parser = csv_export_parser

    def run(self, *args: Any, **_: Any) -> None:
        main(args)


def cli() -> None:
    main(sys.argv[1:])
