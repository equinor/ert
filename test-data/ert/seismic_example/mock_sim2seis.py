#!/usr/bin/env python
"""fmu-sim2seis mock.

Generates a set of output files needed for ERT integration which are formally consistent
with fmu-sim2seis:

 - synthetic seismic CSV observation files.
 - synthetic seismic CSV modelled data files.
"""

import argparse
import hashlib
import itertools
import json
from enum import StrEnum
from pathlib import Path

import pandas as pd


class FieldName(StrEnum):
    FIELD = "field"
    OTHER_FIELD = "tulip_field"


class Attribute(StrEnum):
    AMPLITUDE = "amplitude"
    RELAI = "relai"


class StackingOffset(StrEnum):
    FULL = "full"
    NEAR = "near"
    FAR = "far"


class Calculation(StrEnum):
    MEAN = "mean"
    MIN = "min"


class VerticalDomain(StrEnum):
    DEPTH = "depth"
    TIME = "time"


class BaseDate(StrEnum):
    JAN2024 = "20240101"


class MonitorDate(StrEnum):
    JAN2025 = "20250101"
    JAN2026 = "20260101"
    JAN2030 = "20300101"


# Coordinate/region table: (X_UTME, Y_UTMN, REGION)
COORDINATE_TABLE: list[tuple[float, float, float]] = [
    (463401.665023891, 6929758.90312445, 1.0),
    (463312.374851203, 6929712.58234601, 1.0),
    (463245.488743621, 6929689.04095157, 1.0),
    (463198.920134567, 6929645.33178924, 1.0),
    (463156.233987412, 6929601.72456830, 1.0),
    (463567.123456789, 6929847.23456789, 1.0),
    (463478.891234567, 6929803.45678901, 1.0),
    (463089.547312890, 6929556.81234567, 1.0),
    (463023.456789012, 6929512.34567890, 2.0),
    (463701.234567890, 6929934.56789012, 1.0),
    (463634.789012345, 6929891.01234567, 1.0),
    (462978.901234567, 6929478.90123456, 2.0),
    (462934.567890123, 6929445.67890123, 3.0),
    (463923.456789012, 6930045.67890123, 2.0),
    (463867.901234567, 6930012.34567890, 2.0),
    (463812.345678901, 6929989.89012345, 1.0),
    (463756.890123456, 6929967.12345678, 1.0),
    (462890.123456789, 6929412.45678901, 2.0),
    (462845.678901234, 6929378.23456789, 3.0),
    (462801.234567890, 6929345.01234567, 3.0),
]

N = len(COORDINATE_TABLE)


# Base values, to be modified by the _create_value function.
# fmt: off
BASE_DATA: list[float] = [
     0.0071234567890123456,
    -0.0093335668901534267,
     0.0078901234567890123,
    -0.0089012345678901234,
     0.0021254571230122554,
    -0.0045678901234567890,
     0.0087159011255416512,
    -0.0118305254567295123,
     0.0011000567124015565,
    -0.0056789012345678901,
     0.0033557195122454789,
    -0.0067890123456789012,
     0.0055172311233515201,
    -0.0023456789012345678,
     0.0085011535543575114,
    -0.0005457591501635631,
     0.0058519015151234951,
    -0.0012345678901234567,
     0.0034567890123456789,
    -0.0048502264264879023,
]
# fmt: on
assert len(BASE_DATA) == N


def _string_to_normalized_float(text: str) -> float:
    """Turn a string into a float in [0, 1) using hashing.
    Only the first 8 characters of the hex string are used.
    """
    digest = hashlib.sha256(text.encode()).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def _create_value(
    filename: str,
    value_index: int,
    parameters: dict[str, dict[str, float]],
) -> float:
    """Modify the base value such that it slightly differs for each file (filename
    represents setup parameters combination), each value (value_index) and sampled ert
    parameters (a and b).

    'Calculation' file parameter is expected to be included in the file identification.
    Data row is already mocked to one value, so no separate calculation (mean/min) is
    happening. It means data can be inconsistent (min > mean), but it shouldn't matter.
    """

    a = parameters["a"]["value"]  # [0, 1)
    b = parameters["b"]["value"]  # [0, 2)
    setup_str = f"{filename}:{value_index}"
    setup_value = _string_to_normalized_float(setup_str)  # [0, 1)
    a_contribution = (a - 0.5) * setup_value  # [-0.5, 0.5)
    b_contribution = (b - 1.0) * 0.5  # [-0.5, 0.5)
    blending_factor = 0.02
    offset = (a_contribution + b_contribution) * blending_factor
    return BASE_DATA[value_index] + offset


def _obs_error(obs_value: float) -> float:
    """Return OBS_ERROR, modified ~5% of the time based on the obs value."""
    default_error = 0.005
    normalized_value = _string_to_normalized_float(repr(obs_value))

    non_default_error_chance = 0.05
    if normalized_value >= non_default_error_chance:
        return default_error

    small_perturbation = normalized_value * 0.002
    return default_error + small_perturbation


def _build_filename(
    field: FieldName,
    attribute: Attribute,
    stacking_offset: StackingOffset,
    calculation: Calculation,
    vertical_domain: VerticalDomain,
    base: BaseDate,
    monitor: MonitorDate,
) -> str:
    """Constructs a filename consistent with fmu-sim2seis naming convention."""
    attr_part = (
        f"{attribute.value}_{stacking_offset.value}"
        f"_{calculation.value}_{vertical_domain.value}"
    )
    return f"{field.value}--{attr_part}--{monitor.value}_{base.value}.csv"


def generate_csv(
    parameters: dict[str, dict[str, float]],
    output_dir: Path,
    field: FieldName,
    attribute: Attribute,
    stacking_offset: StackingOffset,
    calculation: Calculation,
    vertical_domain: VerticalDomain,
    base: BaseDate,
    monitor: MonitorDate,
) -> Path:
    """Generate a single seismic CSV file for the given setup parameter combination.

    Both observation file and modelled data file have the same structure, including
    column names.

    All the files in the example have the same number of rows, so we assume it is
    expected.
    """
    filename = _build_filename(
        field, attribute, stacking_offset, calculation, vertical_domain, base, monitor
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    rows = []
    for i, (x, y, region) in enumerate(COORDINATE_TABLE):
        obs = _create_value(filename, i, parameters)
        rows.append(
            {
                "X_UTME": x,
                "Y_UTMN": y,
                "OBS": obs,
                "OBS_ERROR": _obs_error(obs),
                "REGION": region,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Written: {filepath}")
    return filepath


def generate_many(
    parameters: dict[str, dict[str, float]],
    output_dir: Path,
    fields: list[FieldName] | None = None,
    attributes: list[Attribute] | None = None,
    stacking_offsets: list[StackingOffset] | None = None,
    calculations: list[Calculation] | None = None,
    vertical_domains: list[VerticalDomain] | None = None,
    bases: list[BaseDate] | None = None,
    monitors: list[MonitorDate] | None = None,
) -> None:
    """Generate CSV files for every combination of the supplied setup parameter
    lists.
    """
    fields = fields or [FieldName.FIELD]
    attributes = attributes or [Attribute.AMPLITUDE]
    stacking_offsets = stacking_offsets or [StackingOffset.FULL]
    calculations = calculations or [Calculation.MEAN]
    vertical_domains = vertical_domains or [VerticalDomain.DEPTH]
    bases = bases or [BaseDate.JAN2024]
    monitors = monitors or [MonitorDate.JAN2025]

    for combo in itertools.product(
        fields,
        attributes,
        stacking_offsets,
        calculations,
        vertical_domains,
        bases,
        monitors,
    ):
        generate_csv(
            parameters,
            output_dir,
            *combo,
        )


if __name__ == "__main__":
    """Generate .csv files consistent with fmu-sim2seis.

    Run with no parameters to generate modelled data files, or with --observations to
    generate observation files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observations",
        action="store_true",
        help="Generate observation files.",
    )
    args = parser.parse_args()

    if args.observations:
        parameters = {"a": {"value": 0.4}, "b": {"value": 1.25}}
        output_dir = Path("share/preprocessed/tables")
    else:
        parameters = json.loads(Path("parameters.json").read_text(encoding="utf-8"))
        output_dir = Path("share/results/tables")

    generate_many(
        parameters,
        output_dir,
        monitors=[MonitorDate.JAN2025, MonitorDate.JAN2026],
        calculations=[Calculation.MEAN, Calculation.MIN],
    )
