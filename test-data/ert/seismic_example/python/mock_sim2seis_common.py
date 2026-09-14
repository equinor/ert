import itertools
from enum import StrEnum


class HorizonName(StrEnum):
    HORIZON = "horizon"
    OTHER_HORIZON = "fantastic_horizon"
    TOPVOLANTIS = "topvolantis"


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
    JAN2018 = "20180101"
    JAN2024 = "20240101"


class MonitorDate(StrEnum):
    JUL2019 = "20190701"
    JAN2025 = "20250101"
    JAN2026 = "20260101"
    JAN2030 = "20300101"


def build_stem(
    horizon: HorizonName,
    attribute: Attribute,
    stacking_offset: StackingOffset,
    calculation: Calculation,
    vertical_domain: VerticalDomain,
    base: BaseDate,
    monitor: MonitorDate,
) -> str:
    """Constructs a filename stem (no extension) consistent with fmu-sim2seis naming
    convention.
    """
    attr_part = (
        f"{attribute.value}_{stacking_offset.value}"
        f"_{calculation.value}_{vertical_domain.value}"
    )
    return f"{horizon.value}--{attr_part}--{monitor.value}_{base.value}"


def generate_combos(
    horizons: list[HorizonName] | None = None,
    attributes: list[Attribute] | None = None,
    stacking_offsets: list[StackingOffset] | None = None,
    calculations: list[Calculation] | None = None,
    vertical_domains: list[VerticalDomain] | None = None,
    bases: list[BaseDate] | None = None,
    monitors: list[MonitorDate] | None = None,
) -> list[tuple]:
    """Generate all combinations of the supplied setup parameter lists."""
    horizons = horizons or [HorizonName.HORIZON]
    attributes = attributes or [Attribute.AMPLITUDE]
    stacking_offsets = stacking_offsets or [StackingOffset.FULL]
    calculations = calculations or [Calculation.MEAN]
    vertical_domains = vertical_domains or [VerticalDomain.DEPTH]
    bases = bases or [BaseDate.JAN2018]
    monitors = monitors or [MonitorDate.JUL2019]

    return list(
        itertools.product(
            horizons,
            attributes,
            stacking_offsets,
            calculations,
            vertical_domains,
            bases,
            monitors,
        )
    )
