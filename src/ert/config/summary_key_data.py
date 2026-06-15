from dataclasses import dataclass
from functools import partial
from typing import assert_never

from resfo_utilities import SummaryKeyType


class InvalidSummaryKeyError(ValueError):
    pass


@dataclass
class SummaryKeyData:
    keyword: str
    number: int | None = None
    well: str | None = None
    name: str | None = None
    i: int | None = None
    j: int | None = None
    k: int | None = None
    lgr_name: str | None = None
    region1: int | None = None
    region2: int | None = None


def make_summary_key_data(summary_key: str) -> SummaryKeyData:
    fields = summary_key.split(":")
    summary_variable = fields[0]
    skd = partial(SummaryKeyData, keyword=summary_variable)
    try:  # noqa: PLW0717
        match SummaryKeyType.from_variable(summary_variable):
            case SummaryKeyType.FIELD | SummaryKeyType.OTHER:
                return skd()
            case (
                SummaryKeyType.REGION | SummaryKeyType.AQUIFER | SummaryKeyType.NETWORK
            ):
                return skd(number=int(fields[1]))
            case SummaryKeyType.BLOCK:
                i, j, k = fields[1].split(",")
                return skd(i=int(i), j=int(j), k=int(k))
            case SummaryKeyType.WELL:
                return skd(well=fields[1])
            case SummaryKeyType.GROUP:
                return skd(name=fields[1])
            case SummaryKeyType.SEGMENT:
                return skd(name=fields[1], number=int(fields[2]))
            case SummaryKeyType.COMPLETION:
                i, j, k = fields[2].split(",")
                return skd(name=fields[1], i=int(i), j=int(j), k=int(k))
            case SummaryKeyType.INTER_REGION:
                r1, r2 = fields[1].split("-")
                return skd(region1=int(r1), region2=int(r2))
            case SummaryKeyType.LOCAL_WELL:
                return skd(lgr_name=fields[1], name=fields[2])
            case SummaryKeyType.LOCAL_BLOCK:
                i, j, k = fields[2].split(",")
                return skd(lgr_name=fields[1], i=int(i), j=int(j), k=int(k))
            case SummaryKeyType.LOCAL_COMPLETION:
                i, j, k = fields[3].split(",")
                return skd(
                    lgr_name=fields[1], name=fields[2], i=int(i), j=int(j), k=int(k)
                )
            case default:
                assert_never(default)
    except Exception as err:
        raise InvalidSummaryKeyError(f"Invalid summary key {summary_key}") from err
