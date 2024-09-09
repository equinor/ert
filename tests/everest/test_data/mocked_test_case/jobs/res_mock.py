#!/usr/bin/env python
import sys

from resdata.util.test.mock import createSummary


def dump_summary_mock(case_name):
    case = createSummary(
        case_name, [("FOPT", None, 0, "SM3"), ("FOPR", None, 0, "SM3/DAY")]
    )
    case.fwrite()


if __name__ == "__main__":
    name = sys.argv[1]
    dump_summary_mock(name)
