#!/usr/bin/env python
from resdata.summary import Summary


def writeDiff(filename, ecl_sum, key1, key2):
    with open(filename, "w", encoding="utf-8") as f:
        for v1, v2 in zip(ecl_sum.numpy_vector(key1), ecl_sum.numpy_vector(key2)):
            diff = v1 - v2
            f.write(f"{diff}\n")


if __name__ == "__main__":
    summary = Summary("SNAKE_OIL_FIELD")

    report_step = 199
    writeDiff(f"snake_oil_opr_diff_{report_step}.txt", summary, "WOPR:OP1", "WOPR:OP2")
    writeDiff(f"snake_oil_wpr_diff_{report_step}.txt", summary, "WWPR:OP1", "WWPR:OP2")
    writeDiff(f"snake_oil_gpr_diff_{report_step}.txt", summary, "WGPR:OP1", "WGPR:OP2")
