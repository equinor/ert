#!/usr/bin/env python
from resdata.summary import Summary


def writeDiff(filename, vector1, vector2):
    with open(filename, "w", encoding="utf-8") as f:
        for node1, node2 in zip(vector1, vector2):
            f.write(f"{node1-node2:f}\n")


if __name__ == "__main__":
    ecl_sum = Summary("SNAKE_OIL_FIELD")

    report_step = 199
    writeDiff(
        f"snake_oil_opr_diff_{report_step}.txt",
        ecl_sum.numpy_vector("WOPR:OP1"),
        ecl_sum.numpy_vector("WOPR:OP2"),
    )
    writeDiff(
        f"snake_oil_wpr_diff_{report_step}.txt",
        ecl_sum.numpy_vector("WWPR:OP1"),
        ecl_sum.numpy_vector("WWPR:OP2"),
    )
    writeDiff(
        f"snake_oil_gpr_diff_{report_step}.txt",
        ecl_sum.numpy_vector("WGPR:OP1"),
        ecl_sum.numpy_vector("WGPR:OP2"),
    )
