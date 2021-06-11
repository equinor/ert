#!/usr/bin/env python
from ecl.summary import EclSum


def writeDiff(filename, ecl_sum, key1, key2):
    with open(filename, "w") as f:
        for v1, v2 in zip(ecl_sum.numpy_vector(key1), ecl_sum.numpy_vector(key2)):
            diff = v1 - v2
            f.write("%f\n" % diff)


if __name__ == "__main__":
    ecl_sum = EclSum("SNAKE_OIL_FIELD")

    report_step = 199
    writeDiff(
        "snake_oil_opr_diff_%d.txt" % report_step, ecl_sum, "WOPR:OP1", "WOPR:OP2"
    )
    writeDiff(
        "snake_oil_wpr_diff_%d.txt" % report_step, ecl_sum, "WWPR:OP1", "WWPR:OP2"
    )
    writeDiff(
        "snake_oil_gpr_diff_%d.txt" % report_step, ecl_sum, "WGPR:OP1", "WGPR:OP2"
    )
