#!/usr/bin/env python
from ecl.summary import EclSum


def writeDiff(filename, vector1, vector2):
    with open(filename, "w") as f:
        for index in range(len(vector1)):
            node1 = vector1[index]
            node2 = vector2[index]

            diff = node1 - node2
            f.write("%f\n" % diff)


if __name__ == "__main__":
    ecl_sum = EclSum("SNAKE_OIL_FIELD")

    report_step = 199
    writeDiff(
        "snake_oil_opr_diff_%d.txt" % report_step,
        ecl_sum.numpy_vector("WOPR:OP1"),
        ecl_sum.numpy_vector("WOPR:OP2"),
    )
    writeDiff(
        "snake_oil_wpr_diff_%d.txt" % report_step,
        ecl_sum.numpy_vector("WWPR:OP1"),
        ecl_sum.numpy_vector("WWPR:OP2"),
    )
    writeDiff(
        "snake_oil_gpr_diff_%d.txt" % report_step,
        ecl_sum.numpy_vector("WGPR:OP1"),
        ecl_sum.numpy_vector("WGPR:OP2"),
    )
