#!/usr/bin/env python
from resdata.summary import Summary


def writeDiff(filename, vector1, vector2):
    with open(filename, "w", encoding="utf-8") as f:
        for index in range(len(vector1)):
            node1 = vector1[index]
            node2 = vector2[index]

            diff = node1.value - node2.value
            f.write("%f\n" % diff)


if __name__ == "__main__":
    summary = Summary("SNAKE_OIL_FIELD")

    report_step = 199
    writeDiff(
        "snake_oil_opr_diff_%d.txt" % report_step,
        summary["WOPR:OP1"],
        summary["WOPR:OP2"],
    )
    writeDiff(
        "snake_oil_wpr_diff_%d.txt" % report_step,
        summary["WWPR:OP1"],
        summary["WWPR:OP2"],
    )
    writeDiff(
        "snake_oil_gpr_diff_%d.txt" % report_step,
        summary["WGPR:OP1"],
        summary["WGPR:OP2"],
    )
