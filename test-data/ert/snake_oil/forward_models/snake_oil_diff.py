#!/usr/bin/env python
from pathlib import Path

from resdata.summary import Summary


def writeDiff(filename, vector1, vector2):
    Path(filename).write_text(
        "\n".join(
            f"{node1 - node2:f}" for node1, node2 in zip(vector1, vector2, strict=False)
        ),
        encoding="utf-8",
    )


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
