#!/usr/bin/env python
"""A mocked reservoir simulator that can produce Eclipse
summary output, based on parametrized input controlling stochasticity"""
from datetime import datetime
from pathlib import Path
from typing import Dict
from ecl.summary import EclSum
from res.test.synthesizer import OilSimulator


def global_index(i: int, j: int, k: int, nx: int = 10, ny: int = 10) -> int:
    """Compute a scalar index from a three-tupled index defining a cell
    coordinate"""
    # pylint: disable=invalid-name
    return i + nx * (j - 1) + nx * ny * (k - 1)


def read_parameters(filename: str) -> Dict[str, float]:
    """Parse a text file with key:values into a dict"""
    params = {}
    for line in Path(filename).read_text(encoding="utf8").splitlines():
        key, value = line.split(":", 1)
        params[key] = float(value.strip())
    return params


def run_simulator(
    simulator: OilSimulator, history_simulator: OilSimulator, time_step_count: int
) -> EclSum:
    """@rtype: EclSum"""
    ecl_sum = EclSum.writer("SNAKE_OIL_FIELD", datetime(2010, 1, 1), 10, 10, 10)

    ecl_vectors = [
        "OPT",
        "OPR",
        "GPT",
        "GPR",
        "WPT",
        "WPR",
        "GOR",
        "WCT",
    ]

    # Field vectors ("F"-prefix):
    for ecl_vector in ecl_vectors:
        ecl_sum.addVariable("F" + ecl_vector)
        ecl_sum.addVariable("F" + ecl_vector + "H")
        # "H" suffix is for history vectors.

    # Well vectors ("W"-prefix)
    wells = ["OP1", "OP2"]
    for ecl_vector in ecl_vectors:
        for wellname in wells:
            ecl_sum.addVariable("W" + ecl_vector, wgname=wellname)
            ecl_sum.addVariable("W" + ecl_vector + "H", wgname=wellname)

    # Block pressure:
    ecl_sum.addVariable("BPR", num=global_index(5, 5, 5))
    ecl_sum.addVariable("BPR", num=global_index(1, 3, 8))

    mini_step_count = 10
    total_step_count = time_step_count * mini_step_count

    for report_step in range(time_step_count):
        for mini_step in range(mini_step_count):
            t_step = ecl_sum.addTStep(
                report_step + 1, sim_days=report_step * mini_step_count + mini_step
            )
            simulator.step(scale=1.0 / total_step_count)
            history_simulator.step(scale=1.0 / total_step_count)

            t_step["FOPR"] = simulator.fopr()
            t_step["FOPT"] = simulator.fopt()
            t_step["FGPR"] = simulator.fgpr()
            t_step["FGPT"] = simulator.fgpt()
            t_step["FWPR"] = simulator.fwpr()
            t_step["FWPT"] = simulator.fwpt()
            t_step["FGOR"] = simulator.fgor()
            t_step["FWCT"] = simulator.fwct()

            t_step["WOPR:OP1"] = simulator.opr("OP1")
            t_step["WOPR:OP2"] = simulator.opr("OP2")

            t_step["WGPR:OP1"] = simulator.gpr("OP1")
            t_step["WGPR:OP2"] = simulator.gpr("OP2")

            t_step["WWPR:OP1"] = simulator.wpr("OP1")
            t_step["WWPR:OP2"] = simulator.wpr("OP2")

            t_step["WGOR:OP1"] = simulator.gor("OP1")
            t_step["WGOR:OP2"] = simulator.gor("OP2")

            t_step["WWCT:OP1"] = simulator.wct("OP1")
            t_step["WWCT:OP2"] = simulator.wct("OP2")

            t_step["BPR:5,5,5"] = simulator.bpr("5,5,5")
            t_step["BPR:1,3,8"] = simulator.bpr("1,3,8")

            t_step["FOPRH"] = history_simulator.fopr()
            t_step["FOPTH"] = history_simulator.fopt()
            t_step["FGPRH"] = history_simulator.fgpr()
            t_step["FGPTH"] = history_simulator.fgpt()
            t_step["FWPRH"] = history_simulator.fwpr()
            t_step["FWPTH"] = history_simulator.fwpt()
            t_step["FGORH"] = history_simulator.fgor()
            t_step["FWCTH"] = history_simulator.fwct()

            t_step["WOPRH:OP1"] = history_simulator.opr("OP1")
            t_step["WOPRH:OP2"] = history_simulator.opr("OP2")

            t_step["WGPRH:OP1"] = history_simulator.gpr("OP1")
            t_step["WGPRH:OP2"] = history_simulator.gpr("OP2")

            t_step["WWPRH:OP1"] = history_simulator.wpr("OP1")
            t_step["WWPRH:OP2"] = history_simulator.wpr("OP2")

            t_step["WGORH:OP1"] = history_simulator.gor("OP1")
            t_step["WGORH:OP2"] = history_simulator.gor("OP2")

            t_step["WWCTH:OP1"] = history_simulator.wct("OP1")
            t_step["WWCTH:OP2"] = history_simulator.wct("OP2")

    return ecl_sum


def main(
    seedfile: str = "seed.txt", parameters_file: str = "snake_oil_params.txt"
) -> None:
    """Parse input and run the snake oil simulator, producing mocked Eclipse
    binary output file (SMSPEC + UNSMRY)"""
    seed = int(read_parameters(seedfile)["SEED"])
    parameters = read_parameters(parameters_file)

    op1_divergence_scale = parameters["OP1_DIVERGENCE_SCALE"]
    op2_divergence_scale = parameters["OP2_DIVERGENCE_SCALE"]
    op1_persistence = parameters["OP1_PERSISTENCE"]
    op2_persistence = parameters["OP2_PERSISTENCE"]
    op1_offset = parameters["OP1_OFFSET"]
    op2_offset = parameters["OP2_OFFSET"]
    bpr_138_persistence = parameters["BPR_138_PERSISTENCE"]
    bpr_555_persistence = parameters["BPR_555_PERSISTENCE"]

    op1_octaves = int(round(parameters["OP1_OCTAVES"]))
    op2_octaves = int(round(parameters["OP2_OCTAVES"]))

    simulator = OilSimulator()
    simulator.addWell(
        "OP1",
        seed * 997,
        persistence=op1_persistence,
        octaves=op1_octaves,
        divergence_scale=op1_divergence_scale,
        offset=op1_offset,
    )
    simulator.addWell(
        "OP2",
        seed * 13,
        persistence=op2_persistence,
        octaves=op2_octaves,
        divergence_scale=op2_divergence_scale,
        offset=op2_offset,
    )
    simulator.addBlock("5,5,5", seed * 37, persistence=bpr_555_persistence)
    simulator.addBlock("1,3,8", seed * 31, persistence=bpr_138_persistence)

    history_simulator = OilSimulator()
    history_simulator.addWell("OP1", 222118781)
    history_simulator.addWell("OP2", 118116362)

    report_step_count = 200
    ecl_sum = run_simulator(simulator, history_simulator, report_step_count)
    ecl_sum.fwrite()


if __name__ == "__main__":
    main()
