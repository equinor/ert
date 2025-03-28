#!/usr/bin/env python
import datetime

import numpy as np
import resfo
from oil_reservoir_synthesizer import OilSimulator


def globalIndex(i, j, k, nx=10, ny=10, nz=10):
    return i + nx * (j - 1) + nx * ny * (k - 1)


def readParameters(filename):
    params = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            key, value = line.split(":", 1)
            params[key] = value.strip()

    return params


def write_summary_spec(file, keywords, names, units, nums):
    content = [
        ("INTEHEAD", np.array([1, 100], dtype=np.int32)),
        ("RESTART ", [b"        "] * 8),
        ("DIMENS  ", np.array([1 + len(keywords), 10, 10, 10, 0, -1], dtype=np.int32)),
        ("KEYWORDS", [f"{x: <8}" for x in (["TIME", *keywords])]),
        ("WGNAMES ", [b":+:+:+:+", *names]),
        ("NUMS    ", np.array([-32676, *nums], dtype=np.int32)),
        ("UNITS   ", [f"{x: <8}" for x in (["DAYS", *units])]),
        ("STARTDAT", np.array([1, 1, 2010, 0, 0, 0], dtype=np.int32)),
    ]
    resfo.write(file, content)


def write_summary_data(
    file, simulator, history_simulator, time_step_count, mini_step_count=10
):
    time_map = []
    total_step_count = time_step_count * mini_step_count
    start_date = datetime.date(2010, 1, 1)

    def content_generator():
        for report_step in range(time_step_count):
            yield "SEQHDR  ", np.array([0], dtype=np.int32)
            for mini_step in range(mini_step_count):
                simulator.step(scale=1.0 / total_step_count)
                history_simulator.step(scale=1.0 / total_step_count)

                step = report_step * mini_step_count + mini_step
                day = float(step)
                time_map.append(
                    (start_date + datetime.timedelta(days=day)).strftime("%Y-%m-%d")
                )
                values = [
                    simulator.fopt(),
                    simulator.fopr(),
                    simulator.fgpt(),
                    simulator.fgpr(),
                    simulator.fwpt(),
                    simulator.fwpr(),
                    simulator.fgor(),
                    simulator.fwct(),
                    simulator.foip(),
                    simulator.fgip(),
                    simulator.fwip(),
                    history_simulator.fopt(),
                    history_simulator.fopr(),
                    history_simulator.fgpt(),
                    history_simulator.fgpr(),
                    history_simulator.fwpt(),
                    history_simulator.fwpr(),
                    history_simulator.fgor(),
                    history_simulator.fwct(),
                    history_simulator.foip(),
                    history_simulator.fgip(),
                    history_simulator.fwip(),
                    simulator.opr("OP1"),
                    simulator.opr("OP2"),
                    simulator.wpr("OP1"),
                    simulator.wpr("OP2"),
                    simulator.gpr("OP1"),
                    simulator.gpr("OP2"),
                    simulator.gor("OP1"),
                    simulator.gor("OP2"),
                    simulator.wct("OP1"),
                    simulator.wct("OP2"),
                    history_simulator.opr("OP1"),
                    history_simulator.opr("OP2"),
                    history_simulator.wpr("OP1"),
                    history_simulator.wpr("OP2"),
                    history_simulator.gpr("OP1"),
                    history_simulator.gpr("OP2"),
                    history_simulator.gor("OP1"),
                    history_simulator.gor("OP2"),
                    history_simulator.wct("OP1"),
                    history_simulator.wct("OP2"),
                    simulator.bpr("5,5,5"),
                    simulator.bpr("1,3,8"),
                ]
                yield "MINISTEP", np.array([step], dtype=np.int32)
                yield "PARAMS  ", np.array([day, *values], dtype=np.float32)

    resfo.write(file, content_generator())
    print("Wrote summary data")
    return time_map


def runSimulator(simulator, history_simulator, time_step_count):
    write_summary_spec(
        "SNAKE_OIL_FIELD.SMSPEC",
        *zip(
            *[
                ("FOPT", "", "SM3", 0),
                ("FOPR", "", "SM3/DAY", 0),
                ("FGPT", "", "SM3", 0),
                ("FGPR", "", "SM3/DAY", 0),
                ("FWPT", "", "SM3", 0),
                ("FWPR", "", "SM3/DAY", 0),
                ("FGOR", "", "SM3/SM3", 0),
                ("FWCT", "", "SM3/SM3", 0),
                ("FOIP", "", "SM3", 0),
                ("FGIP", "", "SM3", 0),
                ("FWIP", "", "SM3", 0),
                ("FOPTH", "", "SM3", 0),
                ("FOPRH", "", "SM3/DAY", 0),
                ("FGPTH", "", "SM3", 0),
                ("FGPRH", "", "SM3/DAY", 0),
                ("FWPTH", "", "SM3", 0),
                ("FWPRH", "", "SM3/DAY", 0),
                ("FGORH", "", "SM3/SM3", 0),
                ("FWCTH", "", "SM3/SM3", 0),
                ("FOIPH", "", "SM3", 0),
                ("FGIPH", "", "SM3", 0),
                ("FWIPH", "", "SM3", 0),
                ("WOPR", "OP1", "SM3/DAY", 0),
                ("WOPR", "OP2", "SM3/DAY", 0),
                ("WWPR", "OP1", "SM3/DAY", 0),
                ("WWPR", "OP2", "SM3/DAY", 0),
                ("WGPR", "OP1", "SM3/DAY", 0),
                ("WGPR", "OP2", "SM3/DAY", 0),
                ("WGOR", "OP1", "SM3/SM3", 0),
                ("WGOR", "OP2", "SM3/SM3", 0),
                ("WWCT", "OP1", "SM3/SM3", 0),
                ("WWCT", "OP2", "SM3/SM3", 0),
                ("WOPRH", "OP1", "SM3/DAY", 0),
                ("WOPRH", "OP2", "SM3/DAY", 0),
                ("WWPRH", "OP1", "SM3/DAY", 0),
                ("WWPRH", "OP2", "SM3/DAY", 0),
                ("WGPRH", "OP1", "SM3/DAY", 0),
                ("WGPRH", "OP2", "SM3/DAY", 0),
                ("WGORH", "OP1", "SM3/SM3", 0),
                ("WGORH", "OP2", "SM3/SM3", 0),
                ("WWCTH", "OP1", "SM3/SM3", 0),
                ("WWCTH", "OP2", "SM3/SM3", 0),
                ("BPR", "", "BARSA", globalIndex(5, 5, 5)),
                ("BPR", "", "BARSA", globalIndex(1, 3, 8)),
            ],
            strict=False,
        ),
    )
    return write_summary_data(
        "SNAKE_OIL_FIELD.UNSMRY",
        simulator,
        history_simulator,
        time_step_count=time_step_count,
    )


def roundedInt(value):
    return round(float(value))


if __name__ == "__main__":
    seed = int(readParameters("seed.txt")["SEED"])
    parameters = readParameters("snake_oil_params.txt")

    op1_divergence_scale = float(parameters["OP1_DIVERGENCE_SCALE"])
    op2_divergence_scale = float(parameters["OP2_DIVERGENCE_SCALE"])
    op1_persistence = float(parameters["OP1_PERSISTENCE"])
    op2_persistence = float(parameters["OP2_PERSISTENCE"])
    op1_offset = float(parameters["OP1_OFFSET"])
    op2_offset = float(parameters["OP2_OFFSET"])
    bpr_138_persistence = float(parameters["BPR_138_PERSISTENCE"])
    bpr_555_persistence = float(parameters["BPR_555_PERSISTENCE"])

    op1_octaves = roundedInt(parameters["OP1_OCTAVES"])
    op2_octaves = roundedInt(parameters["OP2_OCTAVES"])

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
    time_map = runSimulator(simulator, history_simulator, report_step_count)

    with open("time_map.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{t}\n" for t in time_map)
