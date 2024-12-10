#!/usr/bin/env python
from datetime import datetime

from oil_reservoir_synthesizer import OilSimulator
from resdata.summary import Summary


def globalIndex(i, j, k, nx=10, ny=10):
    return i + nx * (j - 1) + nx * ny * (k - 1)


def read_seed(filename):
    params = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            key, value = line.split(":")
            params[key] = value.strip()

    return params


def read_parameters(filename):
    params = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            key, value = line.split(" ")
            _, name = key.split(":")
            params[name] = value.strip()

    return params


def runSimulator(simulator, history_simulator, time_step_count):
    summary = Summary.writer("SNAKE_OIL_FIELD", datetime(2010, 1, 1), 10, 10, 10)

    summary.add_variable("FOPT")
    summary.add_variable("FOPR")
    summary.add_variable("FGPT")
    summary.add_variable("FGPR")
    summary.add_variable("FWPT")
    summary.add_variable("FWPR")
    summary.add_variable("FGOR")
    summary.add_variable("FWCT")

    summary.add_variable("FOPTH")
    summary.add_variable("FOPRH")
    summary.add_variable("FGPTH")
    summary.add_variable("FGPRH")
    summary.add_variable("FWPTH")
    summary.add_variable("FWPRH")
    summary.add_variable("FGORH")
    summary.add_variable("FWCTH")

    summary.add_variable("WOPR", wgname="OP1")
    summary.add_variable("WOPR", wgname="OP2")
    summary.add_variable("WWPR", wgname="OP1")
    summary.add_variable("WWPR", wgname="OP2")
    summary.add_variable("WGPR", wgname="OP1")
    summary.add_variable("WGPR", wgname="OP2")
    summary.add_variable("WGOR", wgname="OP1")
    summary.add_variable("WGOR", wgname="OP2")
    summary.add_variable("WWCT", wgname="OP1")
    summary.add_variable("WWCT", wgname="OP2")

    summary.add_variable("WOPRH", wgname="OP1")
    summary.add_variable("WOPRH", wgname="OP2")
    summary.add_variable("WWPRH", wgname="OP1")
    summary.add_variable("WWPRH", wgname="OP2")
    summary.add_variable("WGPRH", wgname="OP1")
    summary.add_variable("WGPRH", wgname="OP2")
    summary.add_variable("WGORH", wgname="OP1")
    summary.add_variable("WGORH", wgname="OP2")
    summary.add_variable("WWCTH", wgname="OP1")
    summary.add_variable("WWCTH", wgname="OP2")

    summary.add_variable("BPR", num=globalIndex(5, 5, 5))
    summary.add_variable("BPR", num=globalIndex(1, 3, 8))

    time_map = []
    mini_step_count = 10
    total_step_count = time_step_count * mini_step_count

    for report_step in range(time_step_count):
        for mini_step in range(mini_step_count):
            t_step = summary.add_t_step(
                report_step + 1, sim_days=report_step * mini_step_count + mini_step
            )

            time_map.append(t_step.get_sim_time().datetime().strftime("%d/%m/%Y"))

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

    return summary, time_map


def roundedInt(value):
    return round(float(value))


if __name__ == "__main__":
    seed = int(read_seed("seed.txt")["SEED"])
    parameters = read_parameters("parameters.txt")

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
    ecl_sum, time_map = runSimulator(simulator, history_simulator, report_step_count)

    ecl_sum.fwrite()

    with open("time_map.txt", "w", encoding="utf-8") as filehandle:
        for timestep_string in time_map:
            filehandle.write(f"{timestep_string}\n")
