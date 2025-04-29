#!/usr/bin/env python
import sys

sum_mock = {}
sum_mock["FOPT"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
sum_mock["FWPT"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
sum_mock["FGPT"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
sum_mock["FWIT"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
sum_mock["TIME"] = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
]


def compute_npv(summary):
    fopt = summary.get("FOPT")
    fwpt = summary.get("FWPT")
    fgpt = summary.get("FGPT")
    fwit = summary.get("FWIT")
    elapsedtime = summary.get("TIME")

    with open("debug.txt", "w", encoding="utf-8") as f:
        DCF = compute_dcf(fopt[0], fwpt[0], fgpt[0], fwit[0], elapsedtime[0])
        f.write(
            f"{elapsedtime[0]:f}\t{fopt[0]:f}\t{fwpt[0]:f}\t{fgpt[0]:f}\t{fwit[0]:f}\t{DCF:f}\n"
        )
        NPV = DCF

        for i in range(1, len(fopt)):
            DCF = compute_dcf(
                fopt[i] - fopt[i - 1],
                fwpt[i] - fwpt[i - 1],
                fgpt[i] - fgpt[i - 1],
                fwit[i] - fwit[i - 1],
                elapsedtime[i],
            )
            NPV += DCF
            f.write(
                f"{elapsedtime[i]:f}\t{fopt[i]:f}\t{fwpt[i]:f}\t{fgpt[i]:f}\t{fwit[i]:f}\t{DCF:f}\n"
            )

    return NPV


def compute_dcf(voilp, vwaterp, vgasp, vwateri, elapsedtime):
    """
    Function for computing the discounted cash flow
    """
    OILPRICE = 150
    WATERPRICE = -25
    GASPRICE = 0.000001
    WATERINJPRICE = -5
    DISCOUNTRATE = 0.08

    return (
        voilp * OILPRICE
        + vwaterp * WATERPRICE
        + vgasp * GASPRICE
        + vwateri * WATERINJPRICE
    ) / pow((1 + DISCOUNTRATE), (elapsedtime / 365.25))


def save_object_value(object_value, target_file):
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(f"{object_value:g} \n")


def main(argv):
    # start main script
    target_file = sys.argv[2]

    npv_value = compute_npv(sum_mock)
    save_object_value(npv_value, target_file)


if __name__ == "__main__":
    main(sys.argv[1:])
