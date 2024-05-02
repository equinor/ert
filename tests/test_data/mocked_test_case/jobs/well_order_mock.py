#!/usr/bin/env python
import json
import re
import sys
from collections import OrderedDict


def create_file(order, template_file, target_file):
    writeH = open(target_file, "w", encoding="utf-8")
    readH = open(template_file)

    for line in readH.readlines():
        match_obj = re.search("(__[A-Z]+_[0-9]__)", line)
        if match_obj:
            new_well = order.popitem(False)[0]
            line = line.replace(match_obj.group(1), new_well)
        writeH.write(line)

    readH.close()
    writeH.close()


def well_value(w):
    return -w[1]


def load_well_order(well_order_file):
    well_list = []
    with open(well_order_file) as f:
        for line in f.readlines():
            well, rest = line.split()
            well_list.append((well, float(rest)))

    well_list.sort(key=well_value)
    well_order = OrderedDict()
    index = 0
    for well, value in well_list:
        well_order[well] = value
        index += 1
    return well_order


def create_well_order(well_list_file, well_order_file):
    writeH = open(well_order_file, "w", encoding="utf-8")
    well_list = []
    with open(well_list_file) as f:
        data = json.load(f)

    for well in data:
        well_list.append((well, float(data[well])))

    well_list.sort(key=well_value)
    for well, value in well_list:
        writeH.write(well + "\n")
    writeH.close()


def main(argv):
    # schedule_inc
    target_file = sys.argv[2]

    # Commented out since this is not clear anymore - should be correct later
    # create_well_order(well_order_file, ordered_wells)
    # well_order = load_well_order(well_order_file)
    # create_file(well_order, template_file, target_file)

    # just create the target file to get the forward model going
    with open(target_file, "w", encoding="utf-8"):
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
