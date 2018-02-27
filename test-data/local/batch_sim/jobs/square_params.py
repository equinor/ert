#!/usr/bin/env python
import sys
import json
import random
import time

keys = ["W1", "W2", "W3"]

def copy_file(input_file, output_file):
    data = json.load(open(input_file))

    with open(output_file, "w") as f:
        for key in keys:
            sq = data[key] * data[key]
            f.write("%g\n" % sq)

    time.sleep(random.randint(0,4))

if __name__ == "__main__":
    copy_file("WELL_ORDER.json", "ORDER_0")
    copy_file("WELL_ON_OFF.json", "ON_OFF_0")





