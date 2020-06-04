#!/usr/bin/env python

import json
import sys

if __name__ == '__main__':
    param_file = sys.argv[1]
    json_file = sys.argv[2]

    with open(param_file) as f:
        data = [float(line) for line in f.readlines()]

    json_data = dict(zip(["a", "b", "c"], data))
    with open(json_file, "w") as f:
        json.dump(json_data, f)
