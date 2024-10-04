#!/usr/bin/env python
import json
import sys


def main(control_group_name):
    with open(control_group_name + ".json", encoding="utf-8") as f:
        control_group = json.load(f)

    with open(control_group_name + "_sum", "w", encoding="utf-8") as f_out:
        f_out.write("%f" % sum(control_group.values()))


if __name__ == "__main__":
    main(sys.argv[1])
