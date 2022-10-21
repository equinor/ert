#!/usr/bin/env python
import os
import sys

iens = None
if len(sys.argv) > 1:
    iens = int(sys.argv[1])

numbers = [1, 2, 3, 4, 5, 6]
if iens in numbers:
    random_report_step = (numbers.index(iens) % 3) + 1
    os.remove(f"perlin_{random_report_step}.txt")

    with open("perlin_fail.status", "w", encoding="utf-8") as f:
        f.write(f"Deleted report step: {random_report_step}")

else:
    with open("perlin_fail.status", "w", encoding="utf-8") as f:
        f.write("Did nothing!")
