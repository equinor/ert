#!/usr/bin/env python

import os


def handle_22():
    print("we're failing!")
    raise RuntimeError("Ahhhhh")


if __name__ == "__main__":
    cwd = os.getcwd()
    print(cwd)
    if "22" in cwd:
        handle_22()
    print("all good")
