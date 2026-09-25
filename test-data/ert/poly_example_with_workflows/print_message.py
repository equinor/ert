#!/usr/bin/env python3
import sys

if __name__ == "__main__":
    hook = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
    print(f"Running workflow hooked to {hook}")
