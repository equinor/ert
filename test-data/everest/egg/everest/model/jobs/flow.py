#!/usr/bin/env python
import argparse
import os
import subprocess

arg_parser = argparse.ArgumentParser(description="Run a flow job")
arg_parser.add_argument("input", help="Input directory")
arg_parser.add_argument("--flow-path", help="Path to the flow binary", default="flow")
arg_parser.add_argument("--enable-tuning", help="Enable tuning", action="store_true")
options = arg_parser.parse_args()

call_args = [options.flow_path, options.input]
if options.enable_tuning:
    call_args.append("--enable-tuning=1")
flow_process = subprocess.call(call_args, cwd=os.getcwd())

with open("flow.ok", "w", encoding="utf-8"):
    pass
