#!/usr/bin/python
import os
import glob
import sys

def load_path_list(path_config):
    path_list = []
    fileH = open(path_config)
    for line in fileH.readlines():
        if line.find("=") > -1:
            path = line.split("=")[1]
            path_list.append(path.strip())
    return path_list



path_list = load_path_list("path_config")
for path in path_list:
    cwd = os.getcwd()
    if os.path.exists("%s/src/makefile" % path):
        os.chdir("%s/src" % path)
        os.system("make -j 4")
