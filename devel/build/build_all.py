#!/usr/bin/python
import os
import glob
import sys
import re

def load_path_list(path_config):
    path_list = []
    path_dict = {}
    fileH = open(path_config)
    for line in fileH.readlines():
        comment_start = line.find("#")
        if comment_start > -1:
            line = line[:comment_start]

        if line.find("=") > -1:
            (var , path) = line.split("=")[0:2]
            path_list.append(path.strip())
            path_dict[var.strip()] = path.strip()
    return (path_list , path_dict)



(path_list , path_dict) = load_path_list("path_config")

for path in path_list:
    print path
    cwd = os.getcwd()
    if os.path.exists("%s/src/makefile" % path):
        os.chdir("%s/src" % path)
        os.system("make -s clean")
        nCPU = 4
        if path.find("sample") != -1:
            nCPU = 1
        elif path.find("analysis") != -1:
            nCPU = 1
        print "make -s -j %d MFLAG=m64" % (nCPU)
        os.system("make -s -j %d MFLAG=m64" % (nCPU)) 
        os.chdir(cwd)


