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
if not os.path.exists(path_dict["ENKF_BIN_PATH"]):
    os.makedirs(path_dict["ENKF_BIN_PATH"])
cmd = "cp %s/ecl_submit.py %s/ecl_submit" % ("../Scripts" , path_dict["ENKF_BIN_PATH"])
print cmd
os.system(cmd)
if os.path.exists("%s/ecl_env.py" % "../Scripts"):
   cmd = "cp %s/ecl_env.py %s/ecl_env.py" % ("../Scripts" , path_dict["ENKF_BIN_PATH"])
   print "%s\n" % cmd
   os.system(cmd)
else:
   sys.stderr.write("** WARNING : could not find file \"ecl_env.py\" - build might fail to submit eclipse jobs. ** \n")


for path in path_list:
    print path
    cwd = os.getcwd()
    if os.path.exists("%s/src/makefile" % path):
        os.chdir("%s/src" % path)
        for mflag in ["m32" , "m64"]:
           os.system("make -s clean MFLAG=%s" %  mflag)
           nCPU = 4
           if path.find("sample") != -1:
               nCPU = 1
           elif path.find("analysis") != -1:
               nCPU = 1
           print "make -s -j %d MFLAG=%s" % (nCPU , mflag)
           os.system("make -s -j %d MFLAG=%s" % (nCPU , mflag)) 
        os.chdir(cwd)


