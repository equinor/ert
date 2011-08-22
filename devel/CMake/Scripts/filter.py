#!/usr/bin/python
import sys
import os.path

src_template = sys.argv[1]
target_type  = sys.argv[2]
out_path     = sys.argv[3]

for ext in ["c" , "h"]:
    src = open( src_template + "." + ext , "r")
    target = open( "%s/%s_vector.%s" % (out_path , target_type, ext) , "w")

    for line in src.readlines():
        line = line.replace("<TYPE>" , target_type)
        target.write( line )

    src.close()
    target.close()
