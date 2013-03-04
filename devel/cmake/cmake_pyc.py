#!/usr/bin/env python
import py_compile
import os
import os.path
import sys

def compile_file(src_file , target_file):
    path = os.path.dirname( target_file )
    if not os.path.exists( path ):
        os.makedirs( path )
    try:
        py_compile.compile( src_file , cfile = target_file , doraise = True)
    except Exception,error:
        sys.exit(1)


target_path = sys.argv[-1]
for src_file in sys.argv[1:-1]:
    compile_file( src_file , "%s/%sc" % (target_path , os.path.basename(src_file)))

sys.exit(0)

                   

        
