#!/bin/sh

export QT_EXPERIMENTAL=1 
export PYTHON_EXPERIMENTAL=1 
source /prog/sdpsoft/environment.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/res/x86_64_RH_4/lib

cd ../code

if [ "$1" = "debug" ]
then
    gdb python --command=../bin/gdbcommands    
else
    python main.py
fi

#strace python main.py
#gdb python

