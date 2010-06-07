#!/bin/sh

export QT_EXPERIMENTAL=1 
export PYTHON_EXPERIMENTAL=1 
source /prog/sdpsoft/environment.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/res/x86_64_RH_4/lib

export ERT_HOME=/private/jpb/EnKF/

ORIGINAL_DIRECTORY=$PWD

SCRIPT_DIR="dirname $0"
cd "$SCRIPT_DIR"
cd ../code

python ../bin/clean.py

if [ "$1" = "debug" ]
then
    export CONFIG_FILE="$2"
    gdb python --command=../bin/gdbcommands
elif [ "$1" = "strace" ]
then
    strace -e trace=file python main.py "$2"
else
    python main.py "$1"
fi

cd "$ORIGINAL_DIRECTORY"