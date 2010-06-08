#!/bin/bash

export QT_EXPERIMENTAL=1 
export PYTHON_EXPERIMENTAL=1 
source /prog/sdpsoft/environment.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/res/x86_64_RH_4/lib

#export ERT_HOME=/private/jpb/EnKF/
export ERT_HOME=/d/proj/bg/enkf/jaskje/ERT_GUI/lib/

ORIGINAL_DIRECTORY=$PWD

export script_dir=$(dirname $0)
export ert_gui_dir=$script_dir/../code

python $script_dir/clean.py

if [ "$1" = "debug" ]
then
    export CONFIG_FILE="$2"
    gdb python --command=$script_dir/gdbcommands
elif [ "$1" = "strace" ]
then
    strace -e trace=file python $ert_gui_dir/main.py "$2"
else
    python $ert_gui_dir/main.py "$1"
fi

cd "$ORIGINAL_DIRECTORY"