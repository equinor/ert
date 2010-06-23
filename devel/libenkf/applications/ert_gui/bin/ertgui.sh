#!/bin/sh
export script_dir=$(dirname $0)

while [ -L "script_dir" ]; do
  script_dir=`readlink -e "$script_dir"`
done

#
# This environment variable must be set to the ERT home directory
# or a lib directory containing all .so files of ERT
#
#export ERT_HOME=/private/jpb/EnKF/
#export ERT_HOME=/d/proj/bg/enkf/jaskje/ERT_GUI/lib/

if [ -z "$ERT_HOME" ]
then
    export ERT_HOME=$script_dir/../lib
fi

if [ -z "$ERT_HOME" ]
then
    echo
    echo "The ERT_HOME environment variable has not been set."
    echo "The GUI for ERT requires the ERT_HOME variable to point to a valid ERT directory."
    echo "A valid ERT directory contains the .so files associated with ERT."
    echo
    exit
fi

if [ -z "$1" ]
then
    echo
    echo "A configuration file must be specified."
    echo "  ertgui ert_configuration_file"
    echo
    echo "Options (for debugging):"
    echo "        ertgui.sh debug ert_configuration_file"
    echo "        ertgui.sh strace ert_configuration_file"
    echo
    exit
fi

#
# setup the SDP environment
#
export QT_EXPERIMENTAL=1
export PYTHON_EXPERIMENTAL=1
source /prog/sdpsoft/environment.sh

#
# Required by ERT
#
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/res/x86_64_RH_4/lib


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