#!/bin/sh

script_path=$0
while [ -L "$script_path" ]; do
    script_path=`readlink -f "$script_path"`
done

export script_dir=$(dirname $script_path)
#
# This environment variable must be set to the ERT home directory
# or a lib directory containing all .so files of ERT
#

if [ -z "$ERT_LD_PATH" ]
then
    export ERT_LD_PATH=$script_dir/../lib
fi

if [ -z "$ERT_LD_PATH" ]
then
    echo
    echo "The ERT_LD_PATH environment variable has not been set."
    echo "The GUI for ERT requires the ERT_HOME variable to point to a valid ERT directory."
    echo "A valid ERT directory contains the .so files associated with ERT."
    echo
    exit
fi
echo "Loading ERT shared libraries from:" $ERT_LD_PATH 


if [ -z "$1" ]
then
    echo
    echo "A configuration file must be specified, or if you want to create a new"
    echo "configuration: enter the path to the new (non-existing) configuration file."
    echo "  ertgui.sh ert_configuration_file"
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
source /prog/sdpsoft/environment.sh > /dev/null

#
# Required by ERT
#
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/res/x86_64_RH_4/lib:/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/lib


ORIGINAL_DIRECTORY=$PWD

export ert_gui_dir=$script_dir/../code
echo "Loading python GUI code from:" $ert_gui_dir
echo "-----------------------------------------------------------------"

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