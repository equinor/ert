#!/bin/sh


destination=$1
script_dir=$(dirname $0)

if [ -z "$destination" ]
then
    destination=/d/proj/bg/enkf/ERT_GUI
    echo "Will install to default installation directory:" $destination
fi

ert_gui_dir=$script_dir/..


rm -rf $destination/code
rm -rf $destination/bin
rm -rf $destination/img
rm -rf $destination/help

cp -rf $ert_gui_dir/code $destination/
cp -rf $ert_gui_dir/bin $destination/
cp -rf $ert_gui_dir/img $destination/
cp -rf $ert_gui_dir/help $destination/

find $destination -name ".svn" -type d -exec rm -rf {} +
find $destination -name "*.pyc" -exec rm -rf {} +

chmod a+rwx -R $destination/code
chmod a+rwx -R $destination/bin
chmod a+rwx -R $destination/img
chmod a+rwx -R $destination/help


if [ -n "$ERT_HOME" ]
then
    # Can NOT unconditionally remove the ERT_HOME/lib directory because it
    # will in addition contain links to other external libraries (like LSF).        
    cp -fv $ERT_HOME/libenkf/slib/libenkf.so $destination/lib
    cp -fv $ERT_HOME/libconfig/slib/libconfig.so $destination/lib
    cp -fv $ERT_HOME/libecl/slib/libecl.so $destination/lib
    cp -fv $ERT_HOME/libsched/slib/libsched.so $destination/lib
    cp -fv $ERT_HOME/libutil/slib/libutil.so $destination/lib
    cp -fv $ERT_HOME/librms/slib/librms.so $destination/lib
    cp -fv $ERT_HOME/libjob_queue/slib/libjob_queue.so $destination/lib

    chmod a+rwx -R $destination/lib
fi



echo
echo "The GUI for ERT has been installed in: $destination."
echo