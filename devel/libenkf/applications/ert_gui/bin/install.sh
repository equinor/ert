#!/bin/sh

destination=$1
script_dir=$(dirname $0)

if [ -z "$destination" ]
then
    echo
    echo "A destination directory is required!"
    echo "  install /destination/directory/of/your/choice"
    echo
    exit
fi

ert_gui_dir=$script_dir/..


rm -rf $destination/code
rm -rf $destination/bin
rm -rf $destination/img
rm -rf $destination/help

cp -rf $ert_gui_dir/code $destination/code
cp -rf $ert_gui_dir/bin $destination/bin
cp -rf $ert_gui_dir/img $destination/img
cp -rf $ert_gui_dir/help $destination/help

find $destination -name ".svn" -type d -exec rm -rf {} +
find $destination -name "*.pyc" -exec rm -rf {} +

chmod a+rwx -R $destination/code
chmod a+rwx -R $destination/bin
chmod a+rwx -R $destination/img
chmod a+rwx -R $destination/help


#export ERT_HOME=/private/jpb/EnKF/
if [ -n "$ERT_HOME" ]
then
    rm -rf $destination/lib
    mkdir $destination/lib
    cp -f $ERT_HOME/libenkf/slib/libenkf.so $destination/lib
    cp -f $ERT_HOME/libconfig/slib/libconfig.so $destination/lib
    cp -f $ERT_HOME/libecl/slib/libecl.so $destination/lib
    cp -f $ERT_HOME/libsched/slib/libsched.so $destination/lib
    cp -f $ERT_HOME/libutil/slib/libutil.so $destination/lib
    cp -f $ERT_HOME/librms/slib/librms.so $destination/lib
    cp -f $ERT_HOME/libjob_queue/slib/libjob_queue.so $destination/lib

    chmod a+rwx -R $destination/lib
fi



echo
echo "The GUI for ERT has been installed in: $destination."
#echo "The ERT_HOME environment variable in ertgui.sh must be set before running."
echo