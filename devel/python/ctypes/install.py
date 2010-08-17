#!/usr/bin/python
import os
import os.path
import stat
import shutil

script_mode = 0775
data_mode   = 0664
dir_mode    = 0775



target = "/d/proj/bg/enkf/python"
res_guid    = os.stat("/project/res")[stat.ST_GID]



def chgrp(path , guid ):
    os.chown( path , -1 , guid )


def install_file( src_file , target_file):
    shutil.copyfile( src_file , target_file )
    print "Updating file: %s" % target_file
    chgrp( target_file , res_guid )
    st = os.stat( src_file )
    if st[stat.ST_MODE] & stat.S_IXUSR: 
        os.chmod( target_file , script_mode )
    else:
        os.chmod( target_file , data_mode )



def install_path( src_path , target_root ):
    target_dir = "%s/%s" % (target_root , src_path)
    if not os.path.exists( target_dir ):
        os.makedirs( target_dir )
        print "Creating directory: %s" % target_dir
    chgrp( target_dir , res_guid )
    os.chmod( target_dir , dir_mode )

    dir_entries  = []
    file_entries = []

    for entry in os.listdir( src_path ):
        if entry == ".svn":
            continue

        if entry[-1] == "~":
            continue

        full_path = "%s/%s" % (src_path , entry)
        
        if os.path.isdir( full_path ):
            dir_entries.append( full_path )
        else:
            file_entries.append( full_path )

    for file in file_entries:
        target_file = "%s/%s" % (target_root , file )
        install_file( file , target_file )
            
    #Recursive"
    for dir in dir_entries:
        install_path( dir , target_root )
    

os.umask( 2 )
os.chdir( "/private/joaho/EnKF/devel/EnKF/python/ctypes" )
install_path( "ert" , target )
install_file( "/private/joaho/EnKF/devel/EnKF/libutil/slib/libutil.so"           , "/d/proj/bg/enkf/python/lib/libutil.so")
install_file( "/private/joaho/EnKF/devel/EnKF/libecl/slib/libecl.so"             , "/d/proj/bg/enkf/python/lib/libecl.so")
install_file( "/private/joaho/EnKF/devel/EnKF/libjob_queue/slib/libjob_queue.so" , "/d/proj/bg/enkf/python/lib/libjob_queue.so")
