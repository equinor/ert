#!/prog/sdpsoft/python2.4/bin/python
import os
import os.path
import stat
import shutil
import sys
import py_compile

script_mode = 0775
data_mode   = 0664
dir_mode    = 0775
uid         = os.getuid()
res_guid    = os.stat("/project/res")[stat.ST_GID]

lib_root    = "%s/lib/python/lib"        
python_root = "%s/lib/python"

#################################################################
#
# This file is an installation script for the ert python code at
# Statoil. The script will perform several different tasks:
# 
#  1. The directory tree rooted starting with ert/ in the current
#     directory will be copied plainly to the directory given by
#     @python_root.
# 
#  2. All python files will be byte-compiled in place at the
#     installation directory immediately after installation.
# 
#  3. Everything which is created in the /project/res filesystem
#     should be owned by the group 'res' and have mode according to
#     the variables script_mode / data_mode / dir_mode above (there
#     have been numerous bugs with this ...).
# 
#  ----------------------------------------------------------------
#  
#  4. The shared ert libraries which the ert py library is based on
#     will be copied to the directory given by @lib_root.
# 
#  5. The special shared library C/libpycfile/slib/libpycfile.so will
#     be installed as @lib_root/libpycfile.so and a symlink:
#     
#         @python_root/ert/util/pycfile.so -> @lib_root/libpycfile.so
# 
#     will be installed.
#        
# ----------------------------------------------------------------- 
# 
# Observe that the installation script is NOT coupled to the build
# system used to build the C based shared libraries. This implies
# that:
# 
#  a) If you want to install the shared libraries in addition to the
#     python code you must use scons and compile all C code prior to
#     running this installation script:
# 
#        cd ../../
#        scons -j 4
#        cd -
#        cd C/libpycfile/src
#        scons
#        cd -
#        
#  b) The installation steps 4 & 5 above are optional, in the sense
#     that if the shared libraries in question are not found in the
#     local filesystem a warning is issued and the installation
#     continues. This implies that:
# 
#      i) If you have only updated the Python code you can forget
#         about the C code and still run the installation script.
# 
#      ii) If you have at one stage compiled and built the shared
#          libraries you must sure that you do not inadvertly install
#          old versions on a subsequent later install!
#
#################################################################



def get_SDP_ROOT():
    cpu = os.uname()[4]
    RH  = open('/etc/redhat-release').read().split()[6]
    res_target = "%s_RH_%s" % (cpu , RH)
    sdp_root = "/project/res/%s_RH_%s" % (cpu , RH)
    return (sdp_root , float(RH))


def chgrp(path , guid ):
    os.chown( path , -1 , guid )


def install_link( target_file , link_name):
    if os.path.exists( target_file ):
        if os.path.exists( link_name ):
            os.unlink( link_name )
            print "Linking: %s -> %s" % (link_name , target_file )
        os.symlink( target_file , link_name )
        chgrp( link_name , res_guid )
    else:
        print "Warning: file: %s does not exist - skipping: %s -> %s." % ( target_file , link_name , target_file )



def install_file( src_file , target_file , strict_exists = True):
    if not os.path.exists( src_file ):
        if strict_exists:
            raise Exception("Copying %s -> %s" % (src_file , target_file))
        else:
            print "Warning: file: %s does not exist - skipping"

    shutil.copyfile( src_file , target_file )
    print "Copying %s->%s" % (src_file , target_file)
    if os.stat( target_file )[stat.ST_UID] == uid:
        chgrp( target_file , res_guid )

        st = os.stat( src_file )
        if st[stat.ST_MODE] & stat.S_IXUSR: 
            os.chmod( target_file , script_mode )
        else:
            os.chmod( target_file , data_mode )


def make_dir( target_dir ):
    if not os.path.exists( target_dir ):
        os.makedirs( target_dir )
        print "Creating directory: %s" % target_dir

    if os.stat( target_dir )[stat.ST_UID] == uid:
        chgrp( target_dir , res_guid )
        os.chmod( target_dir , dir_mode )



def install_path( src_path , target_root , extensions = None):
    target_dir = "%s/%s" % (target_root , src_path)
    
    make_dir( target_dir )
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
            if extensions:
                (base, ext) = os.path.splitext( full_path )
                if ext[1:] in extensions:
                    file_entries.append( full_path )
            else:
                file_entries.append( full_path )

    for file in file_entries:
        target_file = "%s/%s" % (target_root , file )
        (base, ext) = os.path.splitext( file )
        install_file( file , target_file )
        if ext == ".py":
            print "Byte compiling: %s" % target_file
            py_compile.compile( target_file )
            pyc_file = target_file + "c"
            chgrp( pyc_file , res_guid )
            os.chmod( pyc_file , data_mode )
    
    #Recursive"
    for dir in dir_entries:
        install_path( dir , target_root , extensions = extensions)
    

os.umask( 2 )
(SDP_ROOT , RH) = get_SDP_ROOT()
python_root = python_root % SDP_ROOT
lib_root    = lib_root % SDP_ROOT

make_dir( python_root )
make_dir( lib_root )

cwd = os.getcwd()
install_path( "ert" , python_root , extensions = ["py"])

install_file( "C/libpycfile/slib/libpycfile.so"            , "%s/libpycfile.so"   % lib_root , strict_exists = False)
install_file( "../../libutil/slib/libutil.so"              , "%s/libutil.so"      % lib_root , strict_exists = False)
install_file( "../../libconfig/slib/libconfig.so"          , "%s/libpconfig.so"   % lib_root , strict_exists = False)
install_file( "../../libecl/slib/libecl.so"                , "%s/libpecl.so"      % lib_root , strict_exists = False)
install_file( "../../libjob_queue/slib/libjob_queue.so"    , "%s/libjob_queue.so" % lib_root , strict_exists = False)

install_link( "%s/lib/python/lib/libpycfile.so" % SDP_ROOT , "%s/ert/util/pycfile.so" % python_root)
install_link( "%s/C/libpycfile/slib/libpycfile.so"  % cwd  , "%s/ert/util/pycfile.so" % cwd)
