#!/prog/sdpsoft/python2.4/bin/python
import os
import os.path
import stat
import shutil
import sys
import py_compile
sys.path += ["SDP"]
import SDP


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

    

(SDP_ROOT , RH) = get_SDP_ROOT()
python_root = "%s/lib/python_root" % SDP_ROOT
lib_root    = "%s/lib/python/lib"  % SDP_ROOT
cwd = os.getcwd()

SDP.make_dir( python_root )
SDP.make_dir( lib_root )

SDP.install_path( "ert" , python_root , extensions = ["py"])

SDP.install_file( "C/libpycfile/slib/libpycfile.so"            , "%s/libpycfile.so"   % lib_root , strict_exists = False)
SDP.install_file( "../../libutil/slib/libutil.so"              , "%s/libutil.so"      % lib_root , strict_exists = False)
SDP.install_file( "../../libconfig/slib/libconfig.so"          , "%s/libconfig.so"   % lib_root , strict_exists = False)
SDP.install_file( "../../libecl/slib/libecl.so"                , "%s/libecl.so"      % lib_root , strict_exists = False)
SDP.install_file( "../../libjob_queue/slib/libjob_queue.so"    , "%s/libjob_queue.so" % lib_root , strict_exists = False)

SDP.install_link( "%s/lib/python/lib/libpycfile.so" % SDP_ROOT , "%s/ert/util/pycfile.so" % python_root)
SDP.install_link( "%s/C/libpycfile/slib/libpycfile.so"  % cwd  , "%s/ert/util/pycfile.so" % cwd)
