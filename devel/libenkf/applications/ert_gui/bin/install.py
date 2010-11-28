#!/usr/bin/python
import sys
import os.path
import os
import re
from   stat import *
import shutil
sys.path += ["../../../../python/ctypes/SDP"]
import SDP

# # The target for installation should be given as the first arguement on
# # the commandline, if no target is given the default_target will be
# # used.
# default_target = "/d/proj/bg/enkf/ERT_GUI"
# 
# 
# # All installed files and directories will get group ownership given by
# # the parameter target_guid.
# target_guid  = os.stat("/d/proj/bg/enkf")[ST_GID]
# 
# 
# verbose = True
# 
# file_mode_x    = S_IRUSR + S_IWUSR + S_IXUSR + S_IRGRP + S_IWGRP + S_IXGRP + S_IROTH + S_IXOTH   # u:rwx    g:rwx    o:rx
# file_mode      = S_IRUSR + S_IWUSR +           S_IRGRP + S_IWGRP +           S_IROTH             # u:rw     g:rw     o:r
# dir_mode       = S_IRUSR + S_IWUSR + S_IXUSR + S_IRGRP + S_IWGRP + S_IXGRP + S_IROTH + S_IXOTH   # u:rwx    g:rwx    o:rx           
# 
# ert_libs = ["util" , "config" , "enkf" , "rms" , "ecl" , "sched" , "job_queue"]
# 
# exclude_re = re.compile("(\.svn|~$|\.pyc$|#)")
# 
# 
# 
# # Relpath function nicked from python2.6
# def __relpath(path, start):
#     """Return a relative version of a path"""
# 
#     if not path:
#         raise ValueError("no path specified")
# 
#     start_list = os.path.abspath(start).split(os.path.sep)
#     path_list = os.path.abspath(path).split(os.path.sep)
# 
#     # Work out how much of the filepath is shared by start and path.
#     i = len(os.path.commonprefix([start_list, path_list]))
# 
#     rel_list = [".."] * (len(start_list)-i) + path_list[i:]
#     if not rel_list:
#         return "."
#     return os.path.join(*rel_list)
# 
# 
# 
# def install_file( src_file , target_file , verbose):
#     shutil.copyfile( src_file , target_file )
#     if verbose:
#         print "Copying file: %s -> %s" % ( src_file , target_file )
#     os.chown( target_file , -1 , target_guid )
#     mode = os.stat( src_file )[ST_MODE]
#     if S_IXUSR & mode:
#         os.chmod( target_file , file_mode_x)
#     else:
#         os.chmod( target_file , file_mode)
# 
# 
# 
# def install_dir( full_target , verbose):
#     if not os.path.exists( full_target ):
#         if verbose:
#             print "Creating directory:%s" % full_target
#         os.chown( target_file , -1 , target_guid )
#         os.mkdir( full_target )
#         os.chmod( full_target , dir_mode )
# 
# 
# def install_walk( arg , dirname , entries ):
#     src_root    = arg["src_root"]
#     target_root = arg["target_root"]
#     verbose     = arg["verbose"]
# 
#     relpath = __relpath( dirname , src_root )
#     if relpath == "bin":
#         # Special case for the bin directory:
#         install_file( "%s/gert" % dirname        , "%s/%s/gert"        % ( target_root , relpath) , verbose)
#         install_file( "%s/gdbcommands" % dirname , "%s/%s/gdbcommands" % ( target_root , relpath) , verbose)
#         install_file( "%s/clean.py" % dirname    , "%s/%s/clean.py"    % ( target_root , relpath) , verbose)
#     else:
#         for entry in entries:
#             full_entry = "%s/%s" % (dirname , entry)
#             if not exclude_re.search( full_entry ):
#                 full_target = "%s/%s/%s" % (target_root , relpath , entry)
#                 if os.path.isfile( full_entry ):
#                     install_file( full_entry , full_target , verbose)
#                 elif os.path.isdir( full_entry ):
#                     install_dir( full_target , verbose)
# 
# 
# 
# def install_python( src , target , verbose):
#     # Recursively walks through the directory rooted at @src, and
#     # copies "everything" to a corresponding directory structure at
#     # @target. The following files are excluded from the copy process:
#     # 
#     #     ~ and # backup files
#     #     .svn directories
#     #     .pyc compiled python files.
#     # 
#     # The copied files and directories are created with group enkf_h,
#     # read (and directory x) access for everyone, and write access for
#     # group.
# 
#     
#     if not os.path.exists( target ):
#         if verbose:
#             print "Creating installation directory: %s" % target
#         os.makedirs( target )
#         
#     arg = { "src_root"    : src ,
#             "target_root" : target,
#             "verbose"     : verbose }
#     
#     os.path.walk( src , install_walk , arg )


#################################################################

(SDP_ROOT , RH) = SDP.get_SDP_ROOT()
python_root = "%s/lib/python" % SDP_ROOT
lib_root    = "%s/lib/python/lib"  % SDP_ROOT

SDP.install_file("../../../../libutil/slib/libutil.so"           , "%s/libutil.so" % lib_root      , strict_exists = False)
SDP.install_file("../../../../libecl/slib/libecl.so"             , "%s/libecl.so" % lib_root       , strict_exists = False)
SDP.install_file("../../../../librms/slib/librms.so"             , "%s/librms.so" % lib_root       , strict_exists = False)
SDP.install_file("../../../../libenkf/slib/libenkf.so"           , "%s/libenkf.so" % lib_root      , strict_exists = False)
SDP.install_file("../../../../libconfig/slib/libconfig.so"       , "%s/libconfig.so" % lib_root    , strict_exists = False)
SDP.install_file("../../../../libjob_queue/slib/libjob_queue.so" , "%s/libjob_queue.so" % lib_root , strict_exists = False)
SDP.install_file("../../../../libplot/slib/libplot.so"           , "%s/libplot.so" % lib_root      , strict_exists = False)
SDP.install_file("../../../../libsched/slib/libsched.so"         , "%s/libsched.so" % lib_root     , strict_exists = False)

SDP.make_dir( "%s/gert" % python_root )
SDP.install_path( "code" , "%s/gert" % python_root  ,  root = "../" , extensions = ["py"])
SDP.install_path( "help" , "%s/gert" % python_root  ,  root = "../" , extensions = ["py"])
SDP.install_path( "img"  , "%s/gert"  % python_root ,  root = "../" , extensions = ["py"])
SDP.install_path( "doc"  , "%s/gert"  % python_root ,  root = "../" , extensions = ["py"])

SDP.make_dir( "%s/gert/bin" % python_root )
SDP.install_file( "gert"        , "%s/bin/gert"        % SDP_ROOT)
SDP.install_file( "gdbcommands" , "%s/bin/gdbcommands" % SDP_ROOT)

