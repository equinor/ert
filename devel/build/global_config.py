#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'global_config.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details. 


import SCons
import os.path
import os
import commands
import stat

# These are the modes we want - assuming umask( 0 );
# and then comes the fxxxing umask into play.
exe_mode  = 0775
file_mode = 0664

umask = os.umask( 0 )
os.umask( umask )

# Guid ID assigned to all files and directories
# created/modified during the installation process.
guid = os.getgid()


def mkdir_gid( path ):
    mkdir( path )
    os.chown( path , -1 , guid )

mkdir    = os.mkdir
os.mkdir = mkdir_gid

    

def chgrp(path , guid ):
    os.chown( path , -1 , guid )

    
#################################################################
from SCons.Script.SConscript import SConsEnvironment
SConsEnvironment.Chmod = SCons.Action.ActionFactory( os.chmod , lambda dest,mode : '')
SConsEnvironment.Chgrp = SCons.Action.ActionFactory(    chgrp , lambda dest,group: '')

def InstallPerm(env , dest , files , mode):
    obj = env.Install( dest , files )
    for f in obj:
        env.AddPostAction(f , env.Chmod(str(f) , mode))
        env.AddPostAction(f , env.Chgrp(str(f) , guid))
    return dest



def InstallProgram(env , dest , files):
    return InstallPerm( env , dest , files , exe_mode)


def InstallHeader(env , dest , files):
    return InstallPerm( env , dest , files , file_mode)


def InstallLibrary(env , dest , files ):
    return InstallPerm( env , dest , files , file_mode)


SConsEnvironment.InstallPerm    = InstallPerm
SConsEnvironment.InstallLibrary = InstallLibrary
SConsEnvironment.InstallHeader  = InstallHeader
SConsEnvironment.InstallProgram = InstallProgram

#################################################################


def add_program(env , conf , bin_path , target , src , **kwlist):
    P = env.Program( target , src , **kwlist)
    env.InstallProgram(bin_path , P)
    conf.local_install[ bin_path ]   = True


def add_static_library( env, conf , lib_path , target , src , **kwlist):
    LIB = env.StaticLibrary( target , src , **kwlist)
    env.InstallLibrary( lib_path , LIB )
    conf.local_install[ lib_path ]   = True


def add_shared_library( env, conf , lib_path , target , src , **kwlist):
    LIB = env.SharedLibrary( target , src , **kwlist)
    env.InstallLibrary( lib_path , LIB )
    conf.local_install[ lib_path ]   = True


def add_header( env, conf , include_path , header_list ):
    env.InstallHeader( include_path , header_list )
    conf.local_install[ include_path ]   = True


def get_targets( env , conf):
    def_list = []
    for target in conf.local_install.keys():
        def_list.append( target )
    
    return (env.Alias('local' , def_list))


#################################################################


# Used as hash keys
LIBUTIL      = 0
LIBECL       = 1
LIBRMS       = 2
LIBENKF      = 3
LIBPLOT      = 4  
LIBJOB_QUEUE = 5
LIBSCHED     = 6
LIBCONFIG    = 7
LIBCONF      = 8



class conf:
    def __init__(self , cwd , package , sub_level_depth):

        self.SVN_VERSION      = commands.getoutput("svnversion ./")
        self.TIME_STAMP       = commands.getoutput("date")
        
        # These are site-dependant; and should really be set OUTSIDE
        # the central build configuration.
        self.SITE_CONFIG_FILE    = "/project/res/etc/ERT/site-config"
        self.INCLUDE_LSF         = True
        self.LSF_INCLUDE_PATH    = "/prog/LSF/7.0/include"
        self.LSF_LIB_PATH        = "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/lib"
        self.g2c                 = False
        self.PLPLOT_INCLUDE_PATH = "/project/res/x86_64_RH_4/include"
        self.PLPLOT_LIB_PATH     = "/project/res/x86_64_RH_4/lib"
        
        self.CCFLAGS             = "-m64 -O2 -std=gnu99 -g -Wall -pipe"
        self.ARFLAGS             = "csr"
        

        tmp = cwd.split("/")
        n   = len(tmp) - sub_level_depth
        self.BUILD_ROOT = ""
        for path in tmp[1:n]:
            self.BUILD_ROOT += "/%s" % path
            
        self.LIB = {}
        self.LIB[LIBUTIL]       = {"home": "%s/libutil"      % self.BUILD_ROOT , "name": "util"}
        self.LIB[LIBECL]        = {"home": "%s/libecl"       % self.BUILD_ROOT , "name": "ecl"}
        self.LIB[LIBRMS]        = {"home": "%s/librms"       % self.BUILD_ROOT , "name": "rms"}
        self.LIB[LIBCONF]       = {"home": "%s/libconf"      % self.BUILD_ROOT , "name": "conf"}
        self.LIB[LIBPLOT]       = {"home": "%s/libplot"      % self.BUILD_ROOT , "name": "plot"}
        self.LIB[LIBENKF]       = {"home": "%s/libenkf"      % self.BUILD_ROOT , "name": "enkf"}
        self.LIB[LIBJOB_QUEUE]  = {"home": "%s/libjob_queue" % self.BUILD_ROOT , "name": "job_queue"}
        self.LIB[LIBSCHED]      = {"home": "%s/libsched"     % self.BUILD_ROOT , "name": "sched"}
        self.LIB[LIBCONFIG]     = {"home": "%s/libconfig"    % self.BUILD_ROOT , "name": "config"}
        self.RPATH = self.PLPLOT_LIB_PATH



    def update_env( self , env , liblist , ext_liblist = None , ext_headerlist = None , link = False):
        self.local_install = {}
        CPPPATH = ["./"]
        LIBPATH = []
        LIBS    = []
        for lib_key in liblist:
            lib  = self.LIB[lib_key]
            home = lib["home"]
            CPPPATH.append("%s/include" % home )
            LIBPATH.append("%s/lib"     % home )
            
            if lib.has_key("name"):
                name = lib["name"]
                LIBS.append( name )
        env.Replace( CC = "gcc" )

        if ext_headerlist:
            for path in ext_headerlist:
                CPPPATH.append( path )

        if ext_liblist:
            LIBS += ext_liblist
        env.Replace( CPPPATH  = CPPPATH ,
                     CCFLAGS  = self.CCFLAGS )
        
        if link:
            env.Replace(LIBPATH  = LIBPATH ,
                        LIBS     = LIBS    , 
                        RPATH    = self.RPATH)

                
        
def get_conf(cwd , package , sub_level_depth):
    return conf( cwd , package , sub_level_depth )
        


