import SCons
import os.path
import os
import commands
import stat

os.umask(2)
res_guid = os.stat("/project/res")[stat.ST_GID]

def mkdir_gid( path ):
    mkdir( path )
    os.chown( path , -1 , res_guid )

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
        env.AddPostAction(f , env.Chgrp(str(f) , res_guid))
    return dest


def InstallProgram(env , dest , files):
    return InstallPerm( env , dest , files , 0775)


def InstallHeader(env , dest , files):
    return InstallPerm( env , dest , files , 0553)


def InstallLibrary(env , dest , files ):
    return InstallPerm( env , dest , files , 0553)


SConsEnvironment.InstallPerm    = InstallPerm
SConsEnvironment.InstallLibrary = InstallLibrary
SConsEnvironment.InstallHeader  = InstallHeader
SConsEnvironment.InstallProgram = InstallProgram

#################################################################


def add_program(env , conf , bin_path , target , src , **kwlist):
    P = env.Program( target , src , **kwlist)
    env.InstallProgram(bin_path , P)
    env.InstallProgram(conf.SDP_BIN_TARGET , P)
    conf.SDP_INSTALL[ conf.SDP_BIN_TARGET ] = True
    conf.local_install[ bin_path ]   = True


def add_static_library( env, conf , lib_path , target , src , **kwlist):
    LIB = env.StaticLibrary( target , src , **kwlist)
    env.InstallLibrary( lib_path , LIB )
    env.InstallLibrary( conf.SDP_LIB_TARGET , LIB )
    conf.SDP_INSTALL[ conf.SDP_LIB_TARGET ] = True
    conf.local_install[ lib_path ]   = True


def add_shared_library( env, conf , lib_path , target , src , **kwlist):
    LIB = env.SharedLibrary( target , src , **kwlist)
    env.InstallLibrary( lib_path , LIB )
    #env.InstallLibrary( conf.SDP_LIB_TARGET , LIB )
    #conf.SDP_INSTALL[ conf.SDP_LIB_TARGET ] = True
    conf.local_install[ lib_path ]   = True


def add_header( env, conf , include_path , header_list ):
    env.InstallHeader( include_path , header_list )
    env.InstallHeader( conf.SDP_INCLUDE_TARGET , header_list )
    conf.SDP_INSTALL[ conf.SDP_INCLUDE_TARGET ] = True
    conf.local_install[ include_path ]   = True


def get_targets( env , conf):
    def_list = []
    SDP_list = []
    for target in conf.local_install.keys():
        def_list.append( target )

    for target in conf.SDP_INSTALL.keys():
        SDP_list.append( target )

    return (env.Alias('local' , def_list) , env.Alias('SDP_INSTALL' , SDP_list))


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



def get_SDP_ROOT():
    cpu = os.uname()[4]
    RH  = open('/etc/redhat-release').read().split()[6]
    res_target = "%s_RH_%s" % (cpu , RH)
    sdp_root = "/project/res/%s_RH_%s" % (cpu , RH)
    return (sdp_root , float(RH))



class conf:
    def __init__(self , cwd , package , sub_level_depth):

        self.SVN_VERSION      = commands.getoutput("svnversion ./")
        self.TIME_STAMP       = commands.getoutput("date")#.replace(" " , "_")
        
        self.SITE_CONFIG_FILE     = "/project/res/etc/ERT/Config/site-config"
        (self.SDP_ROOT , self.RH) = get_SDP_ROOT()
        self.SDP_BIN              = "%s/bin"             %  self.SDP_ROOT
        self.SDP_BIN_TARGET       = "%s/bin"             %  self.SDP_ROOT
        self.SDP_INCLUDE          = "%s/include"         %  self.SDP_ROOT
        self.SDP_INCLUDE_TARGET   = "%s/include/lib%s"   % (self.SDP_ROOT , package)
        self.SDP_LIB              = "%s/lib"             %  self.SDP_ROOT
        self.SDP_LIB_TARGET       = "%s/lib"             %  self.SDP_ROOT
        self.SDP_ERT_RELEASE      = "%s/bin/ert_release" %  self.SDP_ROOT

        self.CCFLAGS  = "-m64 -O2 -std=gnu99 -g -Wall"
        self.ARFLAGS  = "csr"

        
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
        self.RPATH              = self.SDP_LIB

        



    def update_env( self , env , liblist , ext_liblist = None , link = False):
        self.SDP_INSTALL   = {}
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
        CPPPATH.append( self.SDP_INCLUDE )
        LIBPATH.append( self.SDP_LIB )
        env.Replace( CC = "gcc" )

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
        


