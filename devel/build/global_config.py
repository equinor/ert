import SCons
import os.path
import os
import commands

def chgrp(path , group):
    os.system("chgrp %s %s" % (group , path))

#################################################################
from SCons.Script.SConscript import SConsEnvironment
SConsEnvironment.Chmod = SCons.Action.ActionFactory( os.chmod , lambda dest,mode : '')
SConsEnvironment.Chgrp = SCons.Action.ActionFactory(    chgrp , lambda dest,group: '')

def InstallPerm(env , dest , files , mode):
    if not os.path.exists( dest ):
        print "Creating directory: %s" % dest
        os.makedirs( dest )

    try:
        os.chmod( dest , 0775 )
        chgrp( dest , "res" ) 
    except:
        pass
    
    obj = env.Install( dest , files )
    for f in obj:
        env.AddPostAction(f , env.Chmod(str(f) , mode))
        env.AddPostAction(f , env.Chgrp(str(f) , "res"))
    return dest


def InstallProgram(env , dest , files):
    return InstallPerm( env , dest , files , 0775)


def InstallHeader(env , dest , files):
    return InstallPerm( env , dest , files , 0664)


def InstallLibrary(env , dest , files):
    return InstallPerm( env , dest , files , 0664)


SConsEnvironment.InstallPerm    = InstallPerm
SConsEnvironment.InstallLibrary = InstallLibrary
SConsEnvironment.InstallHeader  = InstallHeader
SConsEnvironment.InstallProgram = InstallProgram

#################################################################


def add_program(env , conf , bin_path , target , src):
    P = env.Program( target , src )
    env.InstallProgram(bin_path , P)

    env.InstallProgram(conf.SDP_BIN , P)

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
    return sdp_root



class conf:
    def __init__(self , cwd , sub_level_depth):

        self.SVN_VERSION      = commands.getoutput("svnversion ./")
        self.TIME_STAMP       = commands.getoutput("date").replace(" " , "_")
        
        self.SITE_CONFIG_FILE     = "/project/res/etc/ERT/Config/site-config"
        self.SDP_ROOT             = get_SDP_ROOT()
        self.SDP_BIN              = "%s/bin"             % self.SDP_ROOT
        self.SDP_INCLUDE          = "%s/include"         % self.SDP_ROOT
        self.SDP_LIB              = "%s/lib"             % self.SDP_ROOT
        self.SDP_ERT_RELEASE      = "%s/bin/ert_release" % self.SDP_ROOT
        
        self.CCFLAGS  = "-m64 -O2 -std=gnu99 -g -Wall -fPIC"
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


        if ext_liblist:
            LIBS += ext_liblist
        env.Replace( CPPPATH  = CPPPATH ,
                     CCFLAGS  = self.CCFLAGS )

        if link:
            env.Replace(LIBPATH  = LIBPATH ,
                        LIBS     = LIBS    , 
                        RPATH    = self.RPATH)

                
        
def get_conf(cwd , sub_level_depth):
    return conf( cwd , sub_level_depth )
        


