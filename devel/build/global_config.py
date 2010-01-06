import SCons
import os.path
import commands

#################################################################
from SCons.Script.SConscript import SConsEnvironment
SConsEnvironment.Chmod = SCons.Action.ActionFactory( os.chmod , lambda dest,mode: 'Chmod("%s" , 0%o)' % (dest , mode))


def InstallPerm(env , dest , files , mode):
    if not os.path.exists(dest):
        os.makedirs( dest )
    os.chmod( dest , 0755)
    obj = env.Install( dest , files )
    for f in obj:
        env.AddPostAction(f , env.Chmod(str(f) , mode))
    return dest


def InstallProgram(env , dest , files):
    return InstallPerm( env , dest , files , 0755)


def InstallHeader(env , dest , files):
    return InstallPerm( env , dest , files , 0644)


def InstallLibrary(env , dest , files):
    return InstallPerm( env , dest , files , 0644)


SConsEnvironment.InstallPerm    = InstallPerm
SConsEnvironment.InstallLibrary = InstallLibrary
SConsEnvironment.InstallHeader  = InstallHeader
SConsEnvironment.InstallProgram = InstallProgram

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
EXTERNAL     = 100




class conf:
    def __init__(self , SDP_ROOT , cwd , sub_level_depth):

        self.SVN_VERSION      = commands.getoutput("svnversion ./")
        self.TIME_STAMP       = commands.getoutput("date").replace(" " , "_")
        self.SDP_BIN          = None
        self.SDP_INCLUDE      = None
        self.SDP_LIB          = None
        
        
        self.SITE_CONFIG_FILE     = "/d/proj/bg/enkf/Config/statoil/site-config"
        if SDP_ROOT:
            if not os.path.exists(SDP_ROOT):
                SDP_ROOT = None
            
        self.SDP_ROOT             = SDP_ROOT


        if SDP_ROOT:
            self.SDP_BIN              = "%s/bin"     % self.SDP_ROOT
            self.SDP_INCLUDE          = "%s/include" % self.SDP_ROOT
            self.SDP_LIB              = "%s/lib"     % self.SDP_ROOT
        
        self.CCFLAGS  = "-m64 -O2 -std=gnu99 -g -Wall -fPIC"
        self.ARFLAGS  = "csr"

        
        tmp = cwd.split("/")
        n   = len(tmp) - sub_level_depth
        self.BUILD_ROOT = ""
        for path in tmp[1:n]:
            self.BUILD_ROOT += "/%s" % path
            
        self.EXTERNAL_HOME      = "%s/ext" % self.BUILD_ROOT
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
        self.LIB[EXTERNAL]      = {"home": self.EXTERNAL_HOME}
        self.RPATH    = "%s/lib" % self.EXTERNAL_HOME

        



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

        if ext_liblist:
            LIBS += ext_liblist
        env.Replace( CPPPATH  = CPPPATH ,
                     CCFLAGS  = self.CCFLAGS )

        if link:
            env.Replace(LIBPATH  = LIBPATH ,
                        LIBS     = LIBS    , 
                        RPATH    = self.RPATH)

                
        
def get_conf(SDP_ROOT , cwd , sub_level_depth):
    return conf( SDP_ROOT , cwd , sub_level_depth )
        


