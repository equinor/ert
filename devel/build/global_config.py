import os.path

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
    def __init__(self , cwd , sub_level_depth):
        self.SDP_ROOT      = ""
        self.SDP_BIN       = "/tmp/bin"
        self.EXTERNAL_HOME = "/d/proj/bg/enkf/software"
                

        self.CCFLAGS  = "-m64 -O2 -std=gnu99 -g -Wall -fPIC"
        self.RPATH    = "%s/lib" % self.EXTERNAL_HOME
        self.LDFLAGS  = "-rdynamic -Wl,-rpath %s" % self.RPATH

        
        tmp = cwd.split("/")
        n   = len(tmp) - sub_level_depth
        self.BUILD_ROOT = ""
        for path in tmp[1:n]:
            self.BUILD_ROOT += "/%s" % path

        self.LIB_HOME = {}
        self.LIB_HOME[LIBUTIL]       = "%s/libutil"      % self.BUILD_ROOT
        self.LIB_HOME[LIBECL]        = "%s/libecl"       % self.BUILD_ROOT
        self.LIB_HOME[LIBRMS]        = "%s/librms"       % self.BUILD_ROOT
        self.LIB_HOME[LIBCONF]       = "%s/libconf"      % self.BUILD_ROOT
        self.LIB_HOME[LIBPLOT]       = "%s/libplot"      % self.BUILD_ROOT
        self.LIB_HOME[LIBENKF]       = "%s/libenkf"      % self.BUILD_ROOT
        self.LIB_HOME[LIBJOB_QUEUE]  = "%s/libjob_queue" % self.BUILD_ROOT
        self.LIB_HOME[LIBSCHED]      = "%s/libsched"     % self.BUILD_ROOT
        self.LIB_HOME[LIBCONFIG]     = "%s/libconfig"    % self.BUILD_ROOT
        self.LIB_HOME[EXTERNAL]      = self.EXTERNAL_HOME


    def env_update_path( self , env , liblist ):
        CPPPATH = []
        LIBPATH = []
        for lib in liblist:
            CPPPATH.append("%s/include" % self.LIB_HOME[ lib ])
            LIBPATH.append("%s/lib" % self.LIB_HOME[ lib ])

        env.Replace( CPPPATH = CPPPATH ,
                     LIBPATH = LIBPATH ,
                     RPATH   = self.RPATH )
            
        
        
def get_conf(cwd , sub_level_depth):
    return conf( cwd , sub_level_depth )
        


