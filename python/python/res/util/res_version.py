from ecl.util.util import Version
from res.util import ResUtilPrototype

class ResVersion(Version):
    _build_time = ResUtilPrototype("char* res_version_get_build_time()", bind = False)
    _git_commit = ResUtilPrototype("char* res_version_get_git_commit()", bind = False)
    _major_version = ResUtilPrototype("int res_version_get_major_version()", bind = False)
    _minor_version = ResUtilPrototype("int res_version_get_minor_version()", bind = False)
    _micro_version = ResUtilPrototype("char* res_version_get_micro_version()", bind = False)
    _is_devel = ResUtilPrototype("bool res_version_is_devel_version()", bind = False)

    def __init__(self):
        major = self._major_version( )
        minor = self._minor_version( )
        micro = self._micro_version( )
        git_commit = self._git_commit( )
        build_time = self._build_time( )
        super( ResVersion, self).__init__( major, minor , micro , git_commit, build_time)



