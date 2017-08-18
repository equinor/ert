from ecl.util import Version
from res.util import ResUtilPrototype

class ResVersion(Version):
    _build_time = ResUtilPrototype("char* res_version_get_build_time()")
    _git_commit = ResUtilPrototype("char* res_version_get_git_commit()")
    _major_version = ResUtilPrototype("int res_version_get_major_version()")
    _minor_version = ResUtilPrototype("int res_version_get_minor_version()")
    _micro_version = ResUtilPrototype("char* res_version_get_micro_version()")
    _is_devel = ResUtilPrototype("bool res_version_is_devel_version()")

    def __init__(self):
        major = self._major_version( )
        minor = self._minor_version( )
        micro = self._micro_version( )
        git_commit = self._git_commit( )
        build_time = self._build_time( )
        super( ResVersion, self).__init__( major, minor , micro , git_commit, build_time)



