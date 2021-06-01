from ecl.util.util import Version
from res import ResPrototype


class ResVersion(Version):
    _build_time = ResPrototype("char* res_version_get_build_time()", bind=False)
    _git_commit = ResPrototype("char* res_version_get_git_commit()", bind=False)
    _major_version = ResPrototype("int res_version_get_major_version()", bind=False)
    _minor_version = ResPrototype("int res_version_get_minor_version()", bind=False)
    _micro_version = ResPrototype("char* res_version_get_micro_version()", bind=False)
    _is_devel = ResPrototype("bool res_version_is_devel_version()", bind=False)

    def __init__(self):
        major = self._major_version()
        minor = self._minor_version()
        micro = self._micro_version()
        git_commit = self._git_commit()
        build_time = self._build_time()
        super(ResVersion, self).__init__(major, minor, micro, git_commit, build_time)
