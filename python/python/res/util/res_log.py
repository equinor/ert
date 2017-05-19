from res.util import ResUtilPrototype


class ResLog(object):
    _init = ResUtilPrototype("void res_log_init_log(int, char*, bool)", bind=False)
    _write_log = ResUtilPrototype("void res_log_add_message_py(int, char*)", bind=False)
    _get_filename = ResUtilPrototype("char* res_log_get_filename()", bind=False)

    @classmethod
    def init(cls, log_level, log_filename, verbose):
        cls._init(log_level, log_filename, verbose)

    @classmethod
    def log(cls, log_level, message):
        """ Higher log_level indicates higher importance"""
        cls._write_log(log_level, message)

    @classmethod
    def getFilename(cls):
        """ @rtype: string """
        return cls._get_filename()
