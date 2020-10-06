from res import ResPrototype
from res.util.enums import MessageLevelEnum


class ResLog(object):
    _init = ResPrototype(
        "bool res_log_init_log(message_level_enum, char*, bool)", bind=False
    )
    _write_log = ResPrototype(
        "void res_log_add_message(message_level_enum, char*)", bind=False
    )
    _get_filename = ResPrototype("char* res_log_get_filename()", bind=False)

    @classmethod
    def init(cls, log_level, log_filename, verbose):
        if not cls._init(MessageLevelEnum.to_enum(log_level), log_filename, verbose):
            raise IOError("Failed to open log handle for file:%s" % log_filename)

    @classmethod
    def log(cls, log_level, message):
        """ Higher log_level indicates higher importance"""
        cls._write_log(MessageLevelEnum.to_enum(log_level), message)

    @classmethod
    def getFilename(cls):
        """ @rtype: string """
        return cls._get_filename()
