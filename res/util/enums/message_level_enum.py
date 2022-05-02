from cwrap import BaseCEnum


class MessageLevelEnum(BaseCEnum):
    TYPE_NAME = "message_level_enum"

    LOG_CRITICAL = None
    LOG_ERROR = None
    LOG_WARNING = None
    LOG_INFO = None
    LOG_DEBUG = None

    @classmethod
    def to_enum(cls, val):
        if isinstance(val, MessageLevelEnum):
            return val
        if not isinstance(val, int):
            raise TypeError(f"Cannot convert {type(val)} to MessageLevelEnum.")
        if val is None:
            return MessageLevelEnum.LOG_WARNING
        if val >= 50:
            return MessageLevelEnum.LOG_CRITICAL
        if val >= 40:
            return MessageLevelEnum.LOG_ERROR
        if val >= 30:
            return MessageLevelEnum.LOG_WARNING
        if val >= 20:
            return MessageLevelEnum.LOG_INFO
        # default level for any values <= 10
        return MessageLevelEnum.LOG_DEBUG


MessageLevelEnum.addEnum("LOG_CRITICAL", 50)
MessageLevelEnum.addEnum("LOG_ERROR", 40)
MessageLevelEnum.addEnum("LOG_WARNING", 30)
MessageLevelEnum.addEnum("LOG_INFO", 20)
MessageLevelEnum.addEnum("LOG_DEBUG", 10)
