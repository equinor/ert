from cwrap import BaseCEnum


class MessageLevelEnum(BaseCEnum):
    TYPE_NAME = "message_level_enum"

    LOG_CRITICAL = None
    LOG_ERROR    = None
    LOG_WARNING  = None
    LOG_INFO     = None
    LOG_DEBUG    = None

    @classmethod
    def __legacy_values(cls, val):
        """These are deprecated legacy values that were deprecated in res 2.2.

        They will be removed later."""

        LEGACY_LEVELS = {
            4: MessageLevelEnum.LOG_DEBUG,
            3: MessageLevelEnum.LOG_INFO,
            2: MessageLevelEnum.LOG_WARNING,
            1: MessageLevelEnum.LOG_ERROR,
            0: MessageLevelEnum.LOG_CRITICAL
        }
        return LEGACY_LEVELS.get(val)

    @classmethod
    def to_enum(cls, val):
        legacy = cls.__legacy_values(val)
        if legacy:
            return legacy

        if isinstance(val, MessageLevelEnum):
            return val
        if not isinstance(val, int):
            raise TypeError('Cannot convert %s to MessageLevelEnum.' % type(val))
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
MessageLevelEnum.addEnum("LOG_ERROR",    40)
MessageLevelEnum.addEnum("LOG_WARNING",  30)
MessageLevelEnum.addEnum("LOG_INFO",     20)
MessageLevelEnum.addEnum("LOG_DEBUG",    10)
