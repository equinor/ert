from cwrap import BaseCEnum


class ActiveMode(BaseCEnum):
    TYPE_NAME = "active_mode_enum"
    ALL_ACTIVE = None
    PARTLY_ACTIVE = None


ActiveMode.addEnum("ALL_ACTIVE", 1)
ActiveMode.addEnum("PARTLY_ACTIVE", 3)
