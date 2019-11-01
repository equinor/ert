class SimulationStateStatus(object):
    COLOR_WAITING = (164, 200, 255)
    COLOR_PENDING = (190, 174, 212)
    COLOR_RUNNING = (255, 255, 153)
    COLOR_FAILED = (255, 200, 200)
    COLOR_UNKNOWN = (128, 128, 128)
    COLOR_FINISHED = (127, 201, 127)
    COLOR_NOT_ACTIVE = (255, 255, 255)

    def __init__(self, name, state, color):
        self.__name = name
        self.__state = state
        self.__color = color

        self.__count = 0
        self.__total_count = 1

    @property
    def name(self):
        return self.__name

    @property
    def state(self):
        return self.__state

    @property
    def color(self):
        return self.__color

    @property
    def count(self):
        return self.__count

    @count.setter
    def count(self, value):
        self.__count = value

    @property
    def total_count(self):
        return self.__total_count

    @total_count.setter
    def total_count(self, value):
        self.__total_count = value
