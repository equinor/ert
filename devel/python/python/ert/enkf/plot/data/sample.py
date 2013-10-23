from ert.enkf.plot.data import DictProperty


class Sample(dict):
    index = DictProperty("index")
    x = DictProperty("x")
    value = DictProperty("value")
    std = DictProperty("std")

    group = DictProperty("group")
    name = DictProperty("name")
    single_point = DictProperty("single_point")

    def __init__(self):
        super(Sample, self).__init__()

        self.index = None
        self.x = 0.0
        self.value = 0.0
        self.std = 0.0

        self.group = None
        self.name = None

        self.single_point = False