class MockedBlockObservation(object):
    def __init__(self, data):
        self.values = data["values"]
        self.stds = data["stds"]

    def __iter__(self):
        for i, _ in enumerate(self.values):
            yield i

    def getValue(self, index):
        return self.values[index]

    def getStd(self, index):
        return self.stds[index]
