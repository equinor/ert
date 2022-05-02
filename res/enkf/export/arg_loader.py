import numpy
from pandas import DataFrame


class ArgLoader:
    @staticmethod
    def load(filename, column_names=None):
        rows = 0
        columns = 0
        with open(filename, "r") as fileH:
            for line in fileH.readlines():
                rows += 1
                columns = max(columns, len(line.split()))

        if column_names is not None:
            if len(column_names) <= columns:
                columns = len(column_names)
            else:
                raise ValueError("To many coloumns in input")

        data = numpy.empty(shape=(rows, columns), dtype=numpy.float64)
        data.fill(numpy.nan)

        row = 0
        with open(filename) as fileH:
            for line in fileH.readlines():
                tmp = line.split()
                print(tmp)
                for column in range(columns):
                    data[row][column] = float(tmp[column])
                row += 1

        if column_names is None:
            column_names = []
            for column in range(columns):
                column_names.append(f"Column{column:d}")

        data_frame = DataFrame(data=data, columns=column_names)
        return data_frame
