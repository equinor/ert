import numpy
from pandas import MultiIndex, DataFrame
from ert.test import ExtendedTestCase


class NumpyAndPandasTest(ExtendedTestCase):

    def test_numpy(self):
        data = numpy.empty(shape=(10, 10), dtype=numpy.float64)
        data.fill(numpy.nan)

        self.assertTrue(numpy.isnan(data[0][0]))
        self.assertTrue(numpy.isnan(data[9][9]))

        with self.assertRaises(IndexError):
            v = data[10][9]

        data[5][5] = 1.0

        self.assertEqual(data[5][5], 1.0)

        data[0] = 5.0

        test_data = numpy.empty(shape=10)
        test_data.fill(5.0)

        self.assertTrue(numpy.array_equal(data[0], test_data))

        data = numpy.transpose(data)

        self.assertTrue(numpy.array_equal(data[:,0], test_data))

        row = data[0]
        row[5] = 11
        self.assertEqual(data[0][5], 11)


    def test_pandas_join(self):

        multi_index = MultiIndex.from_product([[1, 2], ["A", "B", "C"]], names=["REALIZATION", "LABEL"])

        data = DataFrame(data=[[1, 2, 3], [2, 4, 6], [4, 8, 12]] * 2, index=multi_index, columns=["C1", "C2", "C3"])

        new_column = DataFrame(data=[4.0, 4.4, 4.8], index=[1, 2, 3], columns=["C4"])
        new_column.index.name = "REALIZATION"

        print(data)
        print(new_column)

        result = data.join(new_column, how='inner')
        print(result)

