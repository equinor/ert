from ecl.util.util import DoubleVector

from ert._c_wrappers.enkf.plot_data import PlotBlockData, PlotBlockVector

from ....libres_utils import ResTest


class PlotDataTest(ResTest):
    def test_plot_block_vector(self):
        vector = DoubleVector()
        vector.append(1.5)
        vector.append(2.5)
        vector.append(3.5)
        plot_block_vector = PlotBlockVector(1, vector)

        self.assertEqual(plot_block_vector.getRealizationNumber(), 1)
        self.assertEqual(plot_block_vector[0], 1.5)
        self.assertEqual(plot_block_vector[2], 3.5)

        self.assertEqual(len(plot_block_vector), len(vector))

    def test_plot_block_data(self):
        depth = DoubleVector()
        depth.append(2.5)
        depth.append(3.5)

        data = PlotBlockData(depth)

        self.assertEqual(data.getDepth(), depth)

        vector = PlotBlockVector(1, DoubleVector())
        data.addPlotBlockVector(vector)
        data.addPlotBlockVector(PlotBlockVector(2, DoubleVector()))

        self.assertEqual(len(data), 2)

        self.assertEqual(vector, data[1])

    def compareLists(self, source, target):
        self.assertEqual(len(source), len(target))
        for index, value in enumerate(source):
            self.assertEqual(value, target[index])
