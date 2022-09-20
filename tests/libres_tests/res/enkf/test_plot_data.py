from ecl.util.util import DoubleVector

from ert._c_wrappers.enkf.plot_data import PlotBlockData, PlotBlockVector


def test_plot_block_vector():
    vector = DoubleVector()
    vector.append(1.5)
    vector.append(2.5)
    vector.append(3.5)
    plot_block_vector = PlotBlockVector(1, vector)

    assert plot_block_vector.getRealizationNumber() == 1
    assert plot_block_vector[0] == 1.5
    assert plot_block_vector[2] == 3.5

    assert len(plot_block_vector) == len(vector)


def test_plot_block_data():
    depth = DoubleVector()
    depth.append(2.5)
    depth.append(3.5)

    data = PlotBlockData(depth)

    assert data.getDepth() == depth

    vector = PlotBlockVector(1, DoubleVector())
    data.addPlotBlockVector(vector)
    data.addPlotBlockVector(PlotBlockVector(2, DoubleVector()))

    assert len(data) == 2

    assert vector == data[1]
