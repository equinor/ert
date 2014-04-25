from OpenGL.GL import *
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QMainWindow, QDockWidget
from ert.ecl import EclTypeEnum, EclKW, EclGrid
from ert_gui.viewer import Texture3D, Bounds, SliceViewer, SliceSettingsWidget, Texture1D


def loadGrid(path, load_actnum=True):
    """ @rtype: EclGrid """
    with open(path, "r") as f:
        specgrid = EclKW.read_grdecl(f, "SPECGRID", ecl_type=EclTypeEnum.ECL_INT_TYPE, strict=False)
        zcorn = EclKW.read_grdecl(f, "ZCORN")
        coord = EclKW.read_grdecl(f, "COORD")

        actnum = None
        if load_actnum:
            actnum = EclKW.read_grdecl(f, "ACTNUM", ecl_type=EclTypeEnum.ECL_INT_TYPE)

        mapaxes = EclKW.read_grdecl(f, "MAPAXES")
        grid = EclGrid.create(specgrid, zcorn, coord, actnum, mapaxes=mapaxes)

    return grid


def loadKW(keyword, ecl_type, path):
    """ @rtype: EclKW """
    with open(path, "r") as f:
        kw_data = EclKW.read_grdecl(f, keyword, ecl_type=ecl_type)

    return kw_data

def loadGridData(path):
    grid = loadGrid(path)

    nx, ny, nz, nactive = grid.dims
    print(nx, ny, nz)

    bounds = Bounds()

    grid_data = []
    index = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                x, y, z = grid.get_corner_xyz(0, global_index=index)
                active = grid.active(global_index=index)
                if active:
                    active = 1.0
                else:
                    active = 0.0

                bounds.addPoint(x, y, z)

                grid_data.append(x)
                grid_data.append(y)
                grid_data.append(z)
                grid_data.append(active)
                index += 1

    print(bounds)

    return nx, ny, nz, grid_data, bounds


def loadKWData(path, keyword, ecl_type=EclTypeEnum.ECL_FLOAT_TYPE):
    kw_data = loadKW(keyword, ecl_type, path)

    print(kw_data.min, kw_data.max)

    min_value = kw_data.min
    data_range = kw_data.max - kw_data.min

    result = []
    for value in kw_data:
        value = float(value - min_value) / data_range
        result.append(value)

    return result, data_range


def rgb(r, g, b):
    return [r / 255.0, g / 255.0, b / 255.0, 1.0]

def createColorBrewerScale():
    color_list = [rgb(141,211,199),
                  rgb(255,255,179),
                  rgb(190,186,218),
                  rgb(251,128,114),
                  rgb(128,177,211),
                  rgb(253,180,98),
                  rgb(179,222,105),
                  rgb(252,205,229),
                  rgb(217,217,217),
                  rgb(188,128,189),
                  rgb(204,235,197),
                  rgb(255,237,111)]

    colors = [component for color in color_list for component in color]

    return Texture1D(len(colors) / 4, colors)

def createSeismicScale():
    color_list = [rgb(0, 0, 255), rgb(255, 255, 255), rgb(255, 0, 0)]
    colors = [component for color in color_list for component in color]

    return Texture1D(len(colors) / 4, colors, wrap=GL_CLAMP_TO_EDGE)

def createLinearGreyScale():
    color_list = [rgb(128, 128, 128), rgb(255, 255, 255)]
    colors = [component for color in color_list for component in color]

    return Texture1D(len(colors) / 4, colors, wrap=GL_CLAMP_TO_EDGE)

def createRainbowScale():
    color_list = [rgb(200, 0, 255), rgb(0, 0, 255), rgb(0, 255, 0), rgb(255, 255, 0), rgb(255, 127, 0), rgb(255, 0, 0)]
    colors = [component for color in color_list for component in color]

    return Texture1D(len(colors) / 4, colors, wrap=GL_CLAMP_TO_EDGE, internal_format=GL_RGBA8)

def createColorScales():
    return {
        "region_colors": createColorBrewerScale(),
        "seismic": createSeismicScale(),
        "linear_grey": createLinearGreyScale(),
        "rainbow": createRainbowScale()
    }


def createDataStructures():
    nx, ny, nz, grid_data, bounds = loadGridData("/Volumes/Statoil/data/faultregion/grid.grdecl")
    # nx, ny, nz, grid_data, bounds = loadGridData("/Volumes/Statoil/data/TestCase/eclipse/include/example_grid_sim.GRDECL")

    data, data_range = loadKWData("/Volumes/Statoil/data/faultregion/fltblck.grdecl", "FLTBLCK", ecl_type=EclTypeEnum.ECL_INT_TYPE)
    # data, data_range = loadKWData("/Volumes/Statoil/data/TestCase/eclipse/include/example_permx.GRDECL", "PERMX", ecl_type=EclTypeEnum.ECL_FLOAT_TYPE)

    grid_texture = Texture3D(nx, ny, nz, grid_data, GL_RGBA32F, GL_RGBA)
    attribute_texture = Texture3D(nx, ny, nz, data)


    textures = {"grid": grid_texture,
                "grid_data": attribute_texture}

    return textures, bounds, nx, ny, nz, data_range


if __name__ == '__main__':
    app = QApplication(["Slice Viewer"])
    window = QMainWindow()
    window.resize(1024, 768)

    textures, bounds, nx, ny, nz, data_range = createDataStructures()

    color_scales = createColorScales()
    textures["color_scale"] = color_scales[color_scales.keys()[0]]

    viewer = SliceViewer(textures=textures, volume_bounds=bounds, color_scales=color_scales, data_range=data_range)
    viewer.setSliceSize(width=nx, height=ny)

    slice_settings = SliceSettingsWidget(max_slice_count=nz, color_scales=color_scales.keys())
    slice_settings.inactiveCellsHidden.connect(viewer.hideInactiveCells)
    slice_settings.currentSliceChanged.connect(viewer.setCurrentSlice)
    slice_settings.toggleOrthographicProjection.connect(viewer.useOrthographicProjection)
    slice_settings.toggleLighting.connect(viewer.useLighting)
    slice_settings.colorScalesChanged.connect(viewer.changeColorScale)
    slice_settings.regionToggling.connect(viewer.useRegionScaling)
    slice_settings.toggleInterpolation.connect(viewer.useInterpolationOnData)


    dock_widget = QDockWidget("Settings")
    dock_widget.setObjectName("SliceSettingsDock")
    dock_widget.setWidget(slice_settings)
    dock_widget.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)

    window.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)


    window.setCentralWidget(viewer)

    window.show()
    window.activateWindow()
    window.raise_()
    app.exec_()