set(PYTHON_SOURCES
    __init__.py
    TinyDashboard.py
    TinyDashHistogram.py
    TinyDashPlot.py
    TinyDashDataView.py
    TinyErtModel.py
    TinyMockErtModel.py
    TinyDashPlotError.py
)

add_python_package("python.tiny_ert_dash" ${PYTHON_INSTALL_PREFIX}/tiny_ert_dash "${PYTHON_SOURCES}" True)