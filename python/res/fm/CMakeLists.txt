set(PYTHON_SOURCES
    __init__.py
)



add_subdirectory(ecl)
add_subdirectory(rms)
add_subdirectory(shell)
add_subdirectory(util)
add_python_package("python.res.fm" ${PYTHON_INSTALL_PREFIX}/res/fm "${PYTHON_SOURCES}" True)

