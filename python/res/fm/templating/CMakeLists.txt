set(PYTHON_SOURCES
        __init__.py
        template_render.py
        )

add_python_package("python.res.fm.templating" ${PYTHON_INSTALL_PREFIX}/res/fm/templating "${PYTHON_SOURCES}" True)
