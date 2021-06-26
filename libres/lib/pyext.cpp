#include <Python.h>

static struct PyModuleDef module_ = {
    PyModuleDef_HEAD_INIT,
    "res._lib",
    "Python interface for the fputs C library function",
    -1,
};

PyMODINIT_FUNC PyInit__lib() { return PyModule_Create(&module_); }
