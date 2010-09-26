#include <Python.h>



FILE * cfile( void * PyFile ) {
  PyObject * arg = (PyObject *) PyFile;
  return PyFile_AsFile( PyFile );
}

