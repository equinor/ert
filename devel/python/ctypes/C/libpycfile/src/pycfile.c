#include <Python.h>

/**
   The purpose of this module is to be able to call C functions which
   expect FILE * input with a Python file handle. This is based on the
   normal Python/C api and not ctypes, the Python module cfile.py
   handles the ctypes side of things.

   What happens is essentially:
   ----------------------------

     1. Python filehandle is created with fileH = open( name , mode).

     2. The pycfile() function below is called with this filehandle as
        input; the pycfile function extracts the underlying FILE *
        pointer with the PyFile_AsFile() function and returns it as an
        opaque pointer.

     3. This opaque pointer is then wrapped with the normal ctypes
        approach in the cfile module, and passed on to C functions
        expecting FILE * input.
*/




static PyObject * pycfile( PyObject * self ,  PyObject * args) {
  PyObject * PyFile;
  if (!PyArg_ParseTuple( args , "O" , &PyFile ))
    return NULL;

  {
    FILE * cfile          = PyFile_AsFile( PyFile );

    if (cfile != NULL) 
      return Py_BuildValue( "l" , cfile );  /* The pointer value itself is packed as a long and returned. */
    else
      return Py_BuildValue( "" );           /* None */
  }
}



static PyMethodDef pycfilemethods[] = {
  {"pycfile" , pycfile , METH_VARARGS , "Will extract and return the FILE* structure of a Python filehandle."},
  {NULL , NULL , 0 , NULL}
};



PyMODINIT_FUNC initpycfile( void ) {
  (void) Py_InitModule("pycfile" , pycfilemethods);
}
