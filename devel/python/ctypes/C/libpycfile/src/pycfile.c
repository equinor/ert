/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'pycfile.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

/* 
   The resulting pycfile.so will be bound to the python version the
   "Python.h" header file used by this include statement. If there is
   a version mismatch initialization of the pycfile module will fail.
*/
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

   The main part of ert-python consist either solely of C code in the
   form of shared libraries, or as pure Python code; i.e. there is no
   dependance on the Python/C API. A fortunate consequence of this is
   that ert-python is independent of Python versions; unfortunately
   this particular file is an exception. It is built using the
   Python/C API and is linked to the Python version of the Python.h
   header file used when compiling.  
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
