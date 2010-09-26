#include <Python.h>



static PyObject * pycfile( PyObject * self ,  PyObject * args) {
  PyObject * PyFile;
  if (!PyArg_ParseTuple( args , "O" , &PyFile ))
    return NULL;

  {
    FILE * cfile          = PyFile_AsFile( PyFile );

    if (cfile != NULL) 
      return Py_BuildValue( "l" , cfile );  /* The pointer value itself is packed as a long and returned. */
    else
      return Py_BuildValue( "" );  /* None */
  }
}



static PyMethodDef pycfilemethods[] = {
  {"pycfile" , pycfile , METH_VARARGS , "Documentation"},
  {NULL , NULL , 0 , NULL}
};



PyMODINIT_FUNC
initpycfile( void ) {
  (void) Py_InitModule("pycfile" , pycfilemethods);
}
