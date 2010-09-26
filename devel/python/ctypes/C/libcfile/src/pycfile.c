#include <Python.h>



static FILE * cfile( PyObject * self ,  PyObject * arg) {
  return PyFile_AsFile( arg );
}



static PyMethodDef cfilemethods[] = {
  {"cfile" , cfile , METH_VARARGS , "Documentation"},
  {NULL , NULL , 0 , NULL}
};



PyMODINIT_FUNC
initcfile( void ) {
  (void) Py_InitModule("cfile" , cfilemethods);
}
