#include <stdio.h>
#include <stdlib.h>
#include <node_ctype.h>
#include <stdlib.h>


const char * node_ctype_name(node_ctype ctype) {
  const char * name;
  switch (ctype) {
  case(void_buffer):
    name =  "void buffer";
    break;
  case(void_pointer):
    name =  "void pointer";
    break;
  case(int_value):
    name =  "integer value";
    break;
  case(double_value):
    name =  "double value";
    break;
  case(float_value):
    name =  "float_value";
    break;
  case(char_value):
    name =  "char value";
    break;
  case(bool_value):
    name =  "bool value";
    break;
  case(size_t_value):
    name =  "size_t value";
    break;
  default:
    fprintf(stderr,"%s: fatal internal error node_ctype:%d not recognized - aborting. \n", __func__ , ctype);
    abort();
  }
  return name;
}
