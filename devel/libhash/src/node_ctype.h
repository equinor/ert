#ifndef __NODE_CTYPE_H__
#define __NODE_CTYPE_H__
#ifdef __cplusplus
extern "C" {
#endif


/*
  value  : means a scalar which has been packed into the container object.
  pointer: means a (typed) pointer which points to a memory location outside the container object (however the container can own the memory).
*/

typedef enum  {void_pointer = 1,
	       int_value    = 2, 
	       double_value = 3, 
	       float_value  = 4 , 
	       char_value   = 5 , 
	       bool_value   = 6 , 
	       size_t_value = 7 ,
	       invalid_ctype = 100} node_ctype;


const char * node_ctype_name(node_ctype );
#ifdef __cplusplus
}
#endif
#endif
