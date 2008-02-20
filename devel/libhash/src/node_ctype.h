#ifndef __NODE_CTYPE_H__
#define __NODE_CTYPE_H__


/*
  value  : means a scalar which has been packed into the container object.
  buffer : means a vector which has been packed into the container object.
  pointer: means a (typed) pointer which points to a memory location outside the container object.
*/

typedef enum  {void_buffer  = 0,
	       void_pointer = 1,
	       int_value    = 2, 
	       double_value = 3, 
	       float_value  = 4 , 
	       char_value   = 5 , 
	       bool_value   = 6 , 
	       size_t_value = 7} node_ctype;

#endif
