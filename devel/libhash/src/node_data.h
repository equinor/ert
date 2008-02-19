#ifndef __NODE_DATA_H__
#define __NODE_DATA_H__

typedef const void * (  copyc_type) (const void *);
typedef void         (  del_type)   (void *);

typedef enum  {__buffer_value = 1, __int_value = 2, __double_value = 3, __float_value = 4 , __char_value = 5 , __bool_value = 6 , __size_t_value = 7} node_data_ctype;

typedef struct {
  node_data_ctype   ctype;
  int               byte_size;
  void             *data;
} node_data_type;


void         node_data_free(void *);
const void * node_data_copyc(const void *);

#endif
