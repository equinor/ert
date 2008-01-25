#ifndef __NODE_DATA_H__
#define __NODE_DATA_H__

typedef const void * (  copyc_type) (const void *);
typedef void         (  del_type)   (void *);

typedef struct {
  int    byte_size;
  void  *data;
} node_data_type;


void         node_data_free(void *);
const void * node_data_copyc(const void *);

#endif
