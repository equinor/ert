#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <node_data.h>
#include <node_ctype.h>


/*
  Used to hold typed storage.
*/


struct node_data_struct {
  node_ctype        ctype;
  int               byte_size;
  void             *data;
};



node_data_type * node_data_alloc(int byte_size , const void * data) {
  node_data_type * node = malloc(sizeof * node);
  node->data            = (void *) data;
  node->byte_size       = byte_size;
  node->ctype           = void_buffer;
  return node;
}


const void * node_data_copyc(const void *_src) {
  const node_data_type *src = (const node_data_type *) _src;
  node_data_type *new;
  new = malloc(sizeof *new);
  new->ctype     = src->ctype;
  new->byte_size = src->byte_size;
  new->data      = malloc(new->byte_size);
  if (new->data == NULL) {
    fprintf(stderr,"%s: allocation of %d bytes failed - aborting \n",__func__ , new->byte_size);
    abort();
  }
  memcpy(new->data , src->data , new->byte_size);
  return new;
}


void node_data_free(void *_node_data) {
  node_data_type *node_data = (node_data_type *) _node_data;
  free(node_data->data);
  free(node_data);
}


const void * node_data_get_data(const node_data_type * node_data) {
  return node_data->data;
}
