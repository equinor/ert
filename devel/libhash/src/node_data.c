#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <node_data.h>


const void * node_data_copyc(const void *_src) {
  const node_data_type *src = (const node_data_type *) _src;
  node_data_type *new;
  new = malloc(sizeof *new);
  new->byte_size = src->byte_size;
  new->data = malloc(new->byte_size);
  memcpy(new->data , src->data , new->byte_size);

  return new;
}


void node_data_free(void *_node_data) {
  node_data_type *node_data = (node_data_type *) _node_data;
  free(node_data->data);
  free(node_data);
}
