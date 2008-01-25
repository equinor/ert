#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include <inttypes.h>
#include <list_node.h>
#include <node_data.h>



struct list_node_struct {
  const void     *value;
  
  /*
    For the copy constructor and delete
    operator the logic is very simple:
    
    if they are present they are used.
  */
  copyc_type  	 *copyc;
  del_type    	 *del;
  
  list_node_type *prev_node;
  list_node_type *next_node;
};
  






list_node_type * list_node_get_next(const list_node_type * node) { return node->next_node; }
list_node_type * list_node_get_prev(const list_node_type * node) { return node->prev_node; }

void * list_node_value_ptr(const list_node_type *node) { return (void *) node->value; }


void list_node_link(list_node_type *node1 , list_node_type *node2) {
  if (node1 != NULL) node1->next_node = node2;
  if (node2 != NULL) node2->prev_node = node1;
}


list_node_type * list_node_alloc_managed(const void *value_ptr , int value_size) {
  list_node_type *node;
  node_data_type list_data;
  list_data.data      = (void *) value_ptr;
  list_data.byte_size = value_size;
  node = list_node_alloc(&list_data , node_data_copyc , node_data_free);
  return node;
}


list_node_type * list_node_alloc(const void *value , copyc_type *copyc , del_type *del) {
  list_node_type *node;
  node      = malloc(sizeof *node);
  node->copyc       = copyc;
  node->del         = del;
  if (node->copyc != NULL) 
    node->value = node->copyc(value);
  else
    node->value   = value;
  
  node->prev_node = NULL;
  node->next_node = NULL;
  return node;
}


void list_node_free(list_node_type *node) {
  if (node->del != NULL) 
    node->del((void *) node->value);
  free(node);
}


const char * list_node_get_string(list_node_type *node) {
  node_data_type *data = list_node_value_ptr(node);
  return (const char *) data->data;                
}


/*****************************************************************/

#define LIST_NODE_AS_SCALAR(FUNC,TYPE)                    \
TYPE FUNC(const list_node_type * node) {                  \
   node_data_type *data = list_node_value_ptr(node);      \
   return *((TYPE *) data->data);                         \
} 


LIST_NODE_AS_SCALAR(list_node_as_int    , int)
LIST_NODE_AS_SCALAR(list_node_as_double , double)

#undef LIST_NODE_AS_SCALAR
