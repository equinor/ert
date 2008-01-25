#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <inttypes.h>
#include <hash_node.h>




struct hash_node_struct {
  char 	      	   *key;
  uint32_t     	    global_index;
  uint32_t     	    table_index;
  const void       *value;
  
  /*
    For the copy constructor and delete
    operator the logic is very simple:

    if they are present they are used.
  */
  copyc_type  	 *copyc;
  del_type    	 *del;

  hash_node_type *next_node;
};
  


bool hash_node_key_eq(const hash_node_type * node , uint32_t global_index , const char *key) {
  bool eq;
  eq = true;
  if (global_index != node->global_index)
    eq = false;
  else if (strcmp(node->key , key) != 0)
    eq = false;
  return eq;
}


uint32_t hash_node_get_table_index(const hash_node_type * node)  { return node->table_index; }
uint32_t hash_node_get_global_index(const hash_node_type * node) { return node->global_index; }
const char * hash_node_get_keyref(const hash_node_type * node)   { return node->key; }

hash_node_type * hash_node_get_next(const hash_node_type * node) { return node->next_node; }


void * hash_node_value_ptr(const hash_node_type *node) { return (void *) node->value; }

void hash_node_set_next(hash_node_type *node , const hash_node_type *next_node) {
  node->next_node = (hash_node_type *) next_node;
}


uint32_t hash_node_set_table_index(hash_node_type *node , uint32_t table_size) {
  node->table_index = (node->global_index % table_size);
  return node->table_index;
}


hash_node_type * hash_node_alloc_new(const char *key, const void *value , copyc_type *copyc , del_type *del , hashf_type *hashf , uint32_t table_size) {
  hash_node_type *node;
  node      = malloc(sizeof *node);
  node->key = calloc(strlen(key) + 1 , sizeof *node->key);
  strcpy(node->key , key);
  node->copyc       = copyc;
  node->del         = del;
  if (node->copyc != NULL) 
    node->value = node->copyc(value);
  else
    node->value   = value;
  node->next_node = NULL;
	
  node->global_index = hashf(node->key , strlen(node->key));
  hash_node_set_table_index(node , table_size);
  return node;
}

void hash_node_printf_key(const hash_node_type *node) {
  printf("%u/%u  %s \n",node->table_index , node->global_index , node->key);
}



void hash_node_free(hash_node_type *node) {
  free(node->key);
  if (node->del != NULL) 
    node->del((void *) node->value);
  free(node);
}



