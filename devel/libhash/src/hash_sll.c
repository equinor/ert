#include <stdlib.h>
#include <stdio.h>
#include <hash_node.h>
#include <hash_sll.h>


struct hash_sll_struct {
  int length;
  hash_node_type * head;
};


static hash_sll_type * hash_sll_alloc(void) {
  hash_sll_type * hash_sll = malloc(sizeof *hash_sll);
  hash_sll->length = 0;
  hash_sll->head   = NULL;
  return hash_sll;
}

hash_sll_type **hash_sll_alloc_table(int size) {
  int i;
  hash_sll_type **table;
  table = calloc(size , sizeof *table);
  for (i=0; i<size; i++)
    table[i] = hash_sll_alloc();
  return table;
}


void hash_sll_del_node(hash_sll_type *hash_sll , hash_node_type *del_node) {
  if (del_node == NULL) {
    fprintf(stderr,"%s: tried to delete NULL node - aborting \n",__func__);
    abort();
  }
  {
    hash_node_type *node, *p_node;
    p_node = NULL;
    node   = hash_sll->head;
    while (node != NULL && node != del_node) {
      p_node = node;
      node   = hash_node_get_next(node);
    }

    if (node == del_node) {
      if (p_node == NULL) 
	/* 
	   We are attempting to delete the first element in the list.
	*/
	hash_sll->head = hash_node_get_next(del_node);
      else
	hash_node_set_next(p_node , hash_node_get_next(del_node));
      hash_node_free(del_node);
      hash_sll->length--;
    } else {
      fprintf(stderr,"%s: tried to delete node not in list - aborting \n",__func__);
      abort();
    }
  }
}


hash_node_type * hash_sll_get_head(const hash_sll_type *hash_sll) { return hash_sll->head; }

void hash_sll_add_node(hash_sll_type *hash_sll , hash_node_type *new_node) {
  hash_node_set_next(new_node, hash_sll->head);
  hash_sll->head = new_node;
  hash_sll->length++;
}



void hash_sll_free(hash_sll_type *hash_sll) {
  if (hash_sll->head != NULL) {
    hash_node_type *node , *next_node;
    node = hash_sll->head;
    while (node != NULL) {
      next_node = hash_node_get_next(node);
      hash_node_free(node);
      node = next_node;
    }
  }
  free(hash_sll);
}


bool hash_sll_empty(const hash_sll_type * hash_sll) {
  if (hash_sll->length == 0)
    return true;
  else
    return false;
}


hash_node_type * hash_sll_get(const hash_sll_type *hash_sll, uint32_t global_index , const char *key) {
  hash_node_type *node = hash_sll->head;
  
  while ((node != NULL) && (!hash_node_key_eq(node , global_index , key))) 
    node = hash_node_get_next(node);
  
  return node;
}


bool hash_sll_has_key(const hash_sll_type *hash_sll , uint32_t global_index , const char *key) {
  if (hash_sll_get(hash_sll , global_index , key))
    return true;
  else
    return false;
}


