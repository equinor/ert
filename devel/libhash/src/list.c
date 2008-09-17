#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <list_node.h>
#include <list.h>


struct list_struct {
  int length;
  list_node_type * head;
  list_node_type * tail;
};




/*****************************************************************/


static void list_append_node(list_type *list , list_node_type *new_node) {
  if (list->head == NULL) 
    list->head = new_node;
  else 
    list_node_link(list->tail , new_node);
  list->tail = new_node;
  list->length++;
}


static void list_insert_node_after(list_type *list , list_node_type *after_this, list_node_type *new_node) {
  if (after_this == NULL) {
    fprintf(stderr,"%s: after_this == NULL - aborting \n",__func__);
    abort();
  }
  {
    list_node_type *old_next = list_node_get_next(after_this);
    list_node_link(after_this , new_node);
    if (old_next == NULL) 
      list->tail = new_node;
    else
      list_node_link(new_node   , old_next);
  }

  list->length++;
}


static void list_insert_node_before(list_type *list , list_node_type *before_this , list_node_type *new_node) {
  if (before_this == NULL) {
    fprintf(stderr,"%s: before_this == NULL - aborting \n",__func__);
    abort();
  }

  {
    list_node_type *old_prev = list_node_get_prev(before_this);
    list_node_link(new_node , before_this);
    if (old_prev == NULL)
      list->head = new_node;
    else
      list_node_link(old_prev , new_node);
  }
  
  list->length++;
}


static list_node_type * list_iget_node_static(const list_type *list, int index, bool abort_on_error) {
  list_node_type *node = list->head;
  int n = 0;
  
  
  while ((node != NULL) && (n < index)) {
    node = list_node_get_next(node);
    n++;
  }

  if (node == NULL && abort_on_error) {
    fprintf(stderr,"%s: element:%d does not exist in list - aborting \n",__func__ , index);
    abort();
  }
  
  return node;
}


list_node_type * list_iget_node(const list_type *list, int index) {

  return list_iget_node_static(list , index, true);

}



void * list_iget_node_value_ptr(const list_type * list, int index)
{
  list_node_type * list_node = list_iget_node(list, index);
  return list_node_value_ptr(list_node);
}



list_node_type * list_iget_node_try(const list_type *list, int index) {
  return list_iget_node_static(list , index, false);
}



/*****************************************************************/
/* Functions which are exported follow below here. */
/*****************************************************************/


void list_del_node(list_type *list , list_node_type *del_node) {
  if (del_node == NULL) {
    fprintf(stderr,"%s: tried to delete NULL node - aborting \n",__func__);
    abort();
  }
  {
    list_node_type *prev_node = list_node_get_prev(del_node);
    list_node_type *next_node = list_node_get_next(del_node);
    list_node_link(prev_node , next_node);
    if (del_node == list->head)
      list->head = next_node;

    if (del_node == list->tail)
      list->tail = prev_node;
    
    list_node_free(del_node);
    list->length--;
  }
}





#define LIST_SCALAR_APPEND(FUNC,TYPE) \
void FUNC(list_type *list , TYPE value) {                               \
  list_node_type *node = list_node_alloc_managed(&value , sizeof value); \
  list_append_node(list ,  node) ;                                       \
}


#define LIST_ARRAY_APPEND(FUNC,TYPE)                                   \
void FUNC(list_type *list, TYPE *value, int SIZE) {  \
  list_node_type *node = list_node_alloc_managed(&value , SIZE * sizeof *value); \
  list_append_node(list ,  node);                                       \
}


#define LIST_SCALAR_AFTER(FUNC,TYPE)                                     \
void FUNC(list_type *list , list_node_type *after_this , TYPE value) {   \
  list_node_type *node = list_node_alloc_managed(&value , sizeof value); \
  list_insert_node_after(list , after_this , node);                     \
}

#define LIST_SCALAR_BEFORE(FUNC,TYPE)                                     \
void FUNC(list_type *list , list_node_type *before_this , TYPE value) {   \
  list_node_type *node = list_node_alloc_managed(&value , sizeof value);   \
  list_insert_node_before(list , before_this , node);                     \
}






void list_free(list_type *list) {
  if (list->head != NULL) {
    list_node_type *node , *next_node;
    node = list->head;
    while (node != NULL) {
      next_node = list_node_get_next(node);
      list_node_free(node);
      node = next_node;
    }
  }
  free(list);
}


list_type * list_alloc(void) {
  list_type * list;
  list = malloc(sizeof *list);
  list->length = 0;
  list->head   = NULL;
  list->tail   = NULL;
  return list;
}


list_node_type * list_append_ref(list_type *list , const void *value) {
  list_node_type *node = list_node_alloc(value , NULL , NULL);
  list_append_node(list , node);
  return node;
}


list_node_type * list_append_list_owned_ref(list_type *list , const void *value , del_type *del) {
  list_node_type *node = list_node_alloc(value , NULL , del);
  list_append_node(list , node);
  return node;
}


list_node_type * list_append_copy(list_type *list , const void *value , copyc_type *copyc , del_type *del) {
  list_node_type *node = list_node_alloc(value , copyc , del);
  list_append_node(list , node);
  return node;
}

list_node_type * list_append_string_copy(list_type *list , const char * s) {
  list_node_type * node = list_node_alloc_managed(s , strlen(s) + 1);
  list_append_node(list , node);
  return node;
}

  

void * list_iget(const list_type *list, int index) {
  list_node_type * node = list_iget_node(list , index);
  return list_node_value_ptr(node);
}

list_node_type * list_get_head(const list_type *list) { return list->head; }
list_node_type * list_get_tail(const list_type *list) { return list->tail; }

int list_get_size(const list_type *list) { return list->length; }

int list_iget_int(const list_type *list , int index) {
  const list_node_type *node = list_iget_node(list , index);
  return list_node_as_int(node);
}

double list_iget_double(const list_type *list , int index) {
  const list_node_type *node = list_iget_node(list , index);
  return list_node_as_double(node);
}




LIST_SCALAR_APPEND(list_int_append          , int)
LIST_SCALAR_APPEND(list_double_append       , double)
LIST_ARRAY_APPEND (list_int_array_append    , int)
LIST_ARRAY_APPEND (list_double_array_append , double)
LIST_SCALAR_AFTER (list_int_insert_after    , int)
LIST_SCALAR_AFTER (list_double_insert_after , double)







