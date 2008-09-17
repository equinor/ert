#ifndef __LIST_H__
#define __LIST_H__

#include <list_node.h>

typedef struct list_struct list_type;

list_type      * list_alloc(void);
void             list_del_node(list_type *list , list_node_type *);
/*
  void             list_add_node(list_type *, list_node_type *);
*/
void             list_free(list_type *);
list_node_type * list_iget_node(const list_type *, int );
void           * list_iget_node_value_ptr(const list_type *, int );
list_node_type * list_get_head(const list_type *);
list_node_type * list_get_tail(const list_type *);
list_node_type * list_append_ref(list_type *list , const void *);
list_node_type * list_append_list_owned_ref(list_type *, const void *, del_type *);
int              list_get_size(const list_type *);
list_node_type * list_append_string_copy(list_type *, const char * );
list_node_type * list_append_copy(list_type *, const void *, copyc_type *, del_type *);
#endif
