#ifndef __HASH_SLL_H__
#define __HASH_SLL_H__

#include <hash_node.h>

typedef struct hash_sll_struct hash_sll_type;

hash_sll_type  **hash_sll_alloc_table(int );
/*hash_sll_type *  hash_sll_alloc(void);*/
void             hash_sll_del_node(hash_sll_type * , hash_node_type *);
void             hash_sll_add_node(hash_sll_type *, hash_node_type *);
void             hash_sll_free(hash_sll_type *);
bool             hash_sll_has_key(const hash_sll_type *, uint32_t , const char *);
bool             hash_sll_empty(const hash_sll_type * hash_sll);
hash_node_type * hash_sll_get(const hash_sll_type *, uint32_t , const char *);
hash_node_type * hash_sll_get_head(const hash_sll_type *);
#endif
