#ifndef __HASH_H__
#define __HASH_H__
#ifdef __cplusplus
extern"C" {
#endif

#include <stdlib.h>
typedef struct hash_struct hash_type;

#include <hash_node.h>

#define HASH_GET_SCALAR(FUNC,TYPE)    	   TYPE   FUNC(const hash_type *, const char *)
#define HASH_GET_ARRAY_PTR(FUNC,TYPE) 	   TYPE * FUNC(const hash_type *, const char *)
#define HASH_INSERT_SCALAR(FUNC,TYPE)      void FUNC(hash_type * , const char * , TYPE )
#define HASH_INSERT_ARRAY(FUNC,TYPE)       void FUNC(hash_type * , const char * , TYPE * , int)
#define HASH_NODE_AS(FUNC,TYPE)            TYPE FUNC(const hash_node_type * node)


void 		  hash_lock  (hash_type * );
void 		  hash_unlock(hash_type * );
hash_type   	* hash_alloc();
void              hash_iter_complete(hash_type * );
void        	  hash_free(hash_type *);
void              hash_free__(void *);
void        	  hash_insert_ref(hash_type * , const char * , const void *);
void        	  hash_insert_copy(hash_type *, const char * , const void *, copyc_type *, del_type *);
void        	  hash_insert_string(hash_type *, const char *, const char *);
bool        	  hash_has_key(const hash_type *, const char *);
void        	* hash_get(const hash_type *, const char *);
const char      * hash_get_string(const hash_type * , const char *);
void        	  hash_del(hash_type *, const char *);
void              hash_clear(hash_type *);
int         	  hash_get_size(const hash_type *);
void        	  hash_set_keylist(const hash_type * , char **);
char           ** hash_alloc_keylist(hash_type *);

char           ** hash_alloc_sorted_keylist (hash_type *hash , int ( hash_get_cmp_value ) (const void *));
char           ** hash_alloc_key_sorted_list(hash_type *hash, int (*cmp)(const void *, const void *));
bool              hash_key_list_compare( hash_type * hash1, hash_type * hash2);
void              hash_insert_hash_owned_ref(hash_type *, const char * , const void *, del_type *);

/*
  hash_node_type  * hash_iter_init(const hash_type *);
  hash_node_type  * hash_iter_next(const hash_type *, const hash_node_type * );
---
  const char 	* hash_iter_get_next_key(hash_type * );
  const char 	* hash_iter_get_first_key(hash_type * );
  void            * hash_iter_get_first(hash_type * , bool *);
  void            * hash_iter_get_next(hash_type *  , bool *);
*/


HASH_GET_SCALAR(hash_get_int       , int);
HASH_GET_SCALAR(hash_get_double    , double);
HASH_INSERT_SCALAR(hash_insert_int    , int);
HASH_INSERT_SCALAR(hash_insert_double , double);
HASH_INSERT_ARRAY(hash_insert_int_array    , int);
HASH_INSERT_ARRAY(hash_insert_double_array , double);
HASH_GET_ARRAY_PTR(hash_get_double_ptr , double);
HASH_GET_ARRAY_PTR(hash_get_int_ptr    , int);

HASH_NODE_AS(hash_node_as_int    , int);
HASH_NODE_AS(hash_node_as_double , double);

#undef HASH_GET_SCALAR
#undef HASH_INSERT_SCALAR
#undef HASH_INSERT_ARRAY
#undef HASH_GET_ARRAY_PTR
#undef HASH_NODE_AS

#ifdef __cplusplus
}
#endif
#endif
