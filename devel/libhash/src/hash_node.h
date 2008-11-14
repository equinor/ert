#ifndef __HASH_NODE_H__
#define __HASH_NODE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <inttypes.h>
#include <node_data.h>

typedef struct hash_node_struct hash_node_type;
typedef uint32_t     (hashf_type) (const uint8_t *key, size_t len);
typedef enum          {hash_ref_data , hash_int_data , hash_double_data , hash_string_data} hash_data_type;


/*
  void             hash_node_assert_type(const hash_node_type * , hash_data_type );
*/
bool             hash_node_key_eq(const hash_node_type * , uint32_t  , const char *);
hash_node_type * hash_node_get_next(const hash_node_type * );
uint32_t         hash_node_get_insert_nr(const hash_node_type * );
void             hash_node_set_next(hash_node_type * , const hash_node_type * );
hash_node_type * hash_node_alloc_new(const char *, const void *, copyc_type *, del_type *, hashf_type *, uint32_t);
void             hash_node_set_insert_nr(hash_node_type *, uint32_t );
uint32_t         hash_node_get_table_index(const hash_node_type * );
uint32_t         hash_node_get_global_index(const hash_node_type * );
const char *     hash_node_get_keyref(const hash_node_type * );
void           * hash_node_value_ptr(const hash_node_type *);
void             hash_node_free(hash_node_type *);
uint32_t         hash_node_set_table_index(hash_node_type *, uint32_t );
void             hash_node_printf_key(const hash_node_type *);
#ifdef __cplusplus
}
#endif
#endif
