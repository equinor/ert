#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <dirent.h>
#include <pthread.h>
#include <hash.h>
#include <hash_sll.h>
#include <hash_node.h>
#include <node_data.h>

#define HASH_DEFAULT_SIZE 16

typedef enum {iter_invalid , iter_active , iter_complete} __iter_mode;


struct hash_struct {
  pthread_mutex_t  iter_mutex;
  uint32_t         size;
  uint32_t         elements;
  double           resize_fill;
  hash_sll_type  **table;
  hashf_type      *hashf;

  char           **iter_keylist;
  __iter_mode      iter_mode;
  int              iter_index , iter_size;
};


typedef struct hash_sort_node {
  char     *key;
  int       cmp_value;
} hash_sort_type;



/*****************************************************************/

#define HASH_GET_SCALAR(FUNC,TYPE) \
TYPE FUNC (const hash_type *hash,  const char *key) { \
   node_data_type *node_data = hash_get(hash , key); \
   return *((TYPE *) node_data_get_data(node_data)); \
}


#define HASH_INSERT_SCALAR(FUNC,TYPE) \
void FUNC(hash_type *hash , const char *key , TYPE value) {     \
  hash_insert_managed_copy(hash , key , &value , sizeof value); \
}


#define HASH_INSERT_ARRAY(FUNC,TYPE)                                   \
void FUNC(hash_type *hash, const char *key , TYPE *value, int SIZE) {  \
  hash_insert_managed_copy(hash , key , value , SIZE * sizeof *value); \
}

#define HASH_GET_ARRAY_PTR(FUNC,TYPE)\
TYPE * FUNC(const hash_type * hash, const char *key) { \
   node_data_type *node_data = hash_get(hash , key);   \
   return ((TYPE *) node_data_get_data(node_data));    \
}

/*****************************************************************/

static char * alloc_string_copy(const char *src) {
  char *new = malloc(strlen(src) + 1);
  strcpy(new , src);
  return new;
}


static void * __hash_get_node(const hash_type *hash , const char *key, bool abort_on_error) {
  const uint32_t global_index = hash->hashf(key , strlen(key));
  const uint32_t table_index  = (global_index % hash->size);
  hash_node_type *node        = hash_sll_get(hash->table[table_index] , global_index , key);
  if (node != NULL) {
    /*hash_node_assert_type(node , data_type); */
    return node;
  } else {
    if (abort_on_error) {
      fprintf(stderr,"%s: tried to get from key:%s which does not exist - aborting \n",__func__ , key);
      abort();
    } else return NULL;
  }
}



void hash_del(hash_type *hash , const char *key) {
  const uint32_t global_index = hash->hashf(key , strlen(key));
  const uint32_t table_index  = (global_index % hash->size);
  hash_node_type *node        = hash_sll_get(hash->table[table_index] , global_index , key);

  if (node == NULL) {
    fprintf(stderr,"%s: hash does not contain key:%s - aborting \n",__func__ , key);
    abort();
  } else
    hash_sll_del_node(hash->table[table_index] , node);

  hash->elements--;
  hash->iter_mode = iter_invalid;
}


void hash_clear(hash_type *hash) {
  int old_size = hash_get_size(hash);
  if (old_size > 0) {
    char **keyList = hash_alloc_keylist(hash);
    int i;
    for (i=0; i < old_size; i++) {
      hash_del(hash , keyList[i]);
      free(keyList[i]);
    }
    free(keyList);
  }
}



void * hash_get(const hash_type *hash , const char *key) {
  hash_node_type * node = __hash_get_node(hash , key , true);
  return hash_node_value_ptr(node);
}



HASH_GET_SCALAR(hash_get_int    , int)
HASH_GET_SCALAR(hash_get_double , double)
HASH_GET_ARRAY_PTR(hash_get_double_ptr , double)
HASH_GET_ARRAY_PTR(hash_get_int_ptr    , int)

     /*
       HASH_NODE_AS(hash_node_as_int    , int)
       HASH_NODE_AS(hash_node_as_double , double)
     */

static uint32_t hash_index(const uint8_t *key, size_t len) {
  uint32_t hash = 0;
  size_t i;

  for (i=0; i < len; i++) {
    hash += key[i];
    hash += (hash << 10);
    hash ^= (hash >>  6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}



static hash_type * __hash_alloc(int size, double resize_fill , hashf_type *hashf) {
  hash_type* hash;
  hash = malloc(sizeof *hash);
  hash->size  	 = size;
  hash->hashf 	 = hashf;
  hash->table 	 = hash_sll_alloc_table(hash->size);
  hash->elements = 0;
  hash->resize_fill  = resize_fill;
  hash->iter_mode    = iter_invalid;
  hash->iter_keylist = NULL;
  pthread_mutex_init( &hash->iter_mutex , NULL);
  return hash;
}


static void hash_resize(hash_type *hash, int new_size) {
  hash_sll_type **new_table = hash_sll_alloc_table(new_size);
  hash_node_type *node;
  int i;
  
  for (i=0; i < hash->size; i++) {
    node = hash_sll_get_head(hash->table[i]);
    while (node != NULL) {
      uint32_t new_table_index  = hash_node_set_table_index(node , new_size);
      hash_node_type *next_node = hash_node_get_next(node);
      hash_sll_add_node(new_table[new_table_index] , node);
      node = next_node;
    }
  }
  
  /* 
     Only freeing the table structure, *NOT* calling the node_free() functions,. which
     happens when hash_sll_free() is called.
  */
  for (i=0; i < hash->size; i++) 
    free(hash->table[i]);
  
  free(hash->table);
  hash->size     = new_size;
  hash->table    = new_table;
}



hash_type * hash_alloc() {
  return __hash_alloc(HASH_DEFAULT_SIZE , 0.50 , hash_index);
}


static void hash_iter_free_keylist(hash_type * hash) {
  {
    int i;
    if (hash->iter_keylist != NULL) {
      for (i=0; i < hash->iter_size; i++) {
		if (hash->iter_keylist[i] != NULL)
	  		free(hash->iter_keylist[i]);
      }
      free(hash->iter_keylist);
    }
  }
  hash->iter_keylist = NULL;
  hash->iter_mode    = iter_invalid;
}


void hash_iter_complete(hash_type * hash) {
   if (hash->iter_mode == iter_active) {
     hash->iter_mode = iter_complete;
     pthread_mutex_unlock( &hash->iter_mutex );
   }
}
	


const char * hash_iter_get_next_key(hash_type * hash) {
  if (hash->iter_mode == iter_active) {
    if (hash->iter_index == hash_get_size(hash)) {
      hash_iter_complete(hash);
      return NULL;
    } else {
      const char * key = hash->iter_keylist[hash->iter_index];
      hash->iter_index++;
      return key;
    }
  } else {
    fprintf(stderr,"%s: called with iter_mode == iter_complete or iter_invalid - aborting \n",__func__);
    abort();
  }
}


const char * hash_iter_get_first_key(hash_type * hash) {
  if (hash->iter_mode == iter_invalid || hash->iter_mode == iter_complete) 
    hash_iter_free_keylist(hash);
 
  pthread_mutex_lock( &hash->iter_mutex );
  hash->iter_keylist = hash_alloc_keylist(hash);
  hash->iter_mode  = iter_active;
  hash->iter_index = 0;
  hash->iter_size  = hash_get_size(hash);
  return hash_iter_get_next_key(hash);
}


void * hash_iter_get_first(hash_type * hash , bool *valid) {
  const char * key = hash_iter_get_first_key(hash);
  if (key != NULL) {
    *valid = true;
    return hash_get(hash , key);
  } else {
    *valid = false;
    return NULL;
  }
}


void * hash_iter_get_next(hash_type * hash , bool *valid) {
  const char * key = hash_iter_get_next_key(hash);
  if (key != NULL) {
    *valid = true;
    return hash_get(hash , key);
  } else {
    *valid = false;
    return NULL;
  }
}


void hash_free(hash_type *hash) {
  int i;
  for (i=0; i < hash->size; i++) 
    hash_sll_free(hash->table[i]);
  free(hash->table);
  hash_iter_free_keylist(hash);
  free(hash);
}


void hash_free__(void *void_hash) {
  hash_free((hash_type *) void_hash);
}


static void hash_insert_node(hash_type *hash , hash_node_type *node) {
  uint32_t table_index = hash_node_get_table_index(node);
  {
    /*
      If a node with the same key already exists in the table
      it is removed.
    */
    hash_node_type *existing_node = __hash_get_node(hash , hash_node_get_keyref(node) , false);
    if (existing_node != NULL) {
      hash_sll_del_node(hash->table[table_index] , existing_node);
      hash->elements--;
    } 
  }
  
  hash_sll_add_node(hash->table[table_index] , node);  
  hash->elements++;
  if ((1.0 * hash->elements / hash->size) > hash->resize_fill) 
    hash_resize(hash , hash->size * 2);
  
  hash->iter_mode = iter_invalid;
}


static void hash_insert_managed_copy(hash_type *hash, const char *key, const void *value_ptr , int value_size) {
  hash_node_type *node; 
  node_data_type *node_data = node_data_alloc(value_size , value_ptr);
  node = hash_node_alloc_new(key , node_data , node_data_copyc , node_data_free , hash->hashf , hash->size);
  hash_insert_node(hash , node);

  /* This only frees the container - actual storage is freed when the hash table is deleted */
  free(node_data);
}




/*****************************************************************/

void hash_insert_copy(hash_type *hash , const char *key , const void *value , copyc_type *copyc , del_type *del) {
  hash_node_type *node;
  if (copyc == NULL || del == NULL) {
    fprintf(stderr,"%s: must provide copy constructer and delete operator for insert copy - aborting \n",__func__);
    abort();
  }
  node = hash_node_alloc_new(key , value , copyc , del , hash->hashf , hash->size);
  hash_insert_node(hash , node);
}

void hash_insert_ref(hash_type *hash , const char *key , const void *value) {
  hash_node_type *node;
  node = hash_node_alloc_new(key , value , NULL , NULL , hash->hashf , hash->size );
  hash_insert_node(hash , node);
}


/**
This function will insert a reference "value" with key "key"; when the
key is deleted - either by an explicit call to hash_del(), or when the
complete hash table is free'd with hash_free(), the destructur 'del'
is called with 'value' as argument.

It is importand to realize that when elements are inserted into a hash
table with this function the calling scope gives up responsibility of
freeing the memory pointed to by value.
*/

void hash_insert_hash_owned_ref(hash_type *hash , const char *key , const void *value , del_type *del) {
  hash_node_type *node;
  if (del == NULL) {
    fprintf(stderr,"%s: must provide delete operator for insert hash_owned_ref - aborting \n",__func__);
    abort();
  }
  node = hash_node_alloc_new(key , value , NULL , del , hash->hashf , hash->size);
  hash_insert_node(hash , node);
}



void hash_insert_string(hash_type *hash, const char *key , const char *value) {
  hash_insert_managed_copy(hash , key , value , strlen(value) + 1);
}




HASH_INSERT_SCALAR(hash_insert_int          , int)
HASH_INSERT_SCALAR(hash_insert_double       , double)
HASH_INSERT_ARRAY (hash_insert_int_array    , int)
HASH_INSERT_ARRAY (hash_insert_double_array , double)



const char * hash_get_string(const hash_type *hash , const char *key) {
  node_data_type *node_data = hash_get(hash , key);
  if (node_data != NULL)
    return node_data_get_data(node_data);
  else
    return NULL;
}


bool hash_has_key(const hash_type *hash , const char *key) {
  if (__hash_get_node(hash , key , false) == NULL)
    return false;
  else
    return true;
}





static hash_node_type * hash_iter_init(const hash_type *hash) {
  uint32_t i = 0;
  while (i < hash->size && hash_sll_empty(hash->table[i]))
    i++;
	
  if (i < hash->size) 
    return hash_sll_get_head(hash->table[i]);
  else
    return NULL;
}


static hash_node_type * hash_iter_next(const hash_type *hash , const hash_node_type * node) {
  hash_node_type *next_node = hash_node_get_next(node);
  if (next_node == NULL) {
    const uint32_t table_index = hash_node_get_table_index(node);
    if (table_index < hash->size) {
      uint32_t i = table_index + 1;
      while (i < hash->size && hash_sll_empty(hash->table[i]))
		i++;
      if (i < hash->size) 
		next_node = hash_sll_get_head(hash->table[i]);
    }
  }
  return next_node;
}


void hash_printf_keys(hash_type *hash) {
  const char * key = hash_iter_get_first_key(hash);
  while (key != NULL) {
  	printf("%s \n",key);
  	key = hash_iter_get_next_key(hash);
  }
}


char ** hash_alloc_keylist(const hash_type *hash) {
  char **keylist;
  int i = 0;
  hash_node_type *node;
  keylist = calloc(hash->elements , sizeof *keylist);
  node = hash_iter_init(hash);
  while (node != NULL) {
    const char *key = hash_node_get_keyref(node); 
    keylist[i] = alloc_string_copy(key);
    node = hash_iter_next(hash , node);
    i++;
  }
  return keylist;
}



/*
void hash_set_keylist(const hash_type *hash , char **keylist) {
  int i = 0;
  hash_node_type *node;
  node = hash_iter_init(hash);
  while (node != NULL) {
    const char *key = hash_node_get_keyref(node); 
    int len = strlen(key);
    strcpy(keylist[i] , key);
    keylist[i][len] = '\0';
    node = hash_iter_next(hash , node);
    i++;
  }
}
*/

int hash_get_size(const hash_type *hash) { 
  return hash->elements; 
}

/******************************************************************/
/* Sorting start */

static hash_sort_type * hash_alloc_sort_list(const hash_type *hash , const char **keylist) {
  int i;
  hash_sort_type * sort_list = calloc(hash_get_size(hash) , sizeof * sort_list);
  for (i=0; i < hash_get_size(hash); i++) 
    sort_list[i].key       = alloc_string_copy(keylist[i]);

  return sort_list;
}

static void hash_free_sort_list(const hash_type *hash , hash_sort_type *sort_list) {
  int i;
  for (i=0; i < hash_get_size(hash); i++) 
    free(sort_list[i].key);
  free(sort_list);
}


static int hash_sortlist_cmp(const void *_p1 , const void  *_p2) {
  const hash_sort_type *p1 = (const hash_sort_type *) _p1;
  const hash_sort_type *p2 = (const hash_sort_type *) _p2;

  if (p1->cmp_value == p2->cmp_value)
    return 0;
  else if (p1->cmp_value < p2->cmp_value)
    return -1;
  else
    return 1;
}


static char ** __hash_alloc_ordered_keylist(const hash_type *hash , int ( hash_get_cmp_value) (const void * )) {    
  int i;
  char **sorted_keylist;
  char **tmp_keylist   = hash_alloc_keylist(hash);
  hash_sort_type * sort_list = hash_alloc_sort_list(hash , (const char **) tmp_keylist);

  for (i = 0; i < hash_get_size(hash); i++) 
    sort_list[i].cmp_value = hash_get_cmp_value( hash_get(hash , sort_list[i].key) );
  
  qsort(sort_list , hash_get_size(hash) , sizeof *sort_list , &hash_sortlist_cmp);
  sorted_keylist = calloc(hash_get_size(hash) , sizeof *sorted_keylist);
  for (i = 0; i < hash_get_size(hash); i++) 
    sorted_keylist[i] = alloc_string_copy(sort_list[i].key);
  
  hash_free_ext_keylist(hash , tmp_keylist);
  hash_free_sort_list(hash , sort_list);
  return sorted_keylist;
}


char ** hash_alloc_sorted_keylist(const hash_type *hash , int ( hash_get_cmp_value) (const void *)) {
  return __hash_alloc_ordered_keylist(hash , hash_get_cmp_value);
}


/* Sorting end */
/******************************************************************/


/*
static int key_cmp(const void *_s1 , const void *_s2) {
  const char * s1 = *((const char **) _s1);
  const char * s2 = *((const char **) _s2);
  
  return strcmp(s1 , s2);
}

static char ** __hash_alloc_sorted_keylist(const hash_type *hash, int (cmp) (const void * , const void *)) { 
  char **keylist = hash_alloc_keylist(hash);
  
  qsort(keylist , hash_get_size(hash) , sizeof *keylist , &cmp);
  return keylist;
}


char ** hash_alloc_sorted_keylist( const hash_type *hash , int
  return __hash_alloc_sorted_keylist(hash , key_cmp);
}
*/


void hash_free_ext_keylist(const hash_type *hash , char ** keylist) {
int i;
 for (i = 0; i < hash_get_size(hash); i++) 
    free(keylist[i]);
  free(keylist);
}


#undef HASH_GET_SCALAR
#undef HASH_INSERT_SCALAR
#undef HASH_INSERT_ARRAY
#undef HASH_GET_ARRAY_PTR
#undef HASH_NODE_AS
#undef HASH_DEFAULT_SIZE
