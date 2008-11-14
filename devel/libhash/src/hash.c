#define  _GNU_SOURCE   /* Must define this to get access to pthread_rwlock_t */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <dirent.h>
#include <hash.h>
#include <hash_sll.h>
#include <hash_node.h>
#include <node_data.h>
#include <pthread.h>

#define HASH_DEFAULT_SIZE 16


struct hash_struct {

  uint32_t          size;
  uint32_t          elements;
  double            resize_fill;
  hash_sll_type   **table;
  hashf_type       *hashf;
  pthread_rwlock_t  rwlock;
  
  /** 
      All variables with __iter prefix are internal variables used to
      support iteration over the keys in the hash table.
  */
  char           ** __iter_keylist;
  bool              __iter_active;
  int               __iter_index;
};


typedef struct hash_sort_node {
  char     *key;
  int       cmp_value;
} hash_sort_type;



/**
   This is a seriously homemade locking system. There are three
   different usage scenarios which require locking:

   1. An external application wants to do something with all the
      hash_keys(). This application take a long time to complete, and
      need to ensure that the hash content is unchanged during the
      execution time. This is done with

        hash_lock( hash );

        .......

        hash_unlock( hash );

      It is fully the responsibility of the application level to call
      lock and unlock.


   2. There are some functions which change the internal data
      structure of the hash table, i.e. inserting and deleting
      calls. These processes are protected hash_lock() / hash_unlock().

   
   3. There are some hash functions, i.e. hash_alloc_keylist() which
      take some time to complete, and require the internal hash state
      to be unchanged in the process. These functions call
      hash_lock_reader() and hash_unlock_reader(). There can be many
      concurrent readers, however only one writer. The writer calls
      the functions hash_block_reader() and hash_unblock_reader().


   The functions with locking are as follows:


   Function                	 locking
   -----------------------------------------------------------
   hash_del                	 hash_lock & hash_block_reader
   hash_insert_node        	 hash_lock & hash_block_reader
   hash_alloc_key_sorted_list    hash_lock_reader
   hash_alloc_sorted_keylist     hash_lock_reader
   hash_alloc_keylist            hash_lock_reader
   -----------------------------------------------------------

   Should probably have more reader locks ...
*/


/**
   This function locks the hash for updating. The purpose of the
   function is to protect the hash for an extended period of time. The
   typical usage is illustrated in the documentation of hash_alloc_keylist().
*/

/* 
   I am afraid this has deadlocked - at least once. 
*/



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

/**
   Takes a write_lock.
*/

static void * __hash_get_node(const hash_type *__hash , const char *key, bool abort_on_error) {
  hash_type * hash = (hash_type *) __hash;  /* The net effect is no change - but .... ?? */
  hash_node_type * node = NULL;
  pthread_rwlock_rdlock( &hash->rwlock );
  {
    const uint32_t global_index = hash->hashf(key , strlen(key));
    const uint32_t table_index  = (global_index % hash->size);
    
    node = hash_sll_get(hash->table[table_index] , global_index , key);
    if (node == NULL && abort_on_error) {
      fprintf(stderr,"%s: tried to get from key:%s which does not exist - aborting \n",__func__ , key);
      abort();
    }
  }
  pthread_rwlock_unlock( &hash->rwlock );
}


  
static void hash_del_unlocked__(hash_type *hash , const char *key) {
  const uint32_t global_index = hash->hashf(key , strlen(key));
  const uint32_t table_index  = (global_index % hash->size);
  hash_node_type *node        = hash_sll_get(hash->table[table_index] , global_index , key);
  
  if (node == NULL) {
    fprintf(stderr,"%s: hash does not contain key:%s - aborting \n",__func__ , key);
    abort();
  } else
    hash_sll_del_node(hash->table[table_index] , node);
  
  hash->elements--;
}



void hash_del(hash_type *hash , const char *key) {
  pthread_rwlock_rdlock( &hash->rwlock );
  hash_del_unlocked__(hash , key);
  pthread_rwlock_unlock( &hash->rwlock );
}


void hash_clear(hash_type *hash) {
  pthread_rwlock_rdlock( &hash->rwlock );
  {
    int old_size = hash_get_size(hash);
    if (old_size > 0) {
      char **keyList = hash_alloc_keylist(hash);
      int i;
      for (i=0; i < old_size; i++) {
	hash_del_unlocked__(hash , keyList[i]);
	free(keyList[i]);
      }
      free(keyList);
    }
  }
  pthread_rwlock_unlock( &hash->rwlock );
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

static uint32_t hash_index(const char *key, size_t len) {
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
  hash->__iter_active  = false;
  hash->__iter_keylist = NULL;
  pthread_rwlock_init( &hash->rwlock , NULL);
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


/*****************************************************************/

//iter:static hash_node_type * hash_iter_init(const hash_type *hash) {
//iter:  
//iter:}



//iter:static void hash_iter_free_keylist(hash_type * hash) {
//iter:  int i;
//iter:  if (hash->__iter_keylist != NULL) {
//iter:    for (i=0; i < hash->size; i++) {
//iter:      if (hash->__iter_keylist[i] != NULL)
//iter:	free(hash->__iter_keylist[i]);
//iter:    }
//iter:    free(hash->__iter_keylist);
//iter:  }
//iter:  hash->__iter_keylist = NULL;
//iter:  hash->__iter_active  = false;
//iter:}
//iter:
//iter:
//iter:
//iter:void hash_iter_complete(hash_type * hash) {
//iter:  if (hash->__iter_active) {
//iter:    
//iter:  }
//iter:}
//iter:	
//iter:
//iter:const char * hash_iter_get_next_key(hash_type * hash) {
//iter:  if (hash->__iter_active) {
//iter:    if (hash->__iter_index == hash_get_size(hash)) {
//iter:      hash_iter_complete(hash);
//iter:      return NULL;
//iter:    } else {
//iter:      const char * key = hash->__iter_keylist[hash->__iter_index];
//iter:      hash->__iter_index++;
//iter:      return key;
//iter:    }
//iter:  } else {
//iter:    fprintf(stderr,"%s: called in invalid mode \n",__func__);
//iter:    abort();
//iter:  }
//iter:}
//iter:
//iter:
//iter:const char * hash_iter_get_first_key(hash_type * hash) {
//iter:
//iter:  if (!hash->__iter_active)
//iter:    hash_iter_init(hash);
//iter:  else
//iter:    hash->__iter_index = 0; /* It always allowed to ask for the first - this does *NOT* induce a new locking. */
//iter:  
//iter:  return hash_iter_get_next_key(hash);
//iter:}
//iter:
//iter:
//iter:void * hash_iter_get_first(hash_type * hash , bool *valid) {
//iter:  const char * key = hash_iter_get_first_key(hash);
//iter:  if (key != NULL) {
//iter:    *valid = true;
//iter:    return hash_get(hash , key);
//iter:  } else {
//iter:    *valid = false;
//iter:    return NULL;
//iter:  }
//iter:}
//iter:
//iter:
//iter:void * hash_iter_get_next(hash_type * hash , bool *valid) {
//iter:  const char * key = hash_iter_get_next_key(hash);
//iter:  if (key != NULL) {
//iter:    *valid = true;
//iter:    return hash_get(hash , key);
//iter:  } else {
//iter:    *valid = false;
//iter:    return NULL;
//iter:  }
//iter:}


void hash_free(hash_type *hash) {
  int i;
  for (i=0; i < hash->size; i++) 
    hash_sll_free(hash->table[i]);
  free(hash->table);
  hash_iter_finalize(hash);
  pthread_rwlock_destroy( &hash->rwlock );
  free(hash);
}


void hash_free__(void *void_hash) {
  hash_free((hash_type *) void_hash);
}



/**
   Observe that this function locks the hash table, so it will block
   if the hash is already locked by another thread. If it is already
   locked by this thread - you have an eternal dead lock. 
*/
   
static void hash_insert_node(hash_type *hash , hash_node_type *node) {
  pthread_rwlock_wlock( &hash->rwlock );
  {
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
  }
  pthread_rwlock_unlock( &hash->rwlock );
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











/**
   In a multithreaded program the access to the hash structure should
   be locked when using hash_alloc_keylist:

   hash_lock( hash );
   keys = hash_alloc_keylist( hash );
    
     <Do something with the hash table>
 
   hash_unlock( hash );

   Otherwise one might risk that another thread is updating the hash
   table, invalidating the key_list (in particular if a key is removed
   the program might abort() on hash_get operations).
*/


/**
   This is an internal iterator - NOT related to the xxx_iter_xxx()
   functions.
*/
static hash_node_type * hash_internal_iter_next(const hash_type *hash , const hash_node_type * node) {
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

   
char ** hash_alloc_keylist(hash_type *hash) {
  char **keylist;
  hash_lock_reader( hash );
  {
    int i = 0;
    hash_node_type *node = NULL;
    keylist = calloc(hash->elements , sizeof *keylist);
    {
      uint32_t i = 0;
      while (i < hash->size && hash_sll_empty(hash->table[i]))
	i++;
      
      if (i < hash->size) 
	node = hash_sll_get_head(hash->table[i]);
    }
    
    while (node != NULL) {
      const char *key = hash_node_get_keyref(node); 
      keylist[i] = alloc_string_copy(key);
      node = hash_internal_iter_next(hash , node);
      i++;
    }
  }
  hash_unlock_reader( hash );
  return keylist;
}




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


static char ** __hash_alloc_ordered_keylist(hash_type *hash , int ( hash_get_cmp_value) (const void * )) {    
  int i;
  char **sorted_keylist;
  char **tmp_keylist   = hash_alloc_keylist(hash);
  hash_sort_type * sort_list = hash_alloc_sort_list(hash , (const char **) tmp_keylist);

  for (i = 0; i < hash_get_size(hash); i++) 
    sort_list[i].cmp_value = hash_get_cmp_value( hash_get(hash , sort_list[i].key) );
  
  qsort(sort_list , hash_get_size(hash) , sizeof *sort_list , &hash_sortlist_cmp);
  sorted_keylist = calloc(hash_get_size(hash) , sizeof *sorted_keylist);
  for (i = 0; i < hash_get_size(hash); i++) {
    sorted_keylist[i] = alloc_string_copy(sort_list[i].key);
    free(tmp_keylist[i]);
  }
  free(tmp_keylist);
  hash_free_sort_list(hash , sort_list);
  return sorted_keylist;
}


char ** hash_alloc_sorted_keylist(hash_type *hash , int ( hash_get_cmp_value) (const void *)) {
  char ** key_list;

  hash_lock_reader( hash );
  key_list = __hash_alloc_ordered_keylist(hash , hash_get_cmp_value);
  hash_unlock_reader( hash );

  return key_list;
}




static int key_cmp(const void *_s1 , const void *_s2) {
  const char * s1 = *((const char **) _s1);
  const char * s2 = *((const char **) _s2);
  
  return strcmp(s1 , s2);
}



static char ** __hash_alloc_key_sorted_list(hash_type *hash, int (*cmp) (const void * , const void *)) { 
  char **keylist = hash_alloc_keylist(hash);
  
  qsort(keylist , hash_get_size(hash) , sizeof *keylist , cmp);
  return keylist;
}



char ** hash_alloc_key_sorted_list(hash_type * hash, int (*cmp) (const void *, const void *))
{
  char ** key_list;

  hash_lock_reader( hash );
  key_list =__hash_alloc_key_sorted_list(hash , cmp);
  hash_unlock_reader( hash );

  return key_list;
}



bool hash_key_list_compare(hash_type * hash1, hash_type * hash2)
{
  bool has_equal_keylist;
  int i,size1, size2;
  char **keylist1, **keylist2; 

  size1 = hash_get_size(hash1);
  size2 = hash_get_size(hash2);

  if(size1 != size2) return false;

  keylist1 = hash_alloc_key_sorted_list(hash1, &key_cmp);
  keylist2 = hash_alloc_key_sorted_list(hash2, &key_cmp);

  has_equal_keylist = true;
  for(i=0; i<size1; i++)
  {
    if(strcmp(keylist1[i],keylist2[i]) != 0)
    {
      has_equal_keylist = false;
      break;
    }
  }

  for(i=0; i<size1; i++) {
    free( keylist1[i] );
    free( keylist2[i] );
  }
  free( keylist1 );
  free( keylist2 );
  return has_equal_keylist;
  
}
/******************************************************************/



#undef HASH_GET_SCALAR
#undef HASH_INSERT_SCALAR
#undef HASH_INSERT_ARRAY
#undef HASH_GET_ARRAY_PTR
#undef HASH_NODE_AS
#undef HASH_DEFAULT_SIZE
