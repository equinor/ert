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
#include <errno.h>

#define HASH_DEFAULT_SIZE 16

/**
   This is **THE** hash function - which actually does the hashing.
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



struct hash_struct {
  uint32_t          size;            /* This is the size of the internal table **NOT NOT** the number of elements in the table. */
  uint32_t          elements;        /* The number of elements in the hash table. */
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


static char * alloc_string_copy(const char *src) {
  char *new = malloc(strlen(src) + 1);
  strcpy(new , src);
  return new;
}

/*****************************************************************/
/*                          locking                              */
/*****************************************************************/

static void __hash_deadlock_abort(hash_type * hash) {
  printf("A deadlock condition has been detected in the hash routine - and the program will abort.\n");
  printf("Currently active hash iterator: ");
  if (hash->__iter_active) {
    printf("\n __iter_keys: [");
    for (int i = 0; i < hash->elements; i++) {
      printf("\'%s\'" , hash->__iter_keylist[i]);
      if (i < (hash->elements - 1))
	printf(", ");
    }
    printf("]\n");
    printf("__iter_index: %d \n",hash->__iter_index);
  } else printf(" None ??? \n");
  abort();
}


static void __hash_rdlock(hash_type * hash) {
  int lock_error = pthread_rwlock_tryrdlock( &hash->rwlock );
  if (lock_error != 0) {
    /* We did not get the lock - let us check why: */
    if (lock_error == EDEADLK)
      /* A deadlock is detected - we just abort. */
      __hash_deadlock_abort(hash);
    else 
      /* We ignore all other error conditions than DEADLOCK and just try again. */
      pthread_rwlock_rdlock( &hash->rwlock );
  }
  /* Ok - when we are here - we are guranteed to have the lock. */
}


static void __hash_wrlock(hash_type * hash) {
  int lock_error = pthread_rwlock_trywrlock( &hash->rwlock );
  if (lock_error != 0) {
    /* We did not get the lock - let us check why: */
    if (lock_error == EDEADLK)
      /* A deadlock is detected - we just abort. */
      __hash_deadlock_abort(hash);
    else 
      /* We ignore all other error conditions than DEADLOCK and just try again. */
      pthread_rwlock_rdlock( &hash->rwlock );
  }
  /* Ok - when we are here - we are guranteed to have the lock. */
}


static void __hash_unlock( hash_type * hash) {
  pthread_rwlock_unlock( &hash->rwlock );
}



/*****************************************************************/
/*                    Low level access functions                 */
/*****************************************************************/


static void * __hash_get_node_unlocked(const hash_type *__hash , const char *key, bool abort_on_error) {
  hash_type * hash = (hash_type *) __hash;  /* The net effect is no change - but .... ?? */
  hash_node_type * node = NULL;
  {
    const uint32_t global_index = hash->hashf(key , strlen(key));
    const uint32_t table_index  = (global_index % hash->size);
    
    node = hash_sll_get(hash->table[table_index] , global_index , key);
    if (node == NULL && abort_on_error) {
      fprintf(stderr,"%s: tried to get from key:%s which does not exist - aborting \n",__func__ , key);
      abort();
    }
  }
  return node;
}


/*
  This function looks up a hash_node from the hash. This is the common
  low-level function to get content from the hash. The function takes
  read-lock which is held during execution.
*/

static void * __hash_get_node(hash_type *hash , const char *key, bool abort_on_error) {
  hash_node_type * node;
  __hash_rdlock( hash );
  node = __hash_get_node_unlocked(hash , key , abort_on_error);
  __hash_unlock( hash );
  return node;
}

/**
   This function resizes the hash table when it has become to full.
   The table only grows - this funuction is called from
   __hash_insert_node().
*/
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


/**
   This is the low-level function for inserting a hash node. This
   function takes a write-lock which is held during the execution of
   the function.
*/
   
static void __hash_insert_node(hash_type *hash , hash_node_type *node) {
  __hash_wrlock( hash );
  {
    uint32_t table_index = hash_node_get_table_index(node);
    {
      /*
	If a node with the same key already exists in the table
	it is removed.
      */
      hash_node_type *existing_node = __hash_get_node_unlocked(hash , hash_node_get_keyref(node) , false);
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
  __hash_unlock( hash );
}



/**
   This function deletes a node from the hash_table. Observe that this
   function does *NOT* do any locking - it is the repsonsibility of
   the calling functions: hash_del() and hash_clear() to take the
   necessary write lock.
*/


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



/**
   This functions takes a hash_node and finds the "next" hash node by
   traversing the internal hash structure. Should NOT be confused with
   the other functions providing iterations to user-space.
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



/**
   This is the low level function which traverses a hash table and
   allocates a char ** list of keys. 

   It takes a read-lock which is held during the execution of the
   function. The locking guarantees that the list of keys is valid
   when this function is exited, but the the hash table can be
   subsequently updated.

   If the hash table is empty NULL is returned.
*/

static char ** hash_alloc_keylist__(hash_type *hash) {
  char **keylist;
  __hash_rdlock( hash );
  {
    if (hash->elements > 0) {
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
    } else keylist = NULL;
  }
  __hash_unlock( hash );
  return keylist;
}


/*****************************************************************/
/** 
   The fundamental functions above relate the hash_node
   structure. Here comes a list of functions for inserting managed
   copies of various types.
*/


static void hash_insert_managed_copy(hash_type *hash, const char *key, const void *value_ptr , int value_size) {
  hash_node_type *node; 
  node_data_type *node_data = node_data_alloc(value_size , value_ptr);
  node = hash_node_alloc_new(key , node_data , node_data_copyc , node_data_free , hash->hashf , hash->size);
  __hash_insert_node(hash , node);

  /* This only frees the container - actual storage is freed when the hash table is deleted */
  free(node_data);
}


#define HASH_INSERT_SCALAR(FUNC,TYPE) \
void FUNC(hash_type *hash , const char *key , TYPE value) {     \
  hash_insert_managed_copy(hash , key , &value , sizeof value); \
}

#define HASH_INSERT_ARRAY(FUNC,TYPE)                                   \
void FUNC(hash_type *hash, const char *key , TYPE *value, int SIZE) {  \
  hash_insert_managed_copy(hash , key , value , SIZE * sizeof *value); \
}

void hash_insert_string(hash_type *hash, const char *key , const char *value) {
  hash_insert_managed_copy(hash , key , value , strlen(value) + 1);
}

const char * hash_get_string(const hash_type *hash , const char *key) {
  node_data_type *node_data = hash_get(hash , key);
  if (node_data != NULL)
    return node_data_get_data(node_data);
  else
    return NULL;
}

HASH_INSERT_SCALAR(hash_insert_int          , int)
HASH_INSERT_SCALAR(hash_insert_double       , double)
HASH_INSERT_ARRAY (hash_insert_int_array    , int)
HASH_INSERT_ARRAY (hash_insert_double_array , double)


#define HASH_GET_SCALAR(FUNC,TYPE) \
TYPE FUNC (const hash_type *hash,  const char *key) { \
   node_data_type *node_data = hash_get(hash , key); \
   return *((TYPE *) node_data_get_data(node_data)); \
}


#define HASH_GET_ARRAY_PTR(FUNC,TYPE)\
TYPE * FUNC(const hash_type * hash, const char *key) { \
   node_data_type *node_data = hash_get(hash , key);   \
   return ((TYPE *) node_data_get_data(node_data));    \
}


HASH_GET_SCALAR(hash_get_int    , int)
HASH_GET_SCALAR(hash_get_double , double)
HASH_GET_ARRAY_PTR(hash_get_double_ptr , double)
HASH_GET_ARRAY_PTR(hash_get_int_ptr    , int)

/*****************************************************************/

void hash_del(hash_type *hash , const char *key) {
  __hash_wrlock( hash );
  hash_del_unlocked__(hash , key);
  __hash_unlock( hash );
}


void hash_clear(hash_type *hash) {
  __hash_wrlock( hash );
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
  __hash_unlock( hash );
}


void * hash_get(const hash_type *hash , const char *key) {
  hash_node_type * node = __hash_get_node(hash , key , true);
  return hash_node_value_ptr(node);
}




/******************************************************************/
/*                     Allocators                                 */
/******************************************************************/


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
  
  if (pthread_rwlock_init( &hash->rwlock , NULL) != 0) {
    fprintf(stderr , "%s: failed to initialize rwlock \n",__func__);
    abort();
  }
  
  return hash;
}


hash_type * hash_alloc() {
  return __hash_alloc(HASH_DEFAULT_SIZE , 0.50 , hash_index);
}


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


char ** hash_alloc_keylist(hash_type *hash) {
  return hash_alloc_keylist__(hash);
}


/*****************************************************************/
/** 
    The standard functions for inserting an entry in the hash table:

    hash_insert_copy(): The hash table uses copyc() to make a copy of
         value, which is inserted in the table. This means that the
         calling scope is free to do whatever it wants with value; and
         is also responsible for freeing it. The hash_table takes
         responsibility for freeing it's own copy.

    hash_insert_hash_owned_ref(): The hash table takes ownership of
         'value', in the sense that the hash table will call the
         'destructor' del() on value when the node is deleted.

    hash_insert_ref(): The hash table ONLY contains a pointer to
         value; the calling scope retains full ownership to
         value. When the hash node is deleted, the hash implementation
         will just drop the reference on the floor.
*/


void hash_insert_copy(hash_type *hash , const char *key , const void *value , copyc_type *copyc , del_type *del) {
  hash_node_type *node;
  if (copyc == NULL || del == NULL) {
    fprintf(stderr,"%s: must provide copy constructer and delete operator for insert copy - aborting \n",__func__);
    abort();
  }
  node = hash_node_alloc_new(key , value , copyc , del , hash->hashf , hash->size);
  __hash_insert_node(hash , node);
}


/**
  This function will insert a reference "value" with key "key"; when
  the key is deleted - either by an explicit call to hash_del(), or
  when the complete hash table is free'd with hash_free(), the
  destructur 'del' is called with 'value' as argument.

  It is importand to realize that when elements are inserted into a
  hash table with this function the calling scope gives up
  responsibility of freeing the memory pointed to by value.
*/

void hash_insert_hash_owned_ref(hash_type *hash , const char *key , const void *value , del_type *del) {
  hash_node_type *node;
  if (del == NULL) {
    fprintf(stderr,"%s: must provide delete operator for insert hash_owned_ref - aborting \n",__func__);
    abort();
  }
  node = hash_node_alloc_new(key , value , NULL , del , hash->hashf , hash->size);
  __hash_insert_node(hash , node);
}


void hash_insert_ref(hash_type *hash , const char *key , const void *value) {
  hash_node_type *node;
  node = hash_node_alloc_new(key , value , NULL , NULL , hash->hashf , hash->size );
  __hash_insert_node(hash , node);
}



bool hash_has_key(const hash_type *hash , const char *key) {
  if (__hash_get_node(hash , key , false) == NULL)
    return false;
  else
    return true;
}


int hash_get_size(const hash_type *hash) { 
  return hash->elements; 
}



/******************************************************************/
/** 
   Here comes a list of functions for allocating keylists which have
   been sorted in various ways.
*/
   

static hash_sort_type * hash_alloc_sort_list(const hash_type *hash ,
					     const char **keylist) { 

  int i; hash_sort_type * sort_list = calloc(hash_get_size(hash) , sizeof * sort_list); 
  for (i=0; i < hash_get_size(hash); i++) 
    sort_list[i].key = alloc_string_copy(keylist[i]);
  
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

  key_list = __hash_alloc_ordered_keylist(hash , hash_get_cmp_value);

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

  key_list =__hash_alloc_key_sorted_list(hash , cmp);

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
/*                        Iteration                               */
/******************************************************************/

/**
   The general usage of the iteration interface is as follows:

     1. Initialize the iteration - this is either done with an
        explicit call to hash_iter_init(), or by calling
        hash_get_first_key() / hash_get_first_value(), which will call
        hash_iter_init().


     2. Iterate through the hash with hash_iter_get_next_key() and
        hash_iter_get_next_value(). 


     3. The hash signals that the iteration is complete by returning
        NULL - both for _next_key() and _next_value(). The latter in
        addition has a pointer to bool which will be set to true when
        the iteration is complete.
	
	When the iteration is complete the hash will automagically
	call hash_iter_finalize() which will release the resources
	held internally to support the iteration.


     4. If the hash has been iterated all the way through there is no
        need to explicitly finalize the iteration, but if you quit the
        iteration before it is complete you MUST EXPLICITLY CALL
        hash_iter_finalize(). Otherwise the implementation WILL
        DEADLOCK.
	
	It is always OK to call hash_iter_finalize().


Deadlock
--------
The hash grabs a read_lock when the iteration is initialized, this
lock is held all the time until the iteration is complete. If you exit
the iteration before it is complete, the implementation will deadlock
on the next hash_insert() call.

This will deadlock:

   hash_iter_init(hash);
   do {
      char * key = hash_iter_get_next_key(hash);
   } while (key != NULL && strcmp(key , "HEI") != 0)  -- This will abort the iteration when/if the key "HEI" is found.
   ....
   ....
   hash_insert(hash , key , value);  -- The readlock from the iteration is still present => deadlock.
 
To ensure against deadlock in this case, you must have a call hash_iter_finalize() after the while() statement.



key/value:
---------
Observe that __get_next_key() and __get_next_value() share the same internal state, so with code like this:

  key1   = hash_iter_get_next_key(hash);
  value2 = hash_iter_get_next_value(hash);

value2 will **NOT** be the value corresponding to key1.


Recursive : careful with that!

*/


/**
   This function will free all the internal resources related to an
   iteration. It is always OK to call hash_iter_finalize.

   Observe that when an iteration has completed naturally it is
   automatically finalized. The only situation where it is necessary
   to explicitly call hash_iter_finalize() is when the iteration has
   been explicitly terminated before completion.
*/





void hash_iter_finalize(hash_type * hash) {
  if (hash->__iter_active) {
    if (hash->elements > 0) {
      for (int ikey = 0; ikey < hash->elements; ikey++)
	free(hash->__iter_keylist[ikey]);
      free(hash->__iter_keylist);
    }
    hash->__iter_keylist = NULL;
    hash->__iter_active  = false;
    __hash_unlock( hash );
  }
}



/**
   This function initializes an iteration. That means the following:

     1. Finalize any currently active iterators.
     2. Take read-lock - this is held all the time until we call hash_iter_finalize().
     3. Initilize the internal list of keys - __iter_keylist.
     4. Reset the __iter_index.

*/
     
void hash_iter_init(hash_type * hash) {
  __hash_rdlock( hash ); /* Just for this function. */
  if (hash->__iter_active)
    hash_iter_finalize(hash);
  
  __hash_rdlock( hash ); /* Until _finalize */
  {
    hash->__iter_keylist = hash_alloc_keylist__(hash);
    hash->__iter_index   = 0;
    hash->__iter_active  = true;
  }
  __hash_unlock( hash );
}



/**
   This functions gets the next key in an ongoing iteration. If there
   is not an ongoing iteration, the function will fail hard. If the
   iteration is complete, the function will return NULL, and finalize
   the iteration.

   Observe that the calling scope should *NOT* retain a reference to
   the keys returned from hash_iter_get_next_key() - they will be
   free'd when the iteration is complete. In that case the calling
   scope should take a copy of the key.
*/

const char * hash_iter_get_next_key(hash_type * hash) {
  char * key = NULL;
  if (hash->__iter_active) {
    if (hash->__iter_index == hash->elements)  /* We are through */ 
      hash_iter_finalize(hash);
    else {
      key = hash->__iter_keylist[hash->__iter_index];
      hash->__iter_index++;
    }
    return key;
  } else {
    fprintf(stderr,"%s: no iteration active - aborting \n",__func__);
    abort();
  }
}


/**
   This function will return the next value (by calling get_next_key
   first). If the iteration is complete the function will return NULL.

   If you have inserted NULL in the hash table, you can not use that
   return value to signal a complete iteration. If you pass in a (bool
   *) to complete that will be set to true when the iteration is
   complete.
*/
  
void * hash_iter_get_next_value(hash_type * hash , bool * complete) {
  const char * key = hash_iter_get_next_key(hash);
  void       * value;
  if (key == NULL) {
    if (complete != NULL)
      *complete = true;
    value  = NULL;
  } else {
    if (complete != NULL)
      *complete = false;
    value = hash_get(hash , key);
  }
  return value;
}


/**
   This function will initialize an iteration, and then return the
   first key.
*/
const char * hash_iter_get_first_key(hash_type * hash) {
  hash_iter_init( hash );
  return hash_iter_get_next_key(hash);
}


/**
   This function will initialize an iteration, and then return the
   first value.
*/
void * hash_iter_get_first_value(hash_type * hash, bool * complete) {
  hash_iter_init( hash );
  return hash_iter_get_next_value(hash , complete);
}



#undef HASH_GET_SCALAR
#undef HASH_INSERT_SCALAR
#undef HASH_INSERT_ARRAY
#undef HASH_GET_ARRAY_PTR
#undef HASH_NODE_AS
#undef HASH_DEFAULT_SIZE
