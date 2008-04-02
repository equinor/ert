#include <stdlib.h>
#include <hash.h>
#include <stdio.h>
#include <stdbool.h>
#include <set.h>
#include <string.h>


struct set_struct {
  hash_type * key_hash;
};




set_type * set_alloc(int size, const char ** keyList) {
  set_type * set = malloc(sizeof * set);
  set->key_hash  = hash_alloc(10);
  {
    int ikey;
    for (ikey = 0; ikey < size; ikey++)
      set_add_key(set , keyList[ikey]);
  }
  return set;
}


set_type * set_alloc_empty() { return set_alloc(0 , NULL); }


bool set_add_key(set_type * set, const char * key) {
  if (hash_has_key(set->key_hash , key))
    return false;
  else {
    hash_insert_int(set->key_hash , key , 1);
    return true;
  }
}

bool set_has_key(set_type * set, const char * key) {
  return hash_has_key(set->key_hash, key);
}


int set_get_size(const set_type *set) { return hash_get_size(set->key_hash); }


char ** set_alloc_keylist(const set_type * set) { return hash_alloc_keylist(set->key_hash); }


void set_free(set_type * set) {
  hash_free(set->key_hash);
  free(set);
}




/* Wrong(??) depenendency between libutil and libhash ... */
static void __fwrite_string(const char * s, FILE *stream) {
  int len = 0;
  if (s != NULL) {
    len = strlen(s);
    fwrite(&len , sizeof len , 1       , stream);
    fwrite(s    , 1          , len + 1 , stream);
  } else
    fwrite(&len , sizeof len , 1       , stream);
}


static char * __fread_alloc_string(FILE *stream) {
  int len;
  char *s = NULL;
  fread(&len , sizeof len , 1 , stream );
  if (len > 0) {
    s = malloc(len + 1 );
    fread(s , 1 , len + 1 , stream );
  } 
  return s;
}



void set_fwrite(const set_type * set, FILE * stream) {
  char ** key_list = set_alloc_keylist(set);
  int size = set_get_size(set);
  int i;
  fwrite(&size , sizeof size , 1 , stream);
  for (i=0; i < size; i++) {
    __fwrite_string(key_list[i] , stream);
    free(key_list[i]);
  }
  free(key_list);
}


void set_fread(set_type * set , FILE * stream) {
  int size, i;
  fread(&size , sizeof size , 1 , stream);
  for (i=0; i < size; i++) {
    char * key = __fread_alloc_string(stream);
    set_add_key(set , key);
    free(key);
  }
}



set_type * set_fread_alloc(FILE * stream) {
  set_type * set = set_alloc_empty();
  set_fread(set , stream);
  return set;
}

