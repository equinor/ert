#include <stdlib.h>
#include <hash.h>
#include <stdio.h>
#include <stdbool.h>
#include <set.h>


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


int set_get_size(const set_type *set) { return hash_get_size(set->key_hash); }

char ** set_alloc_keylist(const set_type * set) { return hash_alloc_keylist(set->key_hash); }


void set_free(set_type * set) {
  hash_free(set->key_hash);
  free(set);
}








