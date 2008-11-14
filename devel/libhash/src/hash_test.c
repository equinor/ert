#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <hash.h>




int main(void) {
  hash_type * h = hash_alloc();
  hash_insert_string(h , "Navn"    , "Joakim Hove");
  hash_insert_string(h , "Adresse" , "Henrik Mohnsvei 6");
  hash_insert_string(h , "legning" , "????");

  hash_get_string(h , "Navn");
  hash_get_string(h , "Adresse");
  hash_get_string(h , "legning");
  

  hash_iter_init(h);
  {
    char * key;
    do {
      key = hash_iter_get_next_key(h);
      if (key != NULL) {
	printf("%s -> %s \n",key,hash_get_string(h , key));
      } else printf(" key == NULL \n");
    } while (key != NULL);
  }
  hash_free(h);
}
