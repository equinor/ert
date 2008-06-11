#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <hash.h>




int main(void) {
  const char *s1 = "555";
  const char *s2 = "555";
  const char *s3 = "555";
  const char *s4 = "555";
  const char *s5 = "555";

  hash_type      *hash;
  
  hash = hash_alloc();

  hash_insert_ref(hash , "HEI" , s1);
  hash_insert_ref(hash , "NAVN" , s2);

  hash_insert_ref(hash , "iJA1" , s1);
  hash_insert_ref(hash , "iJA2" , s2);
  hash_insert_ref(hash , "iJA3" , s3);
  hash_insert_ref(hash , "iJA4" , s4);
  hash_insert_ref(hash , "iJA5" , s5);

  {
    const char *kw;
    kw = hash_iter_get_first_key(hash);
    while (kw != NULL) {
      printf("kw: %s \n",kw);
      kw = hash_iter_get_next_key(hash);
    }
  }
  
  printf("-----------------------------------------------------------------\n");
  
  {
    bool cont;
    const char *s;
    s = hash_iter_get_first(hash , &cont);
    while (cont) {
      printf("Har: %s \n",s);
      s = hash_iter_get_next(hash , &cont);
    }
  }


  hash_free(hash);
  
  return 0;
}
