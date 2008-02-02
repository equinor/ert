#include <stdlib.h>
#include <config.h>
#include <string.h>
#include <hash.h>
#include <util.h>



#define if_strcmp(v,s) if (strcmp(s,v) == 0)


void config_parse(const char * config_file , hash_type * config_hash) {
  FILE * stream = util_fopen(config_file , "r");
  char * token , * upper_token;
  
  token       = util_fscanf_alloc_token(stream);
  if (token != NULL) {
    upper_token = util_alloc_string_copy(token);
    util_strupr(upper_token);
    if_strcmp(upper_token , "SIZE") {
      int value;
      if (util_fscanf_int(stream , &value))    
	hash_insert_int(config_hash , "SIZE" , value);
      else {
	fprintf(stderr,"%s: expected integer after keyword:%s  - aborting \n",__func__ , token);
	abort();
      }
    } else {
      fprintf(stderr,"%s: keyword: %s not recognized by config parser - aborting \n",__func__ , token);
      abort();
    }
      free(upper_token);
  }
  free(token);
}

#undef if_strcmp
