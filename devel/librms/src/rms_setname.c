#include <stdlib.h>
#include <stdio.h>
#include <rms_file.h>
#include <rms_tagkey.h>
#include <rms_stats.h>
#include <list.h>
#include <string.h>

int main (int argc , char **argv) {
  int i;
  char *name;
  int name_length;
  argv++;
  argc--;
  
  name        = argv[0];
  name_length = strlen(name) + 1;

  argv++;
  argc--;
  for (i = 0; i < argc; i++) {
    rms_file_type *file = rms_file_alloc(argv[i] , false);
    rms_tagkey_type *tagkey;
    rms_tag_type    *tag;
    rms_file_fread(file);

    tag = rms_file_get_tag_ref(file , "parameter" , NULL , NULL , true);
    tagkey = rms_tag_get_key(tag , "name");
    rms_tagkey_manual_realloc_data(tagkey , name_length);
    rms_tagkey_set_data(tagkey , name);
    rms_file_fwrite(file , "parameter");
    rms_file_free(file);
  }

  return 0;
}

